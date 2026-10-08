#!/usr/bin/env python
# coding: utf-8
import torch
import numpy as np
import os
import random
from dataset import load_raw_data
from trainer import Trainer
from trigger_locator import TriggerLocator
import yaml
from easydict import EasyDict as edict


def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parser_args():
    default_config = yaml.load(open('configs/default_config.yaml', 'r'), Loader=yaml.FullLoader)
    config = yaml.load(open('configs/train_config.yaml'), Loader=yaml.FullLoader)['Train']

    config['Dataset'] = default_config['Dataset'][config['dataset']]
    config['Dataset']['date_col'] = config.get('date_col', 'date')
    config['Dataset']['target_col'] = config.get('target_col', 'OT')
    config['Dataset']['train_ratio'] = config.get('train_ratio', 0.8)
    config['Dataset']['len_input'] = config.get('seq_len', config['Dataset'].get('len_input', 288))
    config['Dataset']['num_for_predict'] = config.get('pred_len', config['Dataset'].get('num_for_predict', 96))
    config['Dataset']['points_per_day'] = config.get('points_per_day', config['Dataset'].get('points_per_day', 96))

    if 'sample_mode' not in config:
        config['sample_mode'] = 'day_aligned' if bool(config.get('day_aligned', False)) else 'generic'
    config['sample_mode'] = str(config['sample_mode']).lower()
    if 'sample_stride' not in config:
        config['sample_stride'] = 1
    config['day_aligned'] = config['sample_mode'] == 'day_aligned'
    config['Dataset']['sample_mode'] = config['sample_mode']
    config['Dataset']['sample_stride'] = config['sample_stride']

    config['Target_Pattern'] = default_config['Target_Pattern'][config['pattern_type']]

    config['Model'] = default_config['Model'][config['model_name']]
    config['Surrogate'] = default_config['Model'][config['surrogate_name']]

    if 'use_trigger_locator' not in config:
        config['use_trigger_locator'] = False
    if 'target_start_in_pred' not in config:
        config['target_start_in_pred'] = 0

    if 'locator_epochs' not in config:
        config['locator_epochs'] = 50
    if 'locator_batch_size' not in config:
        config['locator_batch_size'] = 64
    if 'locator_lr' not in config:
        config['locator_lr'] = 1e-4
    if 'attn_threshold_ratio' not in config:
        config['attn_threshold_ratio'] = 0.7
    if 'min_trigger_len' not in config:
        config['min_trigger_len'] = 4
    if 'max_trigger_len' not in config:
        config['max_trigger_len'] = 16
    if 'locator_use_selected_vars_only' not in config:
        config['locator_use_selected_vars_only'] = False

    if 'multi_trigger' not in config:
        config['multi_trigger'] = False
    if 'max_trigger_segments' not in config:
        config['max_trigger_segments'] = 3
    if 'min_segment_gap' not in config:
        config['min_segment_gap'] = 4
    if 'multi_peak_ratio' not in config:
        config['multi_peak_ratio'] = 0.75

    if 'train_clean_victim' not in config:
        config['train_clean_victim'] = False
    if 'visualize_trigger_propagation' not in config:
        config['visualize_trigger_propagation'] = False
    if 'trigger_vis_num_samples' not in config:
        config['trigger_vis_num_samples'] = 5
    if 'trigger_vis_split' not in config:
        config['trigger_vis_split'] = 'train'
    if 'trigger_vis_dir' not in config:
        config['trigger_vis_dir'] = './TriggerPropagation'

    if 'save_modal_embeddings' not in config:
        config['save_modal_embeddings'] = False
    if 'modal_embed_num_samples' not in config:
        config['modal_embed_num_samples'] = 200
    if 'modal_embed_split' not in config:
        config['modal_embed_split'] = 'train'
    if 'modal_embed_dir' not in config:
        config['modal_embed_dir'] = './ModalEmbeddings'
    if 'modal_embed_seed' not in config:
        config['modal_embed_seed'] = 1
    if 'modal_embed_batch_size' not in config:
        config['modal_embed_batch_size'] = 32

    if 'timevlm_modal_ablation' not in config:
        config['timevlm_modal_ablation'] = False
    if 'timevlm_modal_mask' not in config:
        config['timevlm_modal_mask'] = 'T1V1L1'
    config['timevlm_modal_mask'] = str(config['timevlm_modal_mask']).upper()
    if 'log_stage2_loss_detail' not in config:
        config['log_stage2_loss_detail'] = False
    if 'stage2_loss_log_dir' not in config:
        config['stage2_loss_log_dir'] = './Stage2LossLogs'
    if 'log_stage1_loss_detail' not in config:
        config['log_stage1_loss_detail'] = False
    if 'stage1_loss_log_dir' not in config:
        config['stage1_loss_log_dir'] = './Stage1LossLogs'

    if 'stage1_probe_grad_conflict' not in config:
        config['stage1_probe_grad_conflict'] = False
    if 'stage1_grad_probe_batches' not in config:
        config['stage1_grad_probe_batches'] = 1
    if 'use_trigger_carrier' not in config:
        config['use_trigger_carrier'] = False
    if 'trigger_carrier_alpha' not in config:
        config['trigger_carrier_alpha'] = 0.5
    if 'carrier_high_freq_ratio' not in config:
        config['carrier_high_freq_ratio'] = 0.3
    if 'carrier_learnable' not in config:
        config['carrier_learnable'] = False
    if 'lam_cgh' not in config:
        config['lam_cgh'] = 0.1
    if 'lam_mag' not in config:
        config['lam_mag'] = 0.05
    if 'cgh_rho' not in config:
        config['cgh_rho'] = 1.0
    if 'load_legacy_attacker_ckpt' not in config:
        config['load_legacy_attacker_ckpt'] = True
    if 'allow_cross_surrogate_legacy_attacker' not in config:
        config['allow_cross_surrogate_legacy_attacker'] = False

    if 'save_figure_data_csv' not in config:
        config['save_figure_data_csv'] = False

    config = edict(config)
    return config


def select_attack_vars(train_data_seq, target_idx, config=None):
    extra_ratio = float(getattr(config, 'extra_atk_ratio', 0.0))

    num_vars = train_data_seq.shape[1]
    feature_indices = [i for i in range(num_vars) if i != target_idx]

    extra_num = int(np.floor(len(feature_indices) * extra_ratio + 0.5))
    extra_num = max(0, min(extra_num, len(feature_indices)))

    if extra_num == 0:
        atk_vars = [int(target_idx)]
        print(f'[Attack Vars] Only OT is attacked: {atk_vars}')
        return atk_vars

    target_series = train_data_seq[:, target_idx]
    corr_scores = []

    for feat_idx in feature_indices:
        feat_series = train_data_seq[:, feat_idx]

        if np.std(feat_series) < 1e-8 or np.std(target_series) < 1e-8:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_series, target_series)[0, 1]
            if np.isnan(corr):
                corr = 0.0

        corr_scores.append((feat_idx, abs(corr), corr))

    corr_scores.sort(key=lambda x: x[1], reverse=True)

    selected_extra = [item[0] for item in corr_scores[:extra_num]]

    atk_vars = [int(target_idx)] + [int(i) for i in selected_extra]

    print(f'[Attack Vars] target_idx={target_idx}, extra_ratio={extra_ratio}, '
          f'selected_extra={selected_extra}, final_atk_vars={atk_vars}')

    return atk_vars


def _build_timevlm_content(config):
    dataset_name = str(getattr(config.Dataset, 'dataset_name', config.dataset))
    data_file = str(getattr(config.Dataset, 'data_filename', dataset_name))
    key = dataset_name.lower()

    descriptions = {
        'ettm1': 'ETTm1 records electricity transformer temperature and related load features at 15-minute frequency.',
        'ettm2': 'ETTm2 records electricity transformer temperature and related load features at 15-minute frequency.',
        'etth1': 'ETTh1 records electricity transformer temperature and related load features at hourly frequency.',
        'etth2': 'ETTh2 records electricity transformer temperature and related load features at hourly frequency.',
        'electricity': 'Electricity records multivariate electricity consumption time series from multiple clients.',
        'exchange_rate': 'Exchange rate records daily exchange rates across multiple countries.',
        'traffic': 'Traffic records multivariate road occupancy time series from traffic sensors.',
        'weather': 'Weather records multivariate meteorological time series including temperature and related variables.',
        'stock_data': 'Stock data records multivariate financial market time series.',
        'pems03': 'PEMS03 records multivariate traffic sensor time series from California highway networks.',
        'pems04': 'PEMS04 records multivariate traffic sensor time series from California highway networks.',
        'pems08': 'PEMS08 records multivariate traffic sensor time series from California highway networks.',
    }

    if key in descriptions:
        return descriptions[key]

    file_key = os.path.splitext(os.path.basename(data_file))[0].lower()
    return descriptions.get(file_key, f'{dataset_name} is a custom multivariate time series forecasting dataset.')


def _sync_single_model_cfg(model_cfg, config, num_vars):
    model_cfg['c_out'] = num_vars
    model_cfg['enc_in'] = num_vars
    model_cfg['dec_in'] = num_vars
    model_cfg['seq_len'] = int(config.seq_len)
    model_cfg['pred_len'] = int(config.pred_len)

    if 'label_len' in model_cfg:
        model_cfg['label_len'] = max(1, int(config.pred_len) // 2)

    model_cfg['task_name'] = 'long_term_forecast'
    freq = 't' if int(config.points_per_day) == 96 else 'h'
    model_cfg['freq'] = freq

    if 'embed' not in model_cfg:
        model_cfg['embed'] = 'timeF'

    if 'vlm_type' in model_cfg:
        model_cfg['periodicity'] = int(config.points_per_day)
        model_cfg['use_gpu'] = torch.cuda.is_available()
        model_cfg['gpu'] = 0
        model_cfg['content'] = config.get('timevlm_content', _build_timevlm_content(config))
        model_cfg['vlm_model_path'] = model_cfg.get(
            'vlm_model_path', './pretrained_models/clip-vit-base-patch32'
        )
        model_cfg['save_images'] = bool(model_cfg.get('save_images', False))

    if 'llm_model' in model_cfg:
        model_cfg['content'] = config.get('timellm_content', _build_timevlm_content(config))
        model_cfg['llm_model_path'] = model_cfg.get('llm_model_path', './pretrained_models/gpt2')

    if 'num_cma_heads' in model_cfg:
        model_cfg['num_nodes'] = num_vars
        model_cfg['dataset_name'] = str(getattr(config.Dataset, 'dataset_name', config.dataset))
        model_cfg['content'] = config.get('t3time_content', _build_timevlm_content(config))
        model_cfg['llm_model_path'] = model_cfg.get('llm_model_path', './pretrained_models/gpt2')

    return model_cfg


def sync_model_cfg(config, num_vars):
    config.Model = _sync_single_model_cfg(config.Model, config, num_vars)
    config.Surrogate = _sync_single_model_cfg(config.Surrogate, config, num_vars)


def build_ckpt_paths(config):
    data_name = getattr(config.Dataset, 'data_filename', config.dataset)
    data_name = os.path.splitext(os.path.basename(str(data_name)))[0]

    ckpt_dir = './checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)

    stage1_modal_suffix = ''
    if bool(getattr(config, 'timevlm_modal_ablation', False)) and config.surrogate_name == 'TimeVLM':
        stage1_modal_suffix = f"_modal{str(getattr(config, 'timevlm_modal_mask', 'T1V1L1')).upper()}"
    attacker_ckpt = os.path.join(
        ckpt_dir,
        f'attacker_{data_name}_sur{config.surrogate_name}{stage1_modal_suffix}.pth'
    )
    legacy_attacker_ckpt = os.path.join(ckpt_dir, f'attacker_{data_name}.pth')

    modal_suffix = ''
    if bool(getattr(config, 'timevlm_modal_ablation', False)) and config.model_name == 'TimeVLM':
        modal_suffix = f"_{str(getattr(config, 'timevlm_modal_mask', 'T1V1L1')).upper()}"
    poisoned_victim_ckpt = os.path.join(ckpt_dir, f'victim_poisoned_{data_name}_{config.model_name}{modal_suffix}.pth')
    clean_victim_ckpt = os.path.join(ckpt_dir, f'victim_clean_{data_name}_{config.model_name}{modal_suffix}.pth')

    return attacker_ckpt, legacy_attacker_ckpt, poisoned_victim_ckpt, clean_victim_ckpt


def main(config):
    gpuid = str(config.gpuid)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuid
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
    print("CUDA:", USE_CUDA, DEVICE)

    seed_torch()

    data_config = config.Dataset

    (
        train_data_seq,
        test_data_seq,
        train_mean,
        train_std,
        train_timestamps,
        test_timestamps,
        target_idx,
        data_columns,
    ) = load_raw_data(data_config)

    sync_model_cfg(config, num_vars=train_data_seq.shape[1])

    atk_vars = select_attack_vars(
        train_data_seq=train_data_seq,
        target_idx=target_idx,
        config=config,
    )
    atk_vars = torch.tensor(atk_vars, dtype=torch.long, device=DEVICE)
    print('shape of attacked_variables', atk_vars.shape)

    target_pattern = torch.tensor(config.Target_Pattern, dtype=torch.float32, device=DEVICE)

    if target_pattern.ndim == 1:
        config.target_len = int(target_pattern.shape[0])
    elif target_pattern.ndim == 2:
        config.target_len = int(target_pattern.shape[0])
    else:
        raise ValueError(f'Unsupported target_pattern shape: {target_pattern.shape}')

    config.pattern_len = int(config.target_len)

    if bool(config.use_trigger_locator):
        print('=' * 20, ' [ Trigger Locator ] ', '=' * 20)
        locator = TriggerLocator(
            config=config,
            device=DEVICE,
            train_data_seq=train_data_seq,
            test_data_seq=test_data_seq,
            train_timestamps=train_timestamps,
            test_timestamps=test_timestamps,
            target_idx=target_idx,
            data_columns=data_columns,
        )
        trigger_info = locator.locate()

        config.trigger_start_in_hist = int(trigger_info['trigger_start_in_hist'])
        config.trigger_len = int(trigger_info['trigger_len'])

        config.multi_trigger = (trigger_info.get('mode', 'single') == 'multi')
        config.trigger_segments = trigger_info.get(
            'segments',
            [(config.trigger_start_in_hist, config.trigger_start_in_hist + config.trigger_len)]
        )
        config.main_trigger_segment = trigger_info.get(
            'main_segment',
            (config.trigger_start_in_hist, config.trigger_start_in_hist + config.trigger_len)
        )

        print(f'[Main] use trigger locator: '
              f'trigger_start_in_hist={config.trigger_start_in_hist}, '
              f'trigger_len={config.trigger_len}, '
              f'target_start_in_pred={config.target_start_in_pred}, '
              f'target_len={config.target_len}, '
              f'multi_trigger={config.multi_trigger}, '
              f'trigger_segments={config.trigger_segments}')
    else:
        config.trigger_start_in_hist = int(config.seq_len - config.trigger_len)
        config.multi_trigger = False
        config.trigger_segments = [(config.trigger_start_in_hist, config.trigger_start_in_hist + config.trigger_len)]
        config.main_trigger_segment = config.trigger_segments[0]
        print(f'[Main] locator disabled, use default trigger_start_in_hist={config.trigger_start_in_hist}')

    attacker_ckpt, legacy_attacker_ckpt, poisoned_victim_ckpt, clean_victim_ckpt = build_ckpt_paths(config)

    exp_trainer = Trainer(
        config,
        atk_vars,
        target_pattern,
        train_mean,
        train_std,
        train_data_seq,
        test_data_seq,
        train_timestamps,
        test_timestamps,
        DEVICE,
        target_idx
    )

    # Inject frequency carrier if enabled
    if bool(getattr(config, 'use_trigger_carrier', False)):
        from attack import FrequencyCarrier
        num_vars = train_data_seq.shape[1]
        carrier = FrequencyCarrier(
            seq_len=config.seq_len,
            num_vars=num_vars,
            high_freq_ratio=float(getattr(config, 'carrier_high_freq_ratio', 0.3)),
            learnable=bool(getattr(config, 'carrier_learnable', False)),
            device=DEVICE
        )
        exp_trainer.attacker.trigger_carrier = carrier
        exp_trainer.attacker.trigger_carrier_alpha = float(getattr(config, 'trigger_carrier_alpha', 0.5))
        print(f'[Main] Enabled trigger carrier: num_vars={num_vars}, high_freq_ratio={carrier.high_freq_ratio}, '
              f'learnable={getattr(config, "carrier_learnable", False)}, alpha={exp_trainer.attacker.trigger_carrier_alpha}')
    else:
        exp_trainer.attacker.trigger_carrier = None
        print('[Main] Trigger carrier disabled')

    attacker_load_path = None
    if os.path.exists(attacker_ckpt):
        attacker_load_path = attacker_ckpt
    elif bool(getattr(config, 'load_legacy_attacker_ckpt', True)) and os.path.exists(legacy_attacker_ckpt):
        can_load_legacy = (
            config.surrogate_name == 'PatchTST'
            or bool(getattr(config, 'allow_cross_surrogate_legacy_attacker', False))
        )
        if can_load_legacy:
            attacker_load_path = legacy_attacker_ckpt
        else:
            print(
                f'[CKPT] Legacy attacker ckpt exists at {legacy_attacker_ckpt}, '
                f'but surrogate_name={config.surrogate_name}; skip it and train Stage 1.'
            )

    if attacker_load_path is not None:
        state = torch.load(attacker_load_path, map_location=DEVICE)
        exp_trainer.load_attacker(state)
        print('load attacker from', attacker_load_path)
    else:
        print('=' * 20, ' [ Stage 1 ] ', '=' * 20)
        print('start training surrogate model and attacker')
        exp_trainer.train()

        state = exp_trainer.save_attacker()
        torch.save(state, attacker_ckpt)
        print(f'[CKPT] saved attacker ckpt to {attacker_ckpt}')

    if bool(config.visualize_trigger_propagation):
        print('=' * 20, ' [ Trigger Propagation Visualization ] ', '=' * 20)
        from visualization import save_trigger_propagation_visuals
        save_trigger_propagation_visuals(
            trainer=exp_trainer,
            config=config,
            num_samples=config.trigger_vis_num_samples,
            split=config.trigger_vis_split,
            save_root=config.trigger_vis_dir,
        )

    if bool(config.save_modal_embeddings):
        print('=' * 20, ' [ Save Modal Embeddings ] ', '=' * 20)
        from modal_embeddings import save_modal_embeddings
        save_modal_embeddings(
            trainer=exp_trainer,
            config=config,
            num_samples=config.modal_embed_num_samples,
            split=config.modal_embed_split,
            save_root=config.modal_embed_dir,
            seed=config.modal_embed_seed,
        )

    print('=' * 20, ' [ Stage 2 - Poisoned Victim ] ', '=' * 20)
    print('start evaluating attack performance on a new poisoned victim model')
    exp_trainer.test(save_poisoned_ckpt_path=poisoned_victim_ckpt)

    if bool(config.train_clean_victim):
        print('=' * 20, ' [ Stage 2 - Clean Victim ] ', '=' * 20)
        print('start training clean victim model for downstream baseline')
        exp_trainer.train_clean_victim(save_path=clean_victim_ckpt)
    else:
        print('[Main] skip clean victim training (train_clean_victim=False)')


if __name__ == "__main__":
    config = parser_args()
    main(config)
