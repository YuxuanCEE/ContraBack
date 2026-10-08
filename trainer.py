import os
import csv
import copy
import torch
import torch.nn.functional as F
from torch import optim
import tqdm
import numpy as np
from dataset import TimeDataset, AttackEvaluateSet
from torch.utils.data import DataLoader
from attack import Attacker, fft_compress
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from forecast_models import TimesNet, Autoformer, FEDformer, PatchTST, TimeVLM, TimeLLM, T3Time

MODEL_MAP = {
    'TimesNet': TimesNet,
    'Autoformer': Autoformer,
    'FEDformer': FEDformer,
    'PatchTST': PatchTST,
    'TimeVLM': TimeVLM,
    'TimeLLM': TimeLLM,
    'T3Time': T3Time
}


class Trainer:
    def __init__(self, config, atk_vars, target_pattern, train_mean, train_std,
                 train_data, test_data, train_data_stamps, test_data_stamps, device, target_idx):
        self.config = config
        self.mean = train_mean
        self.std = train_std
        self.test_data = test_data
        self.device = device
        self.target_idx = int(target_idx)

        self.seq_len = int(config.seq_len)
        self.pred_len = int(config.pred_len)
        self.points_per_day = int(config.points_per_day)
        self.sample_mode = str(getattr(config, 'sample_mode', 'generic'))
        self.sample_stride = int(getattr(config, 'sample_stride', 1))
        self.day_aligned = self.sample_mode == 'day_aligned'

        self.net = MODEL_MAP[self.config.surrogate_name](self.config.Surrogate).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=config.learning_rate)

        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.warmup = config.warmup

        self.train_data = train_data
        self.train_data_stamps = train_data_stamps
        self.test_data_stamps = test_data_stamps

        self.use_timestamps = bool(config.Dataset.use_timestamps)

        train_set = TimeDataset(
            train_data,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            timestamps=train_data_stamps,
            points_per_day=self.points_per_day,
            sample_mode=self.sample_mode,
            sample_stride=self.sample_stride,
            target_idx=self.target_idx,
            use_timestamps=self.use_timestamps,
        )

        channel_features = fft_compress(train_data, 200)
        self.attacker = Attacker(train_set, channel_features, atk_vars, config, target_pattern, device)

        self.prepare_data()

    def load_attacker(self, attacker_state):
        self.attacker.load_state_dict(attacker_state)

    def save_attacker(self):
        attacker_state = self.attacker.state_dict()
        return attacker_state

    def prepare_data(self):
        self.train_set = self.attacker.dataset

        self.clean_train_set = TimeDataset(
            self.train_data,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            timestamps=self.train_data_stamps,
            points_per_day=self.points_per_day,
            sample_mode=self.sample_mode,
            sample_stride=self.sample_stride,
            target_idx=self.target_idx,
            use_timestamps=self.use_timestamps,
        )

        self.cln_test_set = TimeDataset(
            self.test_data,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            timestamps=self.test_data_stamps,
            points_per_day=self.points_per_day,
            sample_mode=self.sample_mode,
            sample_stride=self.sample_stride,
            target_idx=self.target_idx,
            use_timestamps=self.use_timestamps,
        )

        self.atk_test_set = AttackEvaluateSet(self.cln_test_set, self.attacker)

        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.clean_train_loader = DataLoader(self.clean_train_set, batch_size=self.batch_size, shuffle=True)
        self.cln_test_loader = DataLoader(self.cln_test_set, batch_size=self.batch_size, shuffle=False)
        self.atk_test_loader = DataLoader(
            self.atk_test_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.atk_test_set.collate_fn
        )

    def _build_zero_marks(self, batch_size):
        x_mark = torch.zeros(batch_size, self.seq_len, 5, device=self.device)
        y_mark = torch.zeros(batch_size, self.pred_len, 5, device=self.device)
        return x_mark, y_mark

    def _unpack_batch(self, batch_data):
        if len(batch_data) == 4:
            encoder_inputs, labels, clean_labels, idx = batch_data
            x_mark, y_mark = self._build_zero_marks(encoder_inputs.shape[0])
        elif len(batch_data) == 6:
            encoder_inputs, labels, clean_labels, idx, x_mark, y_mark = batch_data
            x_mark = x_mark.to(self.device)
            y_mark = y_mark.to(self.device)
        else:
            raise ValueError(f"Unexpected batch size: {len(batch_data)}")

        encoder_inputs = encoder_inputs.to(self.device)
        labels = labels.to(self.device)
        clean_labels = clean_labels.to(self.device)

        if torch.is_tensor(idx):
            idx = idx.to(self.device)
        else:
            idx = torch.tensor(idx, dtype=torch.long, device=self.device)

        return encoder_inputs, labels, clean_labels, x_mark, y_mark, idx

    def _is_timevlm_modal_ablation_enabled(self, model, model_role='victim'):
        model_name = self.config.model_name if model_role == 'victim' else self.config.surrogate_name
        return (
            bool(getattr(self.config, 'timevlm_modal_ablation', False))
            and model_name == 'TimeVLM'
            and model.__class__.__name__ == 'Model'
        )

    def _parse_timevlm_modal_mask(self, modal_mask=None):
        mask = str(modal_mask if modal_mask is not None else getattr(self.config, 'timevlm_modal_mask', 'T1V1L1')).upper()
        if len(mask) != 6 or mask[0] != 'T' or mask[2] != 'V' or mask[4] != 'L':
            raise ValueError(f'Invalid timevlm_modal_mask={mask}. Expected format like T1V0L0.')
        flags = {'T': mask[1], 'V': mask[3], 'L': mask[5]}
        if any(v not in {'0', '1'} for v in flags.values()):
            raise ValueError(f'Invalid timevlm_modal_mask={mask}. Each modality must be 0 or 1.')
        return flags

    def _get_clean_inputs_by_ids(self, dataset, idx):
        xs = []
        for sid in idx.detach().cpu().numpy().astype(int).tolist():
            x, _, _, _, _ = dataset.get_clean_sample(int(sid))
            xs.append(x)
        return torch.stack(xs, dim=0).to(self.device)

    def _forward_timevlm_modal_ablation(self, model, poisoned_inputs, clean_inputs, labels, x_mark, y_mark, modal_mask=None):
        flags = self._parse_timevlm_modal_mask(modal_mask)
        x_temporal = poisoned_inputs if flags['T'] == '1' else clean_inputs
        x_vision = poisoned_inputs if flags['V'] == '1' else clean_inputs
        x_language = poisoned_inputs if flags['L'] == '1' else clean_inputs
        x_des = torch.zeros_like(labels)
        outputs = model(
            x_temporal,
            x_mark,
            x_des,
            y_mark,
            x_vision=x_vision,
            x_language=x_language,
        )
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        return outputs[:, -labels.shape[1]:, :]

    def _forward_with_optional_modal_ablation(
        self,
        model,
        encoder_inputs,
        labels,
        x_mark,
        y_mark,
        idx,
        dataset,
        model_role='victim',
        modal_mask=None,
    ):
        if self._is_timevlm_modal_ablation_enabled(model, model_role=model_role):
            clean_inputs = self._get_clean_inputs_by_ids(dataset, idx)
            return self._forward_timevlm_modal_ablation(
                model,
                poisoned_inputs=encoder_inputs,
                clean_inputs=clean_inputs,
                labels=labels,
                x_mark=x_mark,
                y_mark=y_mark,
                modal_mask=modal_mask,
            )
        return self._forward_model(model, encoder_inputs, labels, x_mark, y_mark)

    def _forward_model(self, model, encoder_inputs, labels, x_mark, y_mark):
        x_des = torch.zeros_like(labels)
        try:
            outputs = model(encoder_inputs, x_mark, x_des, y_mark)
        except TypeError:
            try:
                outputs = model(encoder_inputs, x_mark, x_des, None)
            except TypeError:
                outputs = model(encoder_inputs)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        outputs = outputs[:, -labels.shape[1]:, :]
        return outputs

    def _denorm_target_numpy(self, arr):
        return arr * float(self.std[self.target_idx]) + float(self.mean[self.target_idx])

    def _get_exp_tag(self):
        data_name = getattr(self.config.Dataset, 'data_filename', self.config.dataset)
        data_name = os.path.splitext(os.path.basename(str(data_name)))[0]
        tag = (
            f"{data_name}_"
            f"{self.config.model_name}_"
            f"sur{self.config.surrogate_name}_"
            f"sl{self.seq_len}_pl{self.pred_len}_"
            f"tr{self.attacker.trigger_len}_"
            f"ts{self.attacker.trigger_start_in_hist}_"
            f"tps{self.attacker.target_start_in_pred}_"
            f"pt{self.config.pattern_type}"
        )
        if (
            bool(getattr(self.config, 'timevlm_modal_ablation', False))
            and (self.config.model_name == 'TimeVLM' or self.config.surrogate_name == 'TimeVLM')
        ):
            tag += f"_modal{str(getattr(self.config, 'timevlm_modal_mask', 'T1V1L1')).upper()}"
        return tag

    def _stage1_loss_log_path(self):
        save_dir = getattr(self.config, 'stage1_loss_log_dir', './Stage1LossLogs')
        os.makedirs(save_dir, exist_ok=True)
        return os.path.join(save_dir, f'{self._get_exp_tag()}.csv')

    def _stage2_loss_log_path(self):
        save_dir = getattr(self.config, 'stage2_loss_log_dir', './Stage2LossLogs')
        os.makedirs(save_dir, exist_ok=True)
        return os.path.join(save_dir, f'{self._get_exp_tag()}.csv')

    def _init_stage1_loss_meter(self):
        return {
            'total_sum': 0.0,
            'total_count': 0,
            'clean_sum': 0.0,
            'clean_count': 0,
            'poison_sum': 0.0,
            'poison_count': 0,
            'attack_sum': 0.0,
            'attack_count': 0,
            'non_attack_sum': 0.0,
            'non_attack_count': 0,
            'clean_gt_sum': 0.0,
            'clean_gt_count': 0,
            'poison_target_sum': 0.0,
            'poison_target_count': 0,
            'poison_clean_gt_sum': 0.0,
            'poison_clean_gt_count': 0,
        }

    def _update_stage1_loss_meter(
        self,
        meter,
        loss_per_timestep,
        loss_per_sample,
        clean_gt_loss_per_timestep,
        idx,
    ):
        batch_size, pred_len = loss_per_timestep.shape
        poison_ids = set(int(i) for i in np.asarray(getattr(self.attacker, 'atk_sample_ids', []), dtype=np.int64).tolist())
        idx_list = idx.detach().cpu().numpy().astype(int).tolist()
        is_poison = torch.tensor([sid in poison_ids for sid in idx_list], dtype=torch.bool, device=loss_per_sample.device)
        is_clean = ~is_poison

        meter['total_sum'] += float(loss_per_sample.detach().sum().item())
        meter['total_count'] += int(batch_size)

        target_start = int(self.attacker.target_start_in_pred)
        target_end = min(pred_len, target_start + int(self.attacker.pattern_len))

        if is_clean.any():
            clean_losses = loss_per_sample[is_clean]
            meter['clean_sum'] += float(clean_losses.detach().sum().item())
            meter['clean_count'] += int(clean_losses.numel())

            clean_gt_losses = clean_gt_loss_per_timestep[is_clean].mean(dim=1)
            meter['clean_gt_sum'] += float(clean_gt_losses.detach().sum().item())
            meter['clean_gt_count'] += int(clean_gt_losses.numel())

        if is_poison.any():
            poison_losses = loss_per_sample[is_poison]
            meter['poison_sum'] += float(poison_losses.detach().sum().item())
            meter['poison_count'] += int(poison_losses.numel())

            if target_end > target_start:
                attack_losses = loss_per_timestep[is_poison, target_start:target_end]
                meter['attack_sum'] += float(attack_losses.detach().sum().item())
                meter['attack_count'] += int(attack_losses.numel())
                meter['poison_target_sum'] += float(attack_losses.detach().sum().item())
                meter['poison_target_count'] += int(attack_losses.numel())

                poison_clean_gt_losses = clean_gt_loss_per_timestep[is_poison, target_start:target_end]
                meter['poison_clean_gt_sum'] += float(poison_clean_gt_losses.detach().sum().item())
                meter['poison_clean_gt_count'] += int(poison_clean_gt_losses.numel())

            non_attack_mask = torch.ones(pred_len, dtype=torch.bool, device=loss_per_timestep.device)
            non_attack_mask[target_start:target_end] = False
            if non_attack_mask.any():
                non_attack_losses = loss_per_timestep[is_poison][:, non_attack_mask]
                meter['non_attack_sum'] += float(non_attack_losses.detach().sum().item())
                meter['non_attack_count'] += int(non_attack_losses.numel())

    def _init_stage2_loss_meter(self):
        return {
            'total_sum': 0.0,
            'total_count': 0,
            'clean_sum': 0.0,
            'clean_count': 0,
            'poison_sum': 0.0,
            'poison_count': 0,
            'attack_sum': 0.0,
            'attack_count': 0,
            'non_attack_sum': 0.0,
            'non_attack_count': 0,
        }

    def _update_stage2_loss_meter(self, meter, loss_per_timestep, loss_per_sample, idx):
        batch_size, pred_len = loss_per_timestep.shape
        poison_ids = set(int(i) for i in np.asarray(getattr(self.attacker, 'atk_sample_ids', []), dtype=np.int64).tolist())
        idx_list = idx.detach().cpu().numpy().astype(int).tolist()
        is_poison = torch.tensor([sid in poison_ids for sid in idx_list], dtype=torch.bool, device=loss_per_sample.device)
        is_clean = ~is_poison

        meter['total_sum'] += float(loss_per_sample.detach().sum().item())
        meter['total_count'] += int(batch_size)

        if is_clean.any():
            clean_losses = loss_per_sample[is_clean]
            meter['clean_sum'] += float(clean_losses.detach().sum().item())
            meter['clean_count'] += int(clean_losses.numel())

        if is_poison.any():
            poison_losses = loss_per_sample[is_poison]
            meter['poison_sum'] += float(poison_losses.detach().sum().item())
            meter['poison_count'] += int(poison_losses.numel())

            target_start = int(self.attacker.target_start_in_pred)
            target_end = min(pred_len, target_start + int(self.attacker.pattern_len))
            if target_end > target_start:
                attack_losses = loss_per_timestep[is_poison, target_start:target_end]
                meter['attack_sum'] += float(attack_losses.detach().sum().item())
                meter['attack_count'] += int(attack_losses.numel())

            non_attack_mask = torch.ones(pred_len, dtype=torch.bool, device=loss_per_timestep.device)
            non_attack_mask[target_start:target_end] = False
            if non_attack_mask.any():
                non_attack_losses = loss_per_timestep[is_poison][:, non_attack_mask]
                meter['non_attack_sum'] += float(non_attack_losses.detach().sum().item())
                meter['non_attack_count'] += int(non_attack_losses.numel())

    def _snapshot_patch_memory_bank(self, model):
        bank = getattr(model, 'patch_memory_bank', None)
        if bank is None:
            return None
        return {
            'patches': bank.patches.detach().clone(),
            'ptr': int(bank.ptr),
        }

    def _restore_patch_memory_bank(self, model, snapshot):
        if snapshot is None:
            return
        bank = getattr(model, 'patch_memory_bank', None)
        if bank is None:
            return
        bank.patches.copy_(snapshot['patches'].to(bank.patches.device))
        bank.ptr = int(snapshot['ptr'])

    def _zero_stage1_probe_grads(self):
        self.optimizer.zero_grad(set_to_none=True)
        self.attacker.attack_optim.zero_grad(set_to_none=True)
        self.net.zero_grad(set_to_none=True)
        self.attacker.trigger_generator.zero_grad(set_to_none=True)

    def _build_stage1_probe_clean_batch(self, batch_ids):
        xs, ys = [], []
        x_marks, y_marks = [], []
        has_mark = False

        for sid in batch_ids:
            x, y, xm, ym, _ = self.attacker.dataset.get_clean_sample(int(sid))
            xs.append(x)
            ys.append(y)
            if xm is not None and ym is not None:
                has_mark = True
                x_marks.append(xm)
                y_marks.append(ym)

        x = torch.stack(xs, dim=0).to(self.device)
        y = torch.stack(ys, dim=0).to(self.device)
        idx = torch.as_tensor(batch_ids, dtype=torch.long, device=self.device)

        if has_mark:
            x_mark = torch.stack(x_marks, dim=0).to(self.device)
            y_mark = torch.stack(y_marks, dim=0).to(self.device)
        else:
            x_mark = None
            y_mark = None

        return x, y, x_mark, y_mark, idx

    def _trigger_grad_vector(self, loss):
        params = [p for p in self.attacker.trigger_generator.parameters() if p.requires_grad]
        if len(params) == 0:
            return torch.empty(0, device=self.device), 0.0

        total_numel = sum(p.numel() for p in params)
        if not loss.requires_grad:
            return torch.zeros(total_numel, device=self.device), 0.0

        grads = torch.autograd.grad(loss, params, allow_unused=True)
        flat_grads = []
        for param, grad in zip(params, grads):
            if grad is None:
                flat_grads.append(torch.zeros_like(param).reshape(-1))
            else:
                flat_grads.append(grad.detach().reshape(-1))

        grad_vec = torch.cat(flat_grads) if flat_grads else torch.zeros(total_numel, device=self.device)
        return grad_vec, float(torch.linalg.vector_norm(grad_vec).detach().item())

    @staticmethod
    def _safe_cosine(vec_a, vec_b):
        if vec_a.numel() == 0 or vec_b.numel() == 0:
            return None
        norm_a = torch.linalg.vector_norm(vec_a)
        norm_b = torch.linalg.vector_norm(vec_b)
        if float(norm_a.detach().item()) == 0.0 or float(norm_b.detach().item()) == 0.0:
            return None
        return float(F.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0)).detach().item())

    def _probe_stage1_mask_gradient(self, model, clean_x, clean_y, x_mark, y_mark, idx, modal_mask, memory_snapshot):
        self._restore_patch_memory_bank(model, memory_snapshot)
        x_bkd, y_bkd = self.attacker.build_poisoned_batch(clean_x, clean_y)
        outputs = self._forward_with_optional_modal_ablation(
            model,
            x_bkd,
            y_bkd,
            x_mark,
            y_mark,
            idx,
            self.train_set,
            model_role='surrogate',
            modal_mask=modal_mask,
        )

        target_start = int(self.attacker.target_start_in_pred)
        target_end = min(y_bkd.shape[1], target_start + int(self.attacker.pattern_len))
        if target_end <= target_start:
            loss = torch.zeros((), dtype=outputs.dtype, device=outputs.device)
        else:
            loss = F.mse_loss(
                outputs[:, target_start:target_end, self.target_idx],
                y_bkd[:, target_start:target_end, self.target_idx]
            )

        grad_vec, grad_norm = self._trigger_grad_vector(loss)
        return grad_vec, grad_norm, float(loss.detach().item())

    @staticmethod
    def _avg_or_none(values):
        values = [v for v in values if v is not None]
        if len(values) == 0:
            return None
        return float(sum(values) / len(values))

    def _empty_stage1_grad_probe_stats(self):
        return {
            'grad_probe_batches': 0,
            'grad_loss_temporal_t1v0l0': None,
            'grad_loss_configured': None,
            'grad_loss_aux_t0v1l1': None,
            'grad_norm_temporal_t1v0l0': None,
            'grad_norm_configured': None,
            'grad_norm_aux_t0v1l1': None,
            'grad_cos_configured_vs_temporal': None,
            'grad_cos_aux_vs_temporal': None,
        }

    def _probe_stage1_grad_conflict(self, epoch):
        if not bool(getattr(self.config, 'stage1_probe_grad_conflict', False)):
            return self._empty_stage1_grad_probe_stats()
        if self.config.surrogate_name != 'TimeVLM' or not bool(getattr(self.config, 'timevlm_modal_ablation', False)):
            return self._empty_stage1_grad_probe_stats()
        if epoch <= self.warmup or (not hasattr(self.attacker, 'atk_sample_ids')) or len(self.attacker.atk_sample_ids) == 0:
            return self._empty_stage1_grad_probe_stats()

        sample_ids = np.asarray(self.attacker.atk_sample_ids, dtype=np.int64)
        probe_batches = max(1, int(getattr(self.config, 'stage1_grad_probe_batches', 1)))
        probe_batch_size = int(getattr(self.config, 'stage1_grad_probe_batch_size', self.batch_size))
        probe_batch_size = max(1, probe_batch_size)

        configured_mask = str(getattr(self.config, 'timevlm_modal_mask', 'T1V1L1')).upper()
        stats = {
            'temporal_loss': [],
            'configured_loss': [],
            'aux_loss': [],
            'temporal_norm': [],
            'configured_norm': [],
            'aux_norm': [],
            'configured_cos': [],
            'aux_cos': [],
        }

        model_was_training = self.net.training
        trigger_was_training = self.attacker.trigger_generator.training
        memory_snapshot = self._snapshot_patch_memory_bank(self.net)
        completed_batches = 0

        try:
            self.net.eval()
            self.attacker.trigger_generator.train()
            self._zero_stage1_probe_grads()

            for batch_index in range(probe_batches):
                beg = batch_index * probe_batch_size
                if beg >= len(sample_ids):
                    break
                batch_ids = sample_ids[beg:beg + probe_batch_size]
                clean_x, clean_y, x_mark, y_mark, idx = self._build_stage1_probe_clean_batch(batch_ids)

                temporal_vec, temporal_norm, temporal_loss = self._probe_stage1_mask_gradient(
                    self.net, clean_x, clean_y, x_mark, y_mark, idx, 'T1V0L0', memory_snapshot
                )
                configured_vec, configured_norm, configured_loss = self._probe_stage1_mask_gradient(
                    self.net, clean_x, clean_y, x_mark, y_mark, idx, configured_mask, memory_snapshot
                )
                aux_vec, aux_norm, aux_loss = self._probe_stage1_mask_gradient(
                    self.net, clean_x, clean_y, x_mark, y_mark, idx, 'T0V1L1', memory_snapshot
                )

                stats['temporal_loss'].append(temporal_loss)
                stats['configured_loss'].append(configured_loss)
                stats['aux_loss'].append(aux_loss)
                stats['temporal_norm'].append(temporal_norm)
                stats['configured_norm'].append(configured_norm)
                stats['aux_norm'].append(aux_norm)
                stats['configured_cos'].append(self._safe_cosine(configured_vec, temporal_vec))
                stats['aux_cos'].append(self._safe_cosine(aux_vec, temporal_vec))
                completed_batches += 1
        finally:
            self._restore_patch_memory_bank(self.net, memory_snapshot)
            self._zero_stage1_probe_grads()
            self.net.train(model_was_training)
            self.attacker.trigger_generator.train(trigger_was_training)

        return {
            'grad_probe_batches': completed_batches,
            'grad_loss_temporal_t1v0l0': self._avg_or_none(stats['temporal_loss']),
            'grad_loss_configured': self._avg_or_none(stats['configured_loss']),
            'grad_loss_aux_t0v1l1': self._avg_or_none(stats['aux_loss']),
            'grad_norm_temporal_t1v0l0': self._avg_or_none(stats['temporal_norm']),
            'grad_norm_configured': self._avg_or_none(stats['configured_norm']),
            'grad_norm_aux_t0v1l1': self._avg_or_none(stats['aux_norm']),
            'grad_cos_configured_vs_temporal': self._avg_or_none(stats['configured_cos']),
            'grad_cos_aux_vs_temporal': self._avg_or_none(stats['aux_cos']),
        }

    @staticmethod
    def _meter_avg(meter, sum_key, count_key):
        count = meter[count_key]
        if count == 0:
            return None
        return meter[sum_key] / count

    def _get_timevlm_modal_mask_label(self, model_name):
        if bool(getattr(self.config, 'timevlm_modal_ablation', False)) and model_name == 'TimeVLM':
            return str(getattr(self.config, 'timevlm_modal_mask', 'T1V1L1')).upper()
        return 'default'

    def _append_stage1_loss_log(self, epoch, meter, trigger_stats, metrics, grad_probe_stats=None):
        if not bool(getattr(self.config, 'log_stage1_loss_detail', False)):
            return

        path = self._stage1_loss_log_path()
        trigger_stats = trigger_stats or {}
        grad_probe_stats = grad_probe_stats or self._empty_stage1_grad_probe_stats()
        row = {
            'epoch': epoch,
            'model_name': self.config.model_name,
            'surrogate_name': self.config.surrogate_name,
            'dataset': self.config.dataset,
            'timevlm_modal_mask': self._get_timevlm_modal_mask_label(self.config.surrogate_name),
            'num_train_samples': meter['total_count'],
            'num_clean_samples': meter['clean_count'],
            'num_poison_samples': meter['poison_count'],
            'surrogate_total_loss': self._meter_avg(meter, 'total_sum', 'total_count'),
            'surrogate_clean_sample_loss': self._meter_avg(meter, 'clean_sum', 'clean_count'),
            'surrogate_poison_sample_loss': self._meter_avg(meter, 'poison_sum', 'poison_count'),
            'surrogate_poison_attack_region_loss': self._meter_avg(meter, 'attack_sum', 'attack_count'),
            'surrogate_poison_non_attack_region_loss': self._meter_avg(meter, 'non_attack_sum', 'non_attack_count'),
            'surrogate_clean_to_gt_loss': self._meter_avg(meter, 'clean_gt_sum', 'clean_gt_count'),
            'surrogate_poison_to_target_loss': self._meter_avg(meter, 'poison_target_sum', 'poison_target_count'),
            'surrogate_poison_to_clean_gt_loss': self._meter_avg(meter, 'poison_clean_gt_sum', 'poison_clean_gt_count'),
            'trigger_total_loss': trigger_stats.get('trigger_total_loss'),
            'trigger_attack_region_loss': trigger_stats.get('trigger_attack_region_loss'),
            'trigger_norm_loss': trigger_stats.get('trigger_norm_loss'),
            'trigger_num_steps': trigger_stats.get('trigger_num_steps'),
            'grad_probe_batches': grad_probe_stats.get('grad_probe_batches'),
            'grad_loss_temporal_t1v0l0': grad_probe_stats.get('grad_loss_temporal_t1v0l0'),
            'grad_loss_configured': grad_probe_stats.get('grad_loss_configured'),
            'grad_loss_aux_t0v1l1': grad_probe_stats.get('grad_loss_aux_t0v1l1'),
            'grad_norm_temporal_t1v0l0': grad_probe_stats.get('grad_norm_temporal_t1v0l0'),
            'grad_norm_configured': grad_probe_stats.get('grad_norm_configured'),
            'grad_norm_aux_t0v1l1': grad_probe_stats.get('grad_norm_aux_t0v1l1'),
            'grad_cos_configured_vs_temporal': grad_probe_stats.get('grad_cos_configured_vs_temporal'),
            'grad_cos_aux_vs_temporal': grad_probe_stats.get('grad_cos_aux_vs_temporal'),
            'clean_rmse_norm': metrics.get('clean_rmse_norm') if metrics is not None else None,
            'attacked_mae_norm': metrics.get('attacked_mae_norm') if metrics is not None else None,
            'attacked_rmse_norm': metrics.get('attacked_rmse_norm') if metrics is not None else None,
        }

        write_header = not os.path.exists(path)
        with open(path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _append_stage2_loss_log(self, epoch, meter, metrics):
        if not bool(getattr(self.config, 'log_stage2_loss_detail', False)):
            return

        path = self._stage2_loss_log_path()
        modal_mask = self._get_timevlm_modal_mask_label(self.config.model_name)

        row = {
            'epoch': epoch,
            'model_name': self.config.model_name,
            'surrogate_name': self.config.surrogate_name,
            'dataset': self.config.dataset,
            'timevlm_modal_mask': modal_mask,
            'num_train_samples': meter['total_count'],
            'num_clean_samples': meter['clean_count'],
            'num_poison_samples': meter['poison_count'],
            'total_loss': self._meter_avg(meter, 'total_sum', 'total_count'),
            'clean_sample_loss': self._meter_avg(meter, 'clean_sum', 'clean_count'),
            'poison_sample_loss': self._meter_avg(meter, 'poison_sum', 'poison_count'),
            'poison_attack_region_loss': self._meter_avg(meter, 'attack_sum', 'attack_count'),
            'poison_non_attack_region_loss': self._meter_avg(meter, 'non_attack_sum', 'non_attack_count'),
            'clean_rmse_norm': metrics.get('clean_rmse_norm') if metrics is not None else None,
            'attacked_mae_norm': metrics.get('attacked_mae_norm') if metrics is not None else None,
            'attacked_rmse_norm': metrics.get('attacked_rmse_norm') if metrics is not None else None,
        }

        write_header = not os.path.exists(path)
        with open(path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _update_best_metrics(self, best_metrics, cur_metrics, key='clean_rmse_norm'):
        if best_metrics is None:
            return dict(cur_metrics)
        if cur_metrics.get(key, None) is None:
            return best_metrics
        if cur_metrics[key] < best_metrics[key]:
            return dict(cur_metrics)
        return best_metrics

    def _print_test_summary(self, final_metrics, best_clean_metrics, best_attack_metrics, title='Final Test Summary'):
        print('\n' + '=' * 24 + f' {title} ' + '=' * 24)

        if final_metrics is not None:
            print(
                f"[Final Epoch {final_metrics['epoch']}] "
                f"clean MAE(raw): {final_metrics['clean_mae_raw']:.6f}, "
                f"clean RMSE(raw): {final_metrics['clean_rmse_raw']:.6f}, "
                f"clean MAE(norm): {final_metrics['clean_mae_norm']:.6f}, "
                f"clean RMSE(norm): {final_metrics['clean_rmse_norm']:.6f}, "
                f"attacked MAE(raw): {final_metrics['attacked_mae_raw']:.6f}, "
                f"attacked RMSE(raw): {final_metrics['attacked_rmse_raw']:.6f}, "
                f"attacked MAE(norm): {final_metrics['attacked_mae_norm']:.6f}, "
                f"attacked RMSE(norm): {final_metrics['attacked_rmse_norm']:.6f}"
            )

        if best_clean_metrics is not None:
            print(
                f"[Best Clean @ Epoch {best_clean_metrics['epoch']}] "
                f"clean MAE(raw): {best_clean_metrics['clean_mae_raw']:.6f}, "
                f"clean RMSE(raw): {best_clean_metrics['clean_rmse_raw']:.6f}, "
                f"clean MAE(norm): {best_clean_metrics['clean_mae_norm']:.6f}, "
                f"clean RMSE(norm): {best_clean_metrics['clean_rmse_norm']:.6f}"
            )

        if best_attack_metrics is not None:
            print(
                f"[Best Attack @ Epoch {best_attack_metrics['epoch']}] "
                f"attacked MAE(raw): {best_attack_metrics['attacked_mae_raw']:.6f}, "
                f"attacked RMSE(raw): {best_attack_metrics['attacked_rmse_raw']:.6f}, "
                f"attacked MAE(norm): {best_attack_metrics['attacked_mae_norm']:.6f}, "
                f"attacked RMSE(norm): {best_attack_metrics['attacked_rmse_norm']:.6f}"
            )

        print('=' * 68 + '\n')

    @torch.no_grad()
    def _predict_single_sample(self, model, x, y, x_mark=None, y_mark=None):
        model.eval()

        x = x.unsqueeze(0).to(self.device)
        y = y.unsqueeze(0).to(self.device)

        if x_mark is None or y_mark is None:
            x_mark_b, y_mark_b = self._build_zero_marks(1)
        else:
            x_mark_b = x_mark.unsqueeze(0).to(self.device)
            y_mark_b = y_mark.unsqueeze(0).to(self.device)

        outputs = self._forward_model(model, x, y, x_mark_b, y_mark_b)
        pred = outputs[0, :, self.target_idx].detach().cpu().numpy()
        return self._denorm_target_numpy(pred)

    @torch.no_grad()
    def save_prediction_figures(self, model, num_samples=10, prefix='final'):
        model.eval()
        self.attacker.eval()

        save_dir = './Figure'
        os.makedirs(save_dir, exist_ok=True)

        total_num = len(self.cln_test_set)
        num_samples = min(num_samples, total_num)
        if num_samples <= 0:
            print('[Figure] No test samples available, skip plotting.')
            return

        sampled_ids = np.random.choice(total_num, size=num_samples, replace=False)
        exp_tag = self._get_exp_tag()

        # collect data for CSV export if enabled
        csv_rows = []

        for fig_idx, sample_id in enumerate(sampled_ids, start=1):
            x, y, x_mark, y_mark, global_start = self.cln_test_set.get_clean_sample(int(sample_id))

            x_clean = x.clone()
            y_clean = y.clone()

            x_bkd, y_bkd = self.attacker.build_poisoned_sample(x_clean, y_clean)

            clean_pred = self._predict_single_sample(model, x_clean, y_clean, x_mark, y_mark)
            bkd_pred = self._predict_single_sample(model, x_bkd, y_clean, x_mark, y_mark)

            hist_clean = self._denorm_target_numpy(
                x_clean[:, self.target_idx].detach().cpu().numpy()
            )
            hist_bkd = self._denorm_target_numpy(
                x_bkd[:, self.target_idx].detach().cpu().numpy()
            )

            gt = self._denorm_target_numpy(
                y_clean[:, self.target_idx].detach().cpu().numpy()
            )

            x_hist = np.arange(len(hist_clean))
            x_futr = np.arange(len(hist_clean), len(hist_clean) + len(gt))

            plt.figure(figsize=(12, 4.5))

            plt.plot(
                x_hist, hist_clean,
                label='History Input (Clean)',
                color='black',
                linewidth=1.8
            )
            plt.plot(
                x_hist, hist_bkd,
                label='History Input (Poisoned)',
                color='darkorange',
                linestyle='--',
                linewidth=2.0
            )

            trigger_segments = getattr(self.attacker, 'trigger_segments', None)
            if trigger_segments is None:
                trigger_segments = [(self.attacker.trigger_start_in_hist,
                                     self.attacker.trigger_start_in_hist + self.attacker.trigger_len)]

            first_seg = True
            for seg_start, seg_end in trigger_segments:
                plt.axvspan(
                    seg_start, seg_end - 1,
                    color='orange',
                    alpha=0.15,
                    label='Trigger Region' if first_seg else None
                )
                first_seg = False

            plt.plot(
                x_futr, gt,
                label='Ground Truth',
                color='green',
                linewidth=2.0
            )
            plt.plot(
                x_futr, clean_pred,
                label='Clean Prediction',
                color='royalblue',
                linewidth=2.0
            )
            plt.plot(
                x_futr, bkd_pred,
                label='Backdoor Prediction',
                color='red',
                linestyle='--',
                linewidth=2.0
            )

            target_start = self.attacker.target_start_in_pred
            target_end = self.attacker.target_start_in_pred + self.attacker.pattern_len - 1
            plt.axvspan(
                len(hist_clean) + target_start,
                len(hist_clean) + target_end,
                color='red',
                alpha=0.08,
                label='Target Region'
            )

            plt.axvline(x=len(hist_clean) - 1, color='gray', linestyle=':', linewidth=1.2)

            plt.title(
                f"{prefix} | Sample {int(sample_id)} | {exp_tag}",
                fontsize=10
            )
            plt.xlabel('Time Step')
            plt.ylabel('Target Value')
            plt.legend()
            plt.tight_layout()

            save_name = f"{prefix}_{exp_tag}_sample{int(sample_id):04d}_{fig_idx:02d}.png"
            save_path = os.path.join(save_dir, save_name)
            plt.savefig(save_path, dpi=300, format='png')
            plt.close()

            # collect data for CSV export
            if bool(getattr(self.config, 'save_figure_data_csv', False)):
                import json
                csv_rows.append({
                    'sample_id': int(sample_id),
                    'fig_idx': fig_idx,
                    'input_clean': json.dumps(hist_clean.tolist()),
                    'input_poisoned': json.dumps(hist_bkd.tolist()),
                    'ground_truth': json.dumps(gt.tolist()),
                    'clean_prediction': json.dumps(clean_pred.tolist()),
                    'backdoor_prediction': json.dumps(bkd_pred.tolist()),
                    'trigger_segments': str(trigger_segments),
                    'target_segment': f'({target_start}, {target_end})',
                })

        print(f'[Figure] Saved {num_samples} figures to {save_dir}/ with prefix={prefix}')

        # export CSV if enabled
        if bool(getattr(self.config, 'save_figure_data_csv', False)) and len(csv_rows) > 0:
            import pandas as pd
            csv_name = f"{prefix}_{exp_tag}_data.csv"
            csv_path = os.path.join(save_dir, csv_name)
            df = pd.DataFrame(csv_rows)
            df.to_csv(csv_path, index=False)
            print(f'[Figure] Saved prediction data to {csv_path}')

    def train(self):
        torch.autograd.set_detect_anomaly(True)

        self.attacker.train()
        poison_metrics = None

        for epoch in range(self.num_epochs):
            self.net.train()

            if epoch > self.warmup:
                if (not hasattr(self.attacker, 'atk_sample_ids')) or (len(self.attacker.atk_sample_ids) == 0):
                    self.attacker.select_atk_timestamp(poison_metrics)
                self.attacker.sparse_inject()

            poison_metrics = np.zeros(len(self.train_set), dtype=np.float32)

            self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
            loss_meter = self._init_stage1_loss_meter()
            pbar = tqdm.tqdm(self.train_loader, desc=f'Training data {epoch}/{self.num_epochs}')

            for batch_index, batch_data in enumerate(pbar):
                encoder_inputs, labels, clean_labels, x_mark, y_mark, idx = self._unpack_batch(batch_data)

                self.optimizer.zero_grad()
                outputs = self._forward_with_optional_modal_ablation(
                    self.net,
                    encoder_inputs,
                    labels,
                    x_mark,
                    y_mark,
                    idx,
                    self.train_set,
                    model_role='surrogate',
                )

                loss_per_timestep = F.smooth_l1_loss(
                    outputs[:, :, self.target_idx],
                    labels[:, :, self.target_idx],
                    reduction='none'
                )
                loss_per_sample = loss_per_timestep.mean(dim=1)
                clean_gt_loss_per_timestep = F.smooth_l1_loss(
                    outputs[:, :, self.target_idx],
                    clean_labels[:, :, self.target_idx],
                    reduction='none'
                )

                idx_np = idx.detach().cpu().numpy().astype(int)
                poison_metrics[idx_np] = loss_per_sample.detach().cpu().numpy()

                loss = loss_per_sample.mean()
                self._update_stage1_loss_meter(
                    loss_meter,
                    loss_per_timestep,
                    loss_per_sample,
                    clean_gt_loss_per_timestep,
                    idx,
                )
                loss.backward()
                self.optimizer.step()

            trigger_stats = None
            if epoch > self.warmup:
                trigger_stats = self.attacker.update_trigger_generator(
                    self.net,
                    batch_size=self.batch_size,
                    device=self.device,
                    forward_fn=lambda model, x_bkd, y_bkd, x_mark, y_mark, idx: self._forward_with_optional_modal_ablation(
                        model,
                        x_bkd,
                        y_bkd,
                        x_mark,
                        y_mark,
                        idx,
                        self.train_set,
                        model_role='surrogate',
                    ),
                    return_detail=True,
                )

            grad_probe_stats = self._probe_stage1_grad_conflict(epoch)
            metrics = self.validate(self.net, epoch, self.warmup, model_role='surrogate')
            self._append_stage1_loss_log(epoch, loss_meter, trigger_stats, metrics, grad_probe_stats)

    def validate(self, model, epoch, atk_eval_epoch=0, model_role='victim'):
        model.eval()
        self.attacker.eval()
        cln_info = atk_info = ''

        metrics = {
            'epoch': epoch,
            'clean_mae_raw': None,
            'clean_rmse_raw': None,
            'clean_mae_norm': None,
            'clean_rmse_norm': None,
            'attacked_mae_raw': None,
            'attacked_rmse_raw': None,
            'attacked_mae_norm': None,
            'attacked_rmse_norm': None,
        }

        with torch.no_grad():
            cln_preds_norm = []
            cln_targets_norm = []
            atk_preds_norm = []
            atk_targets_norm = []

            for batch_index, batch_data in enumerate(self.cln_test_loader):
                encoder_inputs, labels, clean_labels, x_mark, y_mark, idx = self._unpack_batch(batch_data)

                outputs = self._forward_model(model, encoder_inputs, labels, x_mark, y_mark)

                pred_norm = outputs[:, :, self.target_idx].cpu().detach().numpy()
                true_norm = labels[:, :, self.target_idx].cpu().detach().numpy()

                cln_preds_norm.append(pred_norm)
                cln_targets_norm.append(true_norm)

            cln_preds_norm = np.concatenate(cln_preds_norm, axis=0)
            cln_targets_norm = np.concatenate(cln_targets_norm, axis=0)

            cln_mae_norm = mean_absolute_error(cln_targets_norm.reshape(-1, 1), cln_preds_norm.reshape(-1, 1))
            cln_rmse_norm = mean_squared_error(cln_targets_norm.reshape(-1, 1), cln_preds_norm.reshape(-1, 1)) ** 0.5

            cln_preds_raw = self._denorm_target_numpy(cln_preds_norm)
            cln_targets_raw = self._denorm_target_numpy(cln_targets_norm)

            cln_mae_raw = mean_absolute_error(cln_targets_raw.reshape(-1, 1), cln_preds_raw.reshape(-1, 1))
            cln_rmse_raw = mean_squared_error(cln_targets_raw.reshape(-1, 1), cln_preds_raw.reshape(-1, 1)) ** 0.5

            metrics['clean_mae_raw'] = float(cln_mae_raw)
            metrics['clean_rmse_raw'] = float(cln_rmse_raw)
            metrics['clean_mae_norm'] = float(cln_mae_norm)
            metrics['clean_rmse_norm'] = float(cln_rmse_norm)

            cln_info = (
                f' | clean MAE(raw): {cln_mae_raw}, clean RMSE(raw): {cln_rmse_raw}'
                f' | clean MAE(norm): {cln_mae_norm}, clean RMSE(norm): {cln_rmse_norm}'
            )

            if epoch > atk_eval_epoch:
                for batch_index, batch_data in enumerate(self.atk_test_loader):
                    encoder_inputs, labels, clean_labels, x_mark, y_mark, idx = self._unpack_batch(batch_data)

                    if self._is_timevlm_modal_ablation_enabled(model, model_role=model_role):
                        clean_inputs = self._get_clean_inputs_by_ids(self.cln_test_set, idx)
                        outputs = self._forward_timevlm_modal_ablation(
                            model,
                            poisoned_inputs=encoder_inputs,
                            clean_inputs=clean_inputs,
                            labels=labels,
                            x_mark=x_mark,
                            y_mark=y_mark,
                        )
                    else:
                        outputs = self._forward_model(model, encoder_inputs, labels, x_mark, y_mark)

                    target_start = int(self.attacker.target_start_in_pred)
                    target_end = min(labels.shape[1], target_start + self.attacker.pattern_len)

                    pred_norm = outputs[:, target_start:target_end, self.target_idx].cpu().detach().numpy()
                    true_norm = labels[:, target_start:target_end, self.target_idx].cpu().detach().numpy()

                    atk_preds_norm.append(pred_norm)
                    atk_targets_norm.append(true_norm)

                atk_preds_norm = np.concatenate(atk_preds_norm, axis=0)
                atk_targets_norm = np.concatenate(atk_targets_norm, axis=0)

                atk_mae_norm = mean_absolute_error(atk_targets_norm.reshape(-1, 1), atk_preds_norm.reshape(-1, 1))
                atk_rmse_norm = mean_squared_error(atk_targets_norm.reshape(-1, 1), atk_preds_norm.reshape(-1, 1)) ** 0.5

                atk_preds_raw = self._denorm_target_numpy(atk_preds_norm)
                atk_targets_raw = self._denorm_target_numpy(atk_targets_norm)

                atk_mae_raw = mean_absolute_error(atk_targets_raw.reshape(-1, 1), atk_preds_raw.reshape(-1, 1))
                atk_rmse_raw = mean_squared_error(atk_targets_raw.reshape(-1, 1), atk_preds_raw.reshape(-1, 1)) ** 0.5

                metrics['attacked_mae_raw'] = float(atk_mae_raw)
                metrics['attacked_rmse_raw'] = float(atk_rmse_raw)
                metrics['attacked_mae_norm'] = float(atk_mae_norm)
                metrics['attacked_rmse_norm'] = float(atk_rmse_norm)

                atk_info = (
                    f' | attacked MAE(raw): {atk_mae_raw}, attacked RMSE(raw): {atk_rmse_raw}'
                    f' | attacked MAE(norm): {atk_mae_norm}, attacked RMSE(norm): {atk_rmse_norm}'
                )

        info = 'Epoch: {}'.format(epoch) + cln_info + atk_info
        print(info)
        return metrics

    def validate_clean_only(self, model, epoch):
        model.eval()
        info = ''

        metrics = {
            'epoch': epoch,
            'clean_mae_raw': None,
            'clean_rmse_raw': None,
            'clean_mae_norm': None,
            'clean_rmse_norm': None,
        }

        with torch.no_grad():
            cln_preds_norm = []
            cln_targets_norm = []

            for batch_index, batch_data in enumerate(self.cln_test_loader):
                encoder_inputs, labels, clean_labels, x_mark, y_mark, idx = self._unpack_batch(batch_data)
                outputs = self._forward_model(model, encoder_inputs, labels, x_mark, y_mark)

                pred_norm = outputs[:, :, self.target_idx].cpu().detach().numpy()
                true_norm = labels[:, :, self.target_idx].cpu().detach().numpy()

                cln_preds_norm.append(pred_norm)
                cln_targets_norm.append(true_norm)

            cln_preds_norm = np.concatenate(cln_preds_norm, axis=0)
            cln_targets_norm = np.concatenate(cln_targets_norm, axis=0)

            cln_mae_norm = mean_absolute_error(cln_targets_norm.reshape(-1, 1), cln_preds_norm.reshape(-1, 1))
            cln_rmse_norm = mean_squared_error(cln_targets_norm.reshape(-1, 1), cln_preds_norm.reshape(-1, 1)) ** 0.5

            cln_preds_raw = self._denorm_target_numpy(cln_preds_norm)
            cln_targets_raw = self._denorm_target_numpy(cln_targets_norm)

            cln_mae_raw = mean_absolute_error(cln_targets_raw.reshape(-1, 1), cln_preds_raw.reshape(-1, 1))
            cln_rmse_raw = mean_squared_error(cln_targets_raw.reshape(-1, 1), cln_preds_raw.reshape(-1, 1)) ** 0.5

            metrics['clean_mae_raw'] = float(cln_mae_raw)
            metrics['clean_rmse_raw'] = float(cln_rmse_raw)
            metrics['clean_mae_norm'] = float(cln_mae_norm)
            metrics['clean_rmse_norm'] = float(cln_rmse_norm)

            info = (
                f'[CleanVictim][Epoch {epoch}] '
                f'clean MAE(raw): {cln_mae_raw}, clean RMSE(raw): {cln_rmse_raw}'
                f' | clean MAE(norm): {cln_mae_norm}, clean RMSE(norm): {cln_rmse_norm}'
            )

        print(info)
        return metrics

    def train_clean_victim(self, save_path=None):
        """
        使用完全干净的训练集训练一个 clean victim model，
        供 Stage 3 作为 clean baseline 使用。
        """
        model = MODEL_MAP[self.config.model_name](self.config.Model).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)

        best_clean_metrics = None
        best_clean_state = None
        best_clean_epoch = None
        final_metrics = None

        clean_train_loader = DataLoader(self.clean_train_set, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.num_epochs):
            model.train()
            pbar = tqdm.tqdm(clean_train_loader, desc=f'Training clean victim model {epoch}/{self.num_epochs}')

            for batch_index, batch_data in enumerate(pbar):
                encoder_inputs, labels, clean_labels, x_mark, y_mark, idx = self._unpack_batch(batch_data)

                optimizer.zero_grad()
                outputs = self._forward_model(model, encoder_inputs, labels, x_mark, y_mark)

                loss = F.smooth_l1_loss(
                    outputs[:, :, self.target_idx],
                    labels[:, :, self.target_idx]
                )
                loss.backward()
                optimizer.step()

            metrics = self.validate_clean_only(model, epoch)
            final_metrics = metrics

            prev_best = best_clean_metrics
            best_clean_metrics = self._update_best_metrics(best_clean_metrics, metrics, key='clean_rmse_norm')
            if prev_best is None or best_clean_metrics['epoch'] == epoch:
                best_clean_state = copy.deepcopy(model.state_dict())
                best_clean_epoch = epoch

        self._print_test_summary(
            final_metrics={
                'epoch': final_metrics['epoch'],
                'clean_mae_raw': final_metrics['clean_mae_raw'],
                'clean_rmse_raw': final_metrics['clean_rmse_raw'],
                'clean_mae_norm': final_metrics['clean_mae_norm'],
                'clean_rmse_norm': final_metrics['clean_rmse_norm'],
                'attacked_mae_raw': 0.0,
                'attacked_rmse_raw': 0.0,
                'attacked_mae_norm': 0.0,
                'attacked_rmse_norm': 0.0,
            },
            best_clean_metrics={
                'epoch': best_clean_metrics['epoch'],
                'clean_mae_raw': best_clean_metrics['clean_mae_raw'],
                'clean_rmse_raw': best_clean_metrics['clean_rmse_raw'],
                'clean_mae_norm': best_clean_metrics['clean_mae_norm'],
                'clean_rmse_norm': best_clean_metrics['clean_rmse_norm'],
            },
            best_attack_metrics=None,
            title='Clean Victim Summary'
        )

        if save_path is not None and best_clean_state is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(
                {
                    'model_state_dict': best_clean_state,
                    'best_epoch': best_clean_epoch,
                    'best_metrics': best_clean_metrics,
                    'config_model': dict(self.config.Model),
                    'target_idx': self.target_idx,
                },
                save_path
            )
            print(f'[CKPT] Saved clean victim ckpt to {save_path}')

        return {
            'model': model,
            'best_state_dict': best_clean_state,
            'best_epoch': best_clean_epoch,
            'best_metrics': best_clean_metrics,
        }

    def test(self, save_poisoned_ckpt_path=None):
        self.attacker.eval()

        model = MODEL_MAP[self.config.model_name](self.config.Model).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)

        self.attacker.sparse_inject()
        self.train_set = self.attacker.dataset
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

        final_metrics = None
        best_clean_metrics = None
        best_attack_metrics = None

        best_clean_state = None
        best_attack_state = None
        best_final_state = None
        best_clean_epoch = None
        best_attack_epoch = None

        for epoch in range(self.num_epochs):
            model.train()
            loss_meter = self._init_stage2_loss_meter()
            pbar = tqdm.tqdm(self.train_loader, desc=f'Training new forecasting model {epoch}/{self.num_epochs}')

            for batch_index, batch_data in enumerate(pbar):
                encoder_inputs, labels, clean_labels, x_mark, y_mark, idx = self._unpack_batch(batch_data)

                optimizer.zero_grad()
                outputs = self._forward_with_optional_modal_ablation(
                    model,
                    encoder_inputs,
                    labels,
                    x_mark,
                    y_mark,
                    idx,
                    self.train_set,
                    model_role='victim',
                )

                loss_per_timestep = F.smooth_l1_loss(
                    outputs[:, :, self.target_idx],
                    labels[:, :, self.target_idx],
                    reduction='none'
                )
                loss_per_sample = loss_per_timestep.mean(dim=1)
                loss = loss_per_sample.mean()
                self._update_stage2_loss_meter(loss_meter, loss_per_timestep, loss_per_sample, idx)
                loss.backward()
                optimizer.step()

            metrics = self.validate(model, epoch, 0)
            self._append_stage2_loss_log(epoch, loss_meter, metrics)
            final_metrics = metrics
            best_final_state = copy.deepcopy(model.state_dict())

            prev_best_clean = best_clean_metrics
            best_clean_metrics = self._update_best_metrics(best_clean_metrics, metrics, key='clean_rmse_norm')
            if prev_best_clean is None or best_clean_metrics['epoch'] == epoch:
                best_clean_state = copy.deepcopy(model.state_dict())
                best_clean_epoch = epoch

            if metrics.get('attacked_mae_norm', None) is not None:
                prev_best_attack = best_attack_metrics
                best_attack_metrics = self._update_best_metrics(best_attack_metrics, metrics, key='attacked_mae_norm')
                if prev_best_attack is None or best_attack_metrics['epoch'] == epoch:
                    best_attack_state = copy.deepcopy(model.state_dict())
                    best_attack_epoch = epoch

        self._print_test_summary(final_metrics, best_clean_metrics, best_attack_metrics)

        self.save_prediction_figures(model, num_samples=10, prefix='final')

        if best_clean_state is not None:
            model.load_state_dict(best_clean_state)
            self.save_prediction_figures(
                model,
                num_samples=10,
                prefix=f'bestclean_e{best_clean_epoch:03d}'
            )

        if best_attack_state is not None:
            model.load_state_dict(best_attack_state)
            self.save_prediction_figures(
                model,
                num_samples=10,
                prefix=f'bestattack_e{best_attack_epoch:03d}'
            )

        if save_poisoned_ckpt_path is not None and best_attack_state is not None:
            os.makedirs(os.path.dirname(save_poisoned_ckpt_path), exist_ok=True)
            torch.save(
                {
                    'model_state_dict': best_attack_state,
                    'best_epoch': best_attack_epoch,
                    'best_metrics': best_attack_metrics,
                    'final_state_dict': best_final_state,
                    'final_metrics': final_metrics,
                    'config_model': dict(self.config.Model),
                    'target_idx': self.target_idx,
                },
                save_poisoned_ckpt_path
            )
            print(f'[CKPT] Saved poisoned victim ckpt to {save_poisoned_ckpt_path}')

        return {
            'best_attack_state': best_attack_state,
            'best_attack_epoch': best_attack_epoch,
            'best_attack_metrics': best_attack_metrics,
            'best_clean_state': best_clean_state,
            'best_clean_epoch': best_clean_epoch,
            'best_clean_metrics': best_clean_metrics,
            'final_metrics': final_metrics,
        }
