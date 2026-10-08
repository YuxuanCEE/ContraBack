import json
import os

import numpy as np
import torch
import torch.nn.functional as F


class ClipEmbeddingExtractor:
    def __init__(self, config, device):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.model_path = getattr(config.Model, 'vlm_model_path', './pretrained_models/clip-vit-base-patch32')
        self._ensure_safetensors_checkpoint(self.model_path)
        self.processor = CLIPProcessor.from_pretrained(self.model_path)
        self.model = CLIPModel.from_pretrained(
            self.model_path,
            output_hidden_states=True,
            use_safetensors=True,
        ).to(device)
        self.model.eval()

    def _ensure_safetensors_checkpoint(self, model_path):
        safe_path = os.path.join(model_path, 'model.safetensors')
        if os.path.exists(safe_path):
            try:
                from safetensors import safe_open
                with safe_open(safe_path, framework='pt', device='cpu') as f:
                    metadata = f.metadata()
                if metadata is not None and metadata.get('format') == 'pt':
                    return
                os.remove(safe_path)
            except Exception:
                os.remove(safe_path)

        bin_path = os.path.join(model_path, 'pytorch_model.bin')
        if not os.path.exists(bin_path):
            return

        from safetensors.torch import save_file
        state_dict = torch.load(bin_path, map_location='cpu')
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        state_dict = {k: v.contiguous() for k, v in state_dict.items() if torch.is_tensor(v)}
        save_file(state_dict, safe_path, metadata={'format': 'pt'})

    @torch.no_grad()
    def encode_text(self, prompts, batch_size=32):
        outputs = []
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start:start + batch_size]
            inputs = self.processor(text=batch, return_tensors='pt', padding=True, truncation=True).to(self.device)
            emb = self.model.get_text_features(**inputs)
            emb = F.normalize(emb, dim=-1)
            outputs.append(emb.detach().cpu())
        return torch.cat(outputs, dim=0).numpy()

    @torch.no_grad()
    def encode_images(self, images, batch_size=32):
        outputs = []
        images = images.detach().cpu()
        for start in range(0, images.shape[0], batch_size):
            batch = images[start:start + batch_size]
            inputs = self.processor(images=batch, return_tensors='pt').to(self.device)
            emb = self.model.get_image_features(**inputs)
            emb = F.normalize(emb, dim=-1)
            outputs.append(emb.detach().cpu())
        return torch.cat(outputs, dim=0).numpy()


def save_modal_embeddings(trainer, config, num_samples=200, split='train', save_root='./ModalEmbeddings', seed=1):
    os.makedirs(save_root, exist_ok=True)
    rng = np.random.default_rng(int(seed))

    dataset = trainer.train_set if split == 'train' else trainer.cln_test_set
    total_num = len(dataset)
    num_samples = min(int(num_samples), total_num)
    if num_samples <= 0:
        print('[ModalEmbeddings] No samples available, skip saving embeddings.')
        return

    sample_ids = rng.choice(total_num, size=num_samples, replace=False)
    exp_tag = _get_exp_tag(trainer, config)
    save_dir = os.path.join(save_root, exp_tag)
    os.makedirs(save_dir, exist_ok=True)

    clean_xs = []
    poison_xs = []
    global_starts = []

    for sample_id in sample_ids:
        x_clean, y_clean, _, _, global_start = dataset.get_clean_sample(int(sample_id))
        x_poison, _ = trainer.attacker.build_poisoned_sample(x_clean.clone(), y_clean.clone())
        clean_xs.append(x_clean)
        poison_xs.append(x_poison)
        global_starts.append(int(global_start))

    x_clean = torch.stack(clean_xs, dim=0)
    x_poison = torch.stack(poison_xs, dim=0)

    content = _get_content(config)
    clean_prompts = [_build_timevlm_prompt(x_clean[i:i + 1], content, config.pred_len, config.seq_len) for i in range(num_samples)]
    poison_prompts = [_build_timevlm_prompt(x_poison[i:i + 1], content, config.pred_len, config.seq_len) for i in range(num_samples)]

    clean_images = _build_timevlm_images(x_clean, config)
    poison_images = _build_timevlm_images(x_poison, config)

    clip = ClipEmbeddingExtractor(config, trainer.device)
    text_clean = clip.encode_text(clean_prompts, batch_size=int(getattr(config, 'modal_embed_batch_size', 32)))
    text_poisoned = clip.encode_text(poison_prompts, batch_size=int(getattr(config, 'modal_embed_batch_size', 32)))
    image_clean = clip.encode_images(clean_images, batch_size=int(getattr(config, 'modal_embed_batch_size', 32)))
    image_poisoned = clip.encode_images(poison_images, batch_size=int(getattr(config, 'modal_embed_batch_size', 32)))

    time_clean = _time_embedding(x_clean)
    time_poisoned = _time_embedding(x_poison)
    frequency_all_clean = _frequency_embedding(x_clean)
    frequency_all_poisoned = _frequency_embedding(x_poison)

    atk_vars = trainer.attacker.atk_vars.detach().cpu().numpy().astype(int).tolist()
    frequency_atk_clean = _frequency_embedding(x_clean[:, :, atk_vars])
    frequency_atk_poisoned = _frequency_embedding(x_poison[:, :, atk_vars])

    embedding_path = os.path.join(save_dir, 'embeddings.npz')
    np.savez_compressed(
        embedding_path,
        time_clean=time_clean,
        time_poisoned=time_poisoned,
        text_clean=text_clean,
        text_poisoned=text_poisoned,
        image_clean=image_clean,
        image_poisoned=image_poisoned,
        frequency_all_clean=frequency_all_clean,
        frequency_all_poisoned=frequency_all_poisoned,
        frequency_atk_clean=frequency_atk_clean,
        frequency_atk_poisoned=frequency_atk_poisoned,
        sample_ids=np.asarray(sample_ids, dtype=np.int64),
        global_starts=np.asarray(global_starts, dtype=np.int64),
        atk_vars=np.asarray(atk_vars, dtype=np.int64),
    )

    metadata = {
        'dataset': str(config.dataset),
        'model_name': str(config.model_name),
        'surrogate_name': str(config.surrogate_name),
        'split': str(split),
        'num_samples': int(num_samples),
        'seq_len': int(trainer.seq_len),
        'pred_len': int(trainer.pred_len),
        'target_idx': int(trainer.target_idx),
        'atk_vars': atk_vars,
        'trigger_segments': [list(seg) for seg in trainer.attacker.trigger_segments],
        'trigger_len': int(trainer.attacker.trigger_len),
        'target_start_in_pred': int(trainer.attacker.target_start_in_pred),
        'modalities': {
            'time': 'flattened normalized multivariate time series',
            'text': 'CLIP text embedding of TimeVLM-style prompt',
            'image': 'CLIP image embedding of TimeVLM simple image',
            'frequency_all': 'FFT magnitude over all variables',
            'frequency_atk': 'FFT magnitude over attacked variables; diagnostic only',
        },
        'embedding_path': embedding_path,
    }
    with open(os.path.join(save_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    with open(os.path.join(save_dir, 'prompts_clean.txt'), 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(clean_prompts) + '\n')
    with open(os.path.join(save_dir, 'prompts_poisoned.txt'), 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(poison_prompts) + '\n')

    print(f'[ModalEmbeddings] Saved embeddings to {embedding_path}')


def _get_exp_tag(trainer, config):
    data_name = getattr(config.Dataset, 'data_filename', config.dataset)
    data_name = os.path.splitext(os.path.basename(str(data_name)))[0]
    return (
        f'{data_name}_{config.model_name}_sur{config.surrogate_name}_'
        f'sl{trainer.seq_len}_pl{trainer.pred_len}_'
        f'tr{trainer.attacker.trigger_len}_tps{trainer.attacker.target_start_in_pred}'
    )


def _get_content(config):
    if hasattr(config.Model, 'content'):
        return str(config.Model.content)
    dataset_name = str(getattr(config.Dataset, 'dataset_name', config.dataset))
    return f'{dataset_name} is a custom multivariate time series forecasting dataset.'


def _time_embedding(x):
    x = _normalize_series(x)
    return x.reshape(x.shape[0], -1).detach().cpu().numpy()


def _frequency_embedding(x):
    x = _normalize_series(x)
    freq = torch.fft.rfft(x.permute(0, 2, 1), dim=-1).abs()
    return freq.reshape(freq.shape[0], -1).detach().cpu().numpy()


def _normalize_series(x):
    mean = x.mean(dim=1, keepdim=True)
    std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
    return (x - mean) / std


def _build_timevlm_images(x, config):
    image_size = int(getattr(config.Model, 'image_size', 224))
    periodicity = int(getattr(config.Model, 'periodicity', getattr(config, 'points_per_day', 96)))
    x = _normalize_series(x)
    images = _time_series_to_simple_image(x, image_size, x.shape[1], periodicity)
    return _normalize_images_uint8(images)


def _time_series_to_simple_image(x_enc, image_size, context_len, periodicity):
    B, _, nvars = x_enc.shape
    periodicity = max(1, int(periodicity))
    pad_left = 0 if context_len % periodicity == 0 else periodicity - context_len % periodicity
    x = x_enc.permute(0, 2, 1)
    x = F.pad(x, (pad_left, 0), mode='replicate')
    x = x.reshape(B, nvars, -1, periodicity).permute(0, 1, 3, 2).reshape(B * nvars, 1, periodicity, -1)
    x = F.interpolate(x, size=(image_size, image_size), mode='bilinear', align_corners=False)
    x = x.repeat(1, 3, 1, 1)
    return x.reshape(B, nvars, 3, image_size, image_size).mean(dim=1)


def _normalize_images_uint8(images):
    flat = images.reshape(images.shape[0], -1)
    min_vals = flat.min(dim=1)[0].view(-1, 1, 1, 1)
    max_vals = flat.max(dim=1)[0].view(-1, 1, 1, 1)
    images = (images - min_vals) / (max_vals - min_vals + 1e-8)
    return (images * 255).clamp(0, 255).to(torch.uint8)


def _build_timevlm_prompt(x_enc, description, pred_len, seq_len):
    x = _normalize_series(x_enc)[0]
    min_value = torch.min(x).item()
    max_value = torch.max(x).item()
    median_value = torch.median(x).item()
    trend = x.diff(dim=0).sum().item()
    trend_direction = 'upward' if trend > 0 else 'downward'
    return (
        'The time series is converted into an image using convolutional layers, highlighting trends, '
        'periodic patterns, and multi-scale features for forecasting. '
        f'Dataset: {description} '
        f'Task: Forecast the next {int(pred_len)} steps using the past {int(seq_len)} steps. '
        f'Input statistics: min value = {min_value:.3f}, max value = {max_value:.3f}, '
        f'median value = {median_value:.3f}, the overall trend is {trend_direction}.'
    )
