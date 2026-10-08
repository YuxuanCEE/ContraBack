import os

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.Embed import PatchEmbedding


class PatchMemoryBank(nn.Module):
    def __init__(self, max_size, feature_dim):
        super().__init__()
        self.max_size = int(max_size)
        self.feature_dim = int(feature_dim)
        self.register_buffer('patches', torch.zeros(self.max_size, self.feature_dim))
        self.ptr = 0

    @torch.no_grad()
    def update(self, new_patches):
        new_patches = new_patches.mean(dim=1).detach()
        n = min(new_patches.size(0), self.max_size)
        new_patches = new_patches[:n]

        end = self.ptr + n
        if end <= self.max_size:
            self.patches[self.ptr:end] = new_patches
        else:
            first = self.max_size - self.ptr
            self.patches[self.ptr:] = new_patches[:first]
            self.patches[:end - self.max_size] = new_patches[first:]
        self.ptr = end % self.max_size

    def retrieve(self, query_patches, top_k=5):
        query_flat = query_patches.mean(dim=1)
        top_k = min(int(top_k), self.max_size)
        similarity = torch.matmul(query_flat, self.patches.T)
        _, indices = similarity.topk(top_k, dim=-1)
        return self.patches[indices]


class LearnableTimeSeriesToImage(nn.Module):
    def __init__(self, hidden_dim, output_channels, image_size, periodicity):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.output_channels = int(output_channels)
        self.image_size = int(image_size)
        self.periodicity = int(periodicity)
        self.conv1d = nn.Conv1d(in_channels=4, out_channels=self.hidden_dim, kernel_size=3, padding=1)
        self.conv2d_1 = nn.Conv2d(self.hidden_dim, self.hidden_dim // 2, kernel_size=3, padding=1)
        self.conv2d_2 = nn.Conv2d(self.hidden_dim // 2, self.output_channels, kernel_size=3, padding=1)

    def forward(self, x_enc):
        B, L, D = x_enc.shape
        steps = torch.arange(L, dtype=torch.float32, device=x_enc.device).unsqueeze(0).repeat(B, 1)
        periodicity = max(1, self.periodicity)
        periodicity_encoding = torch.cat([
            torch.sin(steps / periodicity * (2 * torch.pi)).unsqueeze(-1),
            torch.cos(steps / periodicity * (2 * torch.pi)).unsqueeze(-1),
        ], dim=-1)
        periodicity_encoding = periodicity_encoding.unsqueeze(-2).repeat(1, 1, D, 1)

        x_fft = torch.fft.rfft(x_enc, dim=1).abs()
        if x_fft.shape[1] < L:
            pad = torch.zeros(B, L - x_fft.shape[1], D, device=x_enc.device, dtype=x_fft.dtype)
            x_fft = torch.cat([x_fft, pad], dim=1)
        x_fft = x_fft[:, :L].unsqueeze(-1)

        x = torch.cat([x_enc.unsqueeze(-1), x_fft, periodicity_encoding], dim=-1)
        x = x.permute(0, 2, 3, 1).reshape(B * D, 4, L)
        x = self.conv1d(x).reshape(B, D, self.hidden_dim, L)
        x = x.permute(0, 2, 1, 3)
        x = F.relu(self.conv2d_1(x))
        x = F.relu(self.conv2d_2(x))
        return F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)


def time_series_to_simple_image(x_enc, image_size, context_len, periodicity):
    B, seq_len, nvars = x_enc.shape
    periodicity = max(1, int(periodicity))
    pad_left = 0 if context_len % periodicity == 0 else periodicity - context_len % periodicity
    x = einops.rearrange(x_enc, 'b s n -> b n s')
    x = F.pad(x, (pad_left, 0), mode='replicate')
    x = einops.rearrange(x, 'b n (p f) -> (b n) 1 f p', f=periodicity)
    x = F.interpolate(x, size=(image_size, image_size), mode='bilinear', align_corners=False)
    x = einops.repeat(x, 'b 1 h w -> b c h w', c=3)
    x = einops.rearrange(x, '(b n) c h w -> b n c h w', b=B, n=nvars)
    return x.mean(dim=1)


class VLMManager:
    def __init__(self, config):
        self.config = config
        self.vlm_type = str(getattr(config, 'vlm_type', 'CLIP')).lower()
        self.device = self._acquire_device()
        self._init_vlm()

    def _acquire_device(self):
        use_gpu = bool(getattr(self.config, 'use_gpu', torch.cuda.is_available()))
        if use_gpu and torch.cuda.is_available():
            return torch.device(f"cuda:{int(getattr(self.config, 'gpu', 0))}")
        return torch.device('cpu')

    def _ensure_safetensors_checkpoint(self, model_path):
        safe_path = os.path.join(model_path, 'model.safetensors')
        if os.path.exists(safe_path):
            try:
                from safetensors import safe_open
                with safe_open(safe_path, framework='pt', device='cpu') as f:
                    metadata = f.metadata()
                if metadata is not None and metadata.get('format') == 'pt':
                    return
                print(f'[TimeVLM] Rebuilding {safe_path} because metadata is missing or invalid.')
                os.remove(safe_path)
            except Exception:
                print(f'[TimeVLM] Rebuilding unreadable safetensors checkpoint: {safe_path}')
                os.remove(safe_path)

        bin_path = os.path.join(model_path, 'pytorch_model.bin')
        if not os.path.exists(bin_path):
            return

        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise ImportError(
                'TimeVLM needs safetensors to load local CLIP weights with this transformers/torch version. '
                'Install it with: pip install safetensors'
            ) from exc

        state_dict = torch.load(bin_path, map_location='cpu')
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        state_dict = {k: v.contiguous() for k, v in state_dict.items() if torch.is_tensor(v)}
        save_file(state_dict, safe_path, metadata={'format': 'pt'})
        print(f'[TimeVLM] Converted local CLIP weights to {safe_path}')

    def _init_vlm(self):
        if self.vlm_type != 'clip':
            raise ValueError("This integration currently supports vlm_type='CLIP'.")

        from transformers import CLIPModel, CLIPProcessor

        model_path = getattr(self.config, 'vlm_model_path', './pretrained_models/clip-vit-base-patch32')
        self._ensure_safetensors_checkpoint(model_path)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model = CLIPModel.from_pretrained(
            model_path,
            output_hidden_states=True,
            use_safetensors=True,
        )
        self.model.to(self.device)
        self._set_requires_grad(self.model, bool(getattr(self.config, 'finetune_vlm', False)))
        self.hidden_size = 512
        self.max_input_text_length = 77
        learnable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'VLM learnable parameters: {learnable_params:,}')

    def _set_requires_grad(self, model, value):
        for param in model.parameters():
            param.requires_grad = value

    def process_inputs(self, images, prompts):
        images_for_processor = images.detach().cpu()
        with torch.set_grad_enabled(bool(getattr(self.config, 'finetune_vlm', False))):
            encoding = self.processor(
                images=images_for_processor,
                text=prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
            ).to(self.device)
            outputs = self.model(**encoding, output_hidden_states=True)
        return outputs.image_embeds, outputs.text_embeds


class Model(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self._ensure_config_defaults()
        self.task_name = self.config.task_name
        self.seq_len = int(self.config.seq_len)
        self.pred_len = int(self.config.pred_len)
        self.enc_in = int(self.config.enc_in)

        self.vlm_manager = VLMManager(self.config)
        self.patch_embedding = PatchEmbedding(
            self.config.d_model,
            self.config.patch_len,
            self.config.stride,
            self.config.padding,
            self.config.dropout,
        )
        self.head_nf = self.config.d_model * int((self.seq_len - self.config.patch_len) / self.config.stride + 2)
        self.flatten = nn.Flatten(start_dim=-2)
        self.memory_head = nn.Sequential(nn.Linear(self.head_nf, self.pred_len), nn.Dropout(self.config.dropout))
        self.temporal_head = nn.Sequential(nn.Linear(self.head_nf, self.config.d_model), nn.Dropout(self.config.dropout))
        self.multimodal_head = nn.Sequential(
            nn.Linear(self.config.d_model, self.pred_len),
            nn.LayerNorm(self.pred_len),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )
        self.multimodal_enhancement = nn.Sequential(
            nn.Linear(self.vlm_manager.hidden_size * 2, self.config.d_model),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.config.d_model,
            num_heads=self.config.timevlm_heads,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(self.pred_len * 2, self.pred_len),
            nn.GELU(),
            nn.Linear(self.pred_len, 2),
            nn.Softmax(dim=-1),
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.pred_len * 2, self.pred_len),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )
        self.local_memory_mlp = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model * 2),
            nn.GELU(),
            nn.Linear(self.config.d_model * 2, self.config.d_model),
        )
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=self.config.d_model,
            num_heads=self.config.timevlm_heads,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.patch_memory_bank = PatchMemoryBank(self.config.patch_memory_size, self.config.d_model)
        self.learnable_image_module = LearnableTimeSeriesToImage(
            hidden_dim=self.config.image_hidden_dim,
            output_channels=3 if self.config.three_channel_image else 1,
            image_size=self.config.image_size,
            periodicity=self.config.periodicity,
        )
        self.layer_norm = nn.LayerNorm(self.config.d_model)

    def _ensure_config_defaults(self):
        defaults = {
            'task_name': 'long_term_forecast',
            'd_model': 128,
            'dropout': 0.1,
            'patch_len': 16,
            'stride': 8,
            'padding': 8,
            'top_k': 5,
            'patch_memory_size': 100,
            'image_size': 224,
            'image_hidden_dim': 48,
            'periodicity': 96,
            'norm_const': 1.0,
            'three_channel_image': True,
            'learnable_image': True,
            'save_images': False,
            'save_image_dir': 'ts-images2/timevlm',
            'save_negative_image_dir': 'ts-images2/negative',
            'color_enhancement': 'blue_mapping_inverted',
            'finetune_vlm': False,
            'vlm_type': 'CLIP',
            'vlm_model_path': './pretrained_models/clip-vit-base-patch32',
            'use_gpu': torch.cuda.is_available(),
            'gpu': 0,
            'content': 'A custom multivariate time series dataset.',
            'timevlm_heads': 4,
        }
        for key, value in defaults.items():
            if not hasattr(self.config, key):
                setattr(self.config, key, value)
        heads = min(int(self.config.timevlm_heads), int(self.config.d_model))
        while int(self.config.d_model) % heads != 0 and heads > 1:
            heads -= 1
        self.config.timevlm_heads = heads

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
        mask=None,
        return_modal=False,
        x_vision=None,
        x_language=None,
    ):
        if self.task_name not in {'long_term_forecast', 'short_term_forecast'}:
            return None

        x_norm, means, stdev = self._normalize_input(x_enc)
        x_vision = x_enc if x_vision is None else x_vision
        x_language = x_enc if x_language is None else x_language
        x_vision_norm, _, _ = self._normalize_input(x_vision)
        x_language_norm, _, _ = self._normalize_input(x_language)

        images, negative_images = self.vision_augmented_learner(
            x_vision_norm,
            self.config.image_size,
            self.seq_len,
            self.config.periodicity,
        )
        prompts = self.text_augmented_learner(x_language_norm, self.config.content, self.pred_len, self.seq_len)
        vision_embeddings, text_embeddings = self.vlm_manager.process_inputs(images, prompts)
        vision_embeddings = vision_embeddings.to(x_enc.device)
        text_embeddings = text_embeddings.to(x_enc.device)

        predictions = self.forward_prediction(x_norm, vision_embeddings, text_embeddings)
        y = self._denormalize_output(predictions, means, stdev)

        self.last_modal_state = {
            'images': images.detach(),
            'negative_images': [img.detach() for img in negative_images],
            'vision_embeddings': vision_embeddings.detach(),
            'text_embeddings': text_embeddings.detach(),
            'prompts': prompts,
        }
        if return_modal:
            return y, self.last_modal_state
        return y

    def forward_prediction(self, x_enc, vision_embeddings, text_embeddings):
        B, _, n_vars = x_enc.shape
        patches, _ = self.patch_embedding(x_enc.transpose(1, 2))
        local_memory = self._compute_local_memory(patches)
        global_memory = self._compute_global_memory(patches)
        memory_features = local_memory + global_memory

        memory_features = self.flatten(memory_features)
        temporal_features = self.temporal_head(memory_features)
        memory_features = self.memory_head(memory_features)
        temporal_features = einops.rearrange(temporal_features, '(b n) d -> b n d', b=B, n=n_vars)
        memory_features = einops.rearrange(memory_features, '(b n) d -> b n d', b=B, n=n_vars)

        multimodal_features = torch.cat([vision_embeddings, text_embeddings], dim=-1)
        multimodal_features = self.multimodal_enhancement(multimodal_features)
        multimodal_features = multimodal_features.unsqueeze(1).expand(-1, n_vars, -1)
        multimodal_features = self.layer_norm(multimodal_features)

        temporal_features = F.normalize(temporal_features, dim=-1)
        multimodal_features = F.normalize(multimodal_features, dim=-1)
        multimodal_features, _ = self.cross_attention(
            query=temporal_features,
            key=multimodal_features,
            value=multimodal_features,
        )
        multimodal_features = self.layer_norm(multimodal_features)
        multimodal_features = self.multimodal_head(multimodal_features)

        combined_features = torch.cat([memory_features, multimodal_features], dim=-1)
        gate_weights = self.gate(combined_features)
        fused_features = (
            gate_weights[:, :, 0:1] * memory_features +
            gate_weights[:, :, 1:2] * multimodal_features
        )
        predictions = self.fusion_layer(torch.cat([memory_features, fused_features], dim=-1)) + memory_features
        return predictions.permute(0, 2, 1)

    def _compute_local_memory(self, patches):
        retrieved_patches = self.patch_memory_bank.retrieve(patches, top_k=self.config.top_k)
        local_memory = self.local_memory_mlp(retrieved_patches).mean(dim=1, keepdim=True)
        return local_memory + patches

    def _compute_global_memory(self, patches):
        global_memory, _ = self.memory_attention(query=patches, key=patches, value=patches)
        self.patch_memory_bank.update(patches)
        return global_memory.mean(dim=1, keepdim=True)

    def _normalize_input(self, x):
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        stdev = stdev / float(self.config.norm_const)
        return x / stdev, means, stdev

    def _denormalize_output(self, y, means, stdev):
        return y * stdev.repeat(1, self.pred_len, 1) + means.repeat(1, self.pred_len, 1)

    def text_augmented_learner(self, x_enc, description, pred_len, seq_len):
        prompts = []
        for b in range(x_enc.shape[0]):
            min_value = torch.min(x_enc[b]).item()
            max_value = torch.max(x_enc[b]).item()
            median_value = torch.median(x_enc[b]).item()
            trend = x_enc[b].diff(dim=0).sum().item()
            trend_direction = 'upward' if trend > 0 else 'downward'
            prompt = (
                'The time series is converted into an image using convolutional layers, highlighting trends, '
                'periodic patterns, and multi-scale features for forecasting. '
                f'Dataset: {description} '
                f'Task: Forecast the next {pred_len} steps using the past {seq_len} steps. '
                f'Input statistics: min value = {min_value:.3f}, max value = {max_value:.3f}, '
                f'median value = {median_value:.3f}, the overall trend is {trend_direction}.'
            )
            prompts.append(prompt[:self.vlm_manager.max_input_text_length * 8])
        return prompts

    def vision_augmented_learner(self, x_enc, image_size, context_len, periodicity):
        if self.config.learnable_image:
            images = self.learnable_image_module(x_enc)
        else:
            images = time_series_to_simple_image(x_enc, image_size, context_len, periodicity)

        images = self._smooth_images(images)
        images = self._normalize_images(images)
        images = self._enhance_images(images)
        negative_samples = self.generate_negative_samples(images) if self.config.save_images else []

        if self.config.save_images:
            self.save_images(images)
            self.save_negative_images(negative_samples)
        return images, negative_samples

    def _enhance_images(self, images):
        method = getattr(self.config, 'color_enhancement', None)
        if method == 'brightness_invert':
            return self._enhance_colors_brightness_invert(images)
        if method == 'blue_mapping':
            return self._enhance_colors_blue_mapping(images, invert=False)
        if method == 'blue_mapping_inverted':
            return self._enhance_colors_blue_mapping(images, invert=True)
        return images

    @staticmethod
    def _smooth_images(images, kernel_size=5, sigma=1.5):
        import math

        channels = images.size(1)
        kernel = torch.zeros(channels, 1, kernel_size, kernel_size, device=images.device)
        center = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                x, y = i - center, j - center
                kernel[:, 0, i, j] = math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return F.conv2d(images, kernel, groups=channels, padding=kernel_size // 2)

    @staticmethod
    def _normalize_images(images):
        min_vals = images.reshape(images.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        max_vals = images.reshape(images.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        scale = (max_vals - min_vals).clamp(min=1e-5)
        return ((images - min_vals) / scale * 255).clamp(0, 255).to(torch.uint8)

    @staticmethod
    def _enhance_colors_brightness_invert(images):
        images_float = images.float() / 255.0
        luminance = images_float.mean(dim=1, keepdim=True)
        enhanced = 1.0 - luminance
        if images_float.shape[1] == 3:
            enhanced = enhanced.repeat(1, 3, 1, 1)
        return (enhanced * 255).clamp(0, 255).to(torch.uint8)

    @staticmethod
    def _enhance_colors_blue_mapping(images, invert=False):
        images_float = images.float() / 255.0
        luminance = images_float.mean(dim=1, keepdim=True)
        if invert:
            luminance = 1.0 - luminance
        dark_blue = torch.tensor([20 / 255, 50 / 255, 100 / 255], device=images.device).view(1, 3, 1, 1)
        light_blue = torch.tensor([200 / 255, 230 / 255, 255 / 255], device=images.device).view(1, 3, 1, 1)
        enhanced = dark_blue + luminance * (light_blue - dark_blue)
        return (enhanced * 255).clamp(0, 255).to(torch.uint8)

    def generate_negative_samples(self, images):
        return [self.patch_swapping(images)] + self.color_jittering(images)

    @staticmethod
    def patch_swapping(images):
        _, _, _, width = images.shape
        part_width = width // 4
        parts = []
        for i in range(4):
            start = i * part_width
            end = (i + 1) * part_width if i < 3 else width
            parts.append(images[:, :, :, start:end])
        return torch.cat([parts[1], parts[0], parts[3], parts[2]], dim=3)

    @staticmethod
    def color_jittering(images):
        images_float = images.float() / 255.0
        jittered = []
        brightness = torch.clamp(images_float * 1.1, 0, 1)
        jittered.append((brightness * 255).to(torch.uint8))
        mean_val = images_float.mean(dim=(2, 3), keepdim=True)
        contrast = torch.clamp((images_float - mean_val) * 1.1 + mean_val, 0, 1)
        jittered.append((contrast * 255).to(torch.uint8))
        if images_float.shape[1] == 3:
            gray = images_float.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            saturation = torch.clamp(gray + (images_float - gray) * 1.1, 0, 1)
        else:
            saturation = images_float
        jittered.append((saturation * 255).to(torch.uint8))
        return jittered

    @torch.no_grad()
    def save_images(self, images):
        from PIL import Image

        save_dir = self.config.save_image_dir
        os.makedirs(save_dir, exist_ok=True)
        for i, img_tensor in enumerate(images):
            img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
            if img.shape[-1] == 1:
                img = img.squeeze(-1)
            Image.fromarray(img).save(os.path.join(save_dir, f'image_{i}.png'))

    @torch.no_grad()
    def save_negative_images(self, negative_samples):
        from PIL import Image

        save_dir = self.config.save_negative_image_dir
        os.makedirs(save_dir, exist_ok=True)
        for group_idx, neg_images in enumerate(negative_samples):
            for image_idx, img_tensor in enumerate(neg_images):
                img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
                if img.shape[-1] == 1:
                    img = img.squeeze(-1)
                Image.fromarray(img).save(os.path.join(save_dir, f'image_{image_idx}_{group_idx}.png'))
