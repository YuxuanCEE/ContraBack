import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from math import ceil
from trigger import TgrGCN


def fft_compress(raw_data_seq, n_components=200):
    if len(raw_data_seq.shape) == 2:
        raw_data_seq = raw_data_seq[:, :, None]
    data_seq = raw_data_seq[:, :, 0:1]
    l, n, c = data_seq.shape
    data_seq = data_seq.reshape(l, -1).transpose()

    fft_data = np.fft.fft(data_seq, axis=1)
    amplitude = np.abs(fft_data)
    phase = np.angle(fft_data)
    frequency = np.fft.fftfreq(l)

    top_indices = np.argsort(amplitude, axis=1)[:, ::-1][:, :n_components]
    amplitude_top = amplitude[np.arange(amplitude.shape[0])[:, None], top_indices]
    phase_top = phase[np.arange(phase.shape[0])[:, None], top_indices]
    frequency_top = frequency[top_indices]
    feature_top = np.concatenate([amplitude_top, phase_top, frequency_top], axis=1)
    return feature_top


class FrequencyCarrier(nn.Module):
    """
    Frequency-domain trigger carrier for cross-modal propagation.
    Constructs high-frequency components in frequency domain that have small amplitude
    in time series but produce significant features in frequency/image modalities.
    """
    def __init__(self, seq_len, num_vars, high_freq_ratio=0.3, learnable=False, device='cuda'):
        super().__init__()
        self.seq_len = seq_len
        self.num_vars = num_vars
        self.high_freq_ratio = high_freq_ratio

        # Frequency mask: only keep high-frequency components
        freq_mask = torch.zeros(seq_len, device=device)
        cutoff = int(seq_len * (1 - high_freq_ratio))
        freq_mask[cutoff:] = 1.0

        # Initialize random amplitude and phase in high-frequency region
        amplitude = torch.randn(seq_len, num_vars, device=device) * freq_mask.unsqueeze(-1)
        phase = torch.randn(seq_len, num_vars, device=device) * 2 * np.pi * freq_mask.unsqueeze(-1)

        if learnable:
            self.amplitude = nn.Parameter(amplitude)
            self.phase = nn.Parameter(phase)
        else:
            self.register_buffer('amplitude', amplitude)
            self.register_buffer('phase', phase)

        self.register_buffer('freq_mask', freq_mask)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, num_vars] input time series
        Returns:
            carrier: [batch, seq_len, num_vars] frequency carrier in time domain
        """
        batch_size = x.shape[0]

        # Construct complex frequency representation
        real = self.amplitude * torch.cos(self.phase)
        imag = self.amplitude * torch.sin(self.phase)
        freq_carrier = torch.complex(real, imag)

        # Inverse FFT to time domain
        time_carrier = torch.fft.ifft(freq_carrier, dim=0).real  # [seq_len, num_vars]

        # Normalize to keep time-domain amplitude small
        max_amp = torch.abs(time_carrier).max()
        if max_amp > 1e-8:
            time_carrier = time_carrier / max_amp * 0.1

        # Expand to batch dimension
        return time_carrier.unsqueeze(0).expand(batch_size, -1, -1)


class Attacker:
    def __init__(self, dataset, channel_features, atk_vars, config, target_pattern, device='cuda'):
        self.device = device
        self.dataset = dataset

        self.target_pattern = target_pattern
        self.atk_vars = atk_vars

        if torch.is_tensor(atk_vars):
            self.primary_target_var = int(atk_vars[0].detach().cpu().item())
        else:
            self.primary_target_var = int(atk_vars[0])

        self.trigger_generator = TgrGCN(config, sim_feats=channel_features, atk_vars=atk_vars, device=device)

        self.trigger_len = int(config.trigger_len)
        self.pattern_len = int(config.pattern_len)
        self.bef_tgr_len = int(config.bef_tgr_len)

        self.fct_input_len = getattr(config.Dataset, 'len_input', config.seq_len)
        self.fct_output_len = getattr(config.Dataset, 'num_for_predict', config.pred_len)

        self.alpha_t = config.alpha_t
        self.alpha_s = config.alpha_s
        self.temporal_poison_num = ceil(self.alpha_t * len(self.dataset))

        # ---------- trigger / target 位置参数 ----------
        self.trigger_start_in_hist = int(getattr(config, 'trigger_start_in_hist', self.fct_input_len - self.trigger_len))
        self.target_start_in_pred = int(getattr(config, 'target_start_in_pred', 0))

        self.trigger_start_in_hist = max(self.bef_tgr_len, self.trigger_start_in_hist)
        self.trigger_start_in_hist = min(self.trigger_start_in_hist, self.fct_input_len - self.trigger_len)

        self.target_start_in_pred = max(0, self.target_start_in_pred)
        self.target_start_in_pred = min(self.target_start_in_pred, self.fct_output_len - 1)

        # ---------- 多片段支持 ----------
        self.multi_trigger = bool(getattr(config, 'multi_trigger', False))
        self.trigger_segments = getattr(config, 'trigger_segments', None)  # list of (start, end), 半开区间
        self.main_trigger_segment = getattr(config, 'main_trigger_segment', None)

        # 若未给多片段，则退化为单片段
        if self.trigger_segments is None:
            self.trigger_segments = [(self.trigger_start_in_hist, self.trigger_start_in_hist + self.trigger_len)]
        self.trigger_segments = [(int(s), int(e)) for s, e in self.trigger_segments]
        self.trigger_segments = sorted(self.trigger_segments, key=lambda x: x[0])

        if self.main_trigger_segment is None:
            self.main_trigger_segment = self.trigger_segments[-1]
        self.main_trigger_segment = (int(self.main_trigger_segment[0]), int(self.main_trigger_segment[1]))

        self.trigger_generator = self.trigger_generator.to(device)
        self.attack_optim = optim.Adam(self.trigger_generator.parameters(), lr=config.attack_lr)
        self.optimizer = self.attack_optim
        self.atk_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.attack_optim, milestones=[20, 40], gamma=0.9)

        self.lam_norm = config.lam_norm
        self.lam_cgh = float(getattr(config, 'lam_cgh', 0.1))
        self.lam_mag = float(getattr(config, 'lam_mag', 0.05))
        self.cgh_rho = float(getattr(config, 'cgh_rho', 1.0))

    def state_dict(self):
        attacker_state = {
            'target_pattern': self.target_pattern.cpu().detach().numpy(),
            'trigger_generator': self.trigger_generator.state_dict(),
            'trigger_len': self.trigger_len,
            'pattern_len': self.pattern_len,
            'bef_tgr_len': self.bef_tgr_len,
            'fct_input_len': self.fct_input_len,
            'fct_output_len': self.fct_output_len,
            'alpha_t': self.alpha_t,
            'alpha_s': self.alpha_s,
            'temporal_poison_num': self.temporal_poison_num,
            'lam_norm': self.lam_norm,
            'attack_optim': self.attack_optim.state_dict(),
            'atk_scheduler': self.atk_scheduler.state_dict(),
            'atk_ts': self.atk_ts.cpu().detach().numpy() if hasattr(self, 'atk_ts') else None,
            'atk_vars': self.atk_vars.cpu().detach().numpy() if hasattr(self, 'atk_vars') else None,
            'trigger_start_in_hist': self.trigger_start_in_hist,
            'target_start_in_pred': self.target_start_in_pred,
            'multi_trigger': self.multi_trigger,
            'trigger_segments': self.trigger_segments,
            'main_trigger_segment': self.main_trigger_segment,
        }
        return attacker_state

    def load_state_dict(self, attacker_state):
        self.trigger_len = attacker_state['trigger_len']
        self.pattern_len = attacker_state['pattern_len']
        self.bef_tgr_len = attacker_state['bef_tgr_len']
        self.fct_input_len = attacker_state['fct_input_len']
        self.fct_output_len = attacker_state['fct_output_len']
        self.alpha_t = attacker_state['alpha_t']
        self.alpha_s = attacker_state['alpha_s']
        self.temporal_poison_num = attacker_state['temporal_poison_num']
        self.lam_norm = attacker_state['lam_norm']

        self.trigger_start_in_hist = int(attacker_state.get('trigger_start_in_hist', self.fct_input_len - self.trigger_len))
        self.target_start_in_pred = int(attacker_state.get('target_start_in_pred', 0))
        self.multi_trigger = bool(attacker_state.get('multi_trigger', False))
        self.trigger_segments = attacker_state.get(
            'trigger_segments',
            [(self.trigger_start_in_hist, self.trigger_start_in_hist + self.trigger_len)]
        )
        self.trigger_segments = [(int(s), int(e)) for s, e in self.trigger_segments]
        self.main_trigger_segment = attacker_state.get('main_trigger_segment', self.trigger_segments[-1])
        self.main_trigger_segment = (int(self.main_trigger_segment[0]), int(self.main_trigger_segment[1]))

        self.trigger_generator.load_state_dict(attacker_state['trigger_generator'])
        self.attack_optim.load_state_dict(attacker_state['attack_optim'])
        self.atk_scheduler.load_state_dict(attacker_state['atk_scheduler'])
        self.target_pattern = torch.from_numpy(attacker_state['target_pattern'])
        self.atk_ts = torch.from_numpy(attacker_state['atk_ts']) if attacker_state['atk_ts'] is not None else None
        self.atk_vars = torch.from_numpy(attacker_state['atk_vars']) if attacker_state['atk_vars'] is not None else None

        self.trigger_generator = self.trigger_generator.to(self.device)
        self.target_pattern = self.target_pattern.to(self.device)
        if self.atk_ts is not None:
            self.atk_ts = self.atk_ts.to(self.device)
        if self.atk_vars is not None:
            self.atk_vars = self.atk_vars.to(self.device)
        if self.atk_ts is not None:
            self.atk_sample_ids = self.atk_ts.detach().cpu().numpy()

    def eval(self):
        self.trigger_generator.eval()

    def train(self):
        self.trigger_generator.train()

    def select_atk_timestamp(self, poison_metrics=None):
        import math
        import numpy as np
        import torch

        num_samples = len(self.dataset)
        poison_num = max(1, int(math.ceil(self.alpha_t * num_samples)))
        poison_num = min(poison_num, num_samples)

        if poison_metrics is None:
            candidate_ids = np.arange(num_samples)
            self.atk_sample_ids = np.random.choice(
                candidate_ids,
                size=poison_num,
                replace=False
            ).copy()
        else:
            metrics = np.asarray(poison_metrics).reshape(-1)

            if len(metrics) != num_samples:
                metrics = metrics[:num_samples]

            order = np.argsort(metrics)[::-1].copy()
            self.atk_sample_ids = order[:poison_num].copy()

        self.atk_ts = torch.as_tensor(
            self.atk_sample_ids.copy(),
            dtype=torch.long,
            device=self.device
        )

        print(f"[ContraBack] selected poisoned samples: {len(self.atk_sample_ids)} / {num_samples}")
        return self.atk_sample_ids

    def _format_pattern(self, device, dtype, num_atk_vars):
        pattern = self.target_pattern
        if not torch.is_tensor(pattern):
            pattern = torch.as_tensor(pattern, dtype=dtype, device=device)
        else:
            pattern = pattern.to(device=device, dtype=dtype)

        if pattern.ndim == 1:
            pattern = pattern[:self.pattern_len].view(self.pattern_len, 1)
            pattern = pattern.repeat(1, num_atk_vars)
        elif pattern.ndim == 2:
            if pattern.shape[0] == num_atk_vars:
                pattern = pattern[:, :self.pattern_len].transpose(0, 1)
            else:
                pattern = pattern[:self.pattern_len, :num_atk_vars]
        else:
            raise ValueError(f"Unsupported target_pattern shape: {pattern.shape}")

        return pattern

    def _call_trigger_generator(self, hist):
        hist = hist.to(self.device)
        B, A, L = hist.shape

        try:
            out = self.trigger_generator(hist)
        except Exception:
            out = self.trigger_generator(hist.reshape(B * A, L))

        if isinstance(out, tuple):
            triggers, perturbations = out
        else:
            triggers, perturbations = out, None

        if triggers.ndim == 2:
            triggers = triggers.reshape(B, A, -1)
        elif triggers.ndim == 3:
            pass
        else:
            raise ValueError(f"Unexpected trigger output shape: {triggers.shape}")

        if perturbations is not None:
            if perturbations.ndim == 2:
                perturbations = perturbations.reshape(B, A, -1)

        return triggers, perturbations

    def _get_hist_for_segment(self, x, atk_vars, seg_start):
        """
        对某个 trigger 片段，取其前面的 bef_tgr_len 历史作为上下文。
        若长度不足则左侧补齐。
        """
        hist_end = int(seg_start)
        hist_start = max(0, hist_end - self.bef_tgr_len)

        hist = x[:, hist_start:hist_end, :][:, :, atk_vars]   # [B, L, A]
        hist = hist.permute(0, 2, 1).contiguous()             # [B, A, L]

        if hist.shape[-1] < self.bef_tgr_len:
            pad_len = self.bef_tgr_len - hist.shape[-1]
            pad = hist[:, :, :1].repeat(1, 1, pad_len)
            hist = torch.cat([pad, hist], dim=-1)

        return hist

    def sparse_inject(self):
        if not hasattr(self, "atk_sample_ids"):
            self.select_atk_timestamp(None)

        self.dataset.clear_poisoned_samples()

        for sample_id in self.atk_sample_ids:
            sample_id = int(sample_id)
            x, y, _, _, _ = self.dataset.get_clean_sample(sample_id)

            device = next(self.trigger_generator.parameters()).device
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                x_poison, y_poison = self.build_poisoned_sample(x, y)

            self.dataset.set_poisoned_sample(
                sample_id,
                x_poison.detach().cpu(),
                y_poison.detach().cpu(),
            )

        print(f"[ContraBack] injected poisoned samples: {len(self.atk_sample_ids)}")

    def build_poisoned_batch(self, x, y):
        """
        支持多片段 trigger：
        - 每个片段独立生成 trigger（方案 A）
        - target pattern 只注入到 primary_target_var（OT）
        """
        original_device = x.device
        compute_device = self.device

        x = x.to(compute_device)
        y = y.to(compute_device)

        dtype = x.dtype
        B, seq_len, num_vars = x.shape
        pred_len = y.shape[1]

        atk_vars = torch.as_tensor(self.atk_vars, dtype=torch.long, device=compute_device)
        target_var = torch.as_tensor([self.primary_target_var], dtype=torch.long, device=compute_device)

        # ---------- 先做 trigger 注入 ----------
        x_mask = torch.zeros_like(x)
        x_trigger_values = torch.zeros_like(x)

        for seg_start, seg_end in self.trigger_segments:
            seg_start = int(seg_start)
            seg_end = int(seg_end)

            seg_start = max(self.bef_tgr_len, seg_start)
            seg_end = min(seq_len, seg_end)
            actual_len = seg_end - seg_start

            if actual_len <= 0:
                continue

            hist = self._get_hist_for_segment(x, atk_vars, seg_start)   # [B, A, bef_tgr_len]
            triggers, _ = self._call_trigger_generator(hist)
            triggers = triggers.to(device=compute_device, dtype=dtype)
            triggers = triggers[:, :, :actual_len]                     # [B, A, L]

            trigger_values = triggers.permute(0, 2, 1)                # [B, L, A]

            # Optional: mix with frequency carrier
            if hasattr(self, 'trigger_carrier') and self.trigger_carrier is not None:
                carrier_full = self.trigger_carrier(x)  # [B, seq_len, num_vars]
                # Extract carrier for attack variables and trigger segment
                carrier_slice = carrier_full[:, seg_start:seg_end, atk_vars]  # [B, L, A]
                alpha = float(getattr(self, 'trigger_carrier_alpha', 0.5))
                trigger_values = alpha * carrier_slice + (1.0 - alpha) * trigger_values

            for j, var_idx in enumerate(atk_vars):
                x_mask[:, seg_start:seg_end, var_idx] = 1.0
                x_trigger_values[:, seg_start:seg_end, var_idx] = trigger_values[:, :, j]

        x_poison = x * (1.0 - x_mask) + x_trigger_values * x_mask

        # ---------- 再做 target 注入 ----------
        y_mask = torch.zeros_like(y)
        y_target_values = torch.zeros_like(y)

        pattern = self._format_pattern(compute_device, dtype, 1)

        target_start = int(self.target_start_in_pred)
        target_end = min(pred_len, target_start + self.pattern_len)
        actual_target_len = target_end - target_start

        if target_start == 0:
            # 若 target 从预测段起点开始，基线取输入最后一个值
            base = x_poison[:, -1:, target_var]   # [B, 1, 1]
        else:
            # 若 target 位于预测段内部，基线取 target 前一时刻的干净未来值
            base = y[:, target_start - 1:target_start, target_var]   # [B, 1, 1]

        target_values = base + pattern[:actual_target_len].unsqueeze(0)

        y_mask[:, target_start:target_end, self.primary_target_var] = 1.0
        y_target_values[:, target_start:target_end, self.primary_target_var] = target_values[:, :, 0]

        y_poison = y * (1.0 - y_mask) + y_target_values * y_mask

        return x_poison.to(original_device), y_poison.to(original_device)

    def _extract_fusion_head_params(self, model):
        """Extract TimeVLM fusion head parameters (gate + fusion_layer)"""
        fusion_params = []
        for name, param in model.named_parameters():
            if 'gate.' in name or 'fusion_layer.' in name:
                if param.requires_grad:
                    fusion_params.append(param)
        return fusion_params

    def _compute_gradient_vector(self, loss, params, create_graph=False):
        """Compute gradient vector for given parameters"""
        if len(params) == 0:
            return None, 0.0
        if not loss.requires_grad:
            return None, 0.0

        grads = torch.autograd.grad(loss, params, create_graph=create_graph, allow_unused=True)
        flat_grads = []
        for grad in grads:
            if grad is None:
                continue
            flat_grads.append(grad.reshape(-1))

        if len(flat_grads) == 0:
            return None, 0.0

        grad_vec = torch.cat(flat_grads)
        grad_norm = float(torch.linalg.vector_norm(grad_vec).detach().item())
        return grad_vec, grad_norm

    def _compute_cgh_loss(self, model, x_bkd, y_bkd, x_mark, y_mark, idx, forward_fn, target_var, target_start, target_end):
        """
        Compute Cross-modal Gradient Harmonization loss.
        Returns: (L_cgh, L_mag, g_T_norm, g_all_norm) or (None, None, None, None) if not applicable
        """
        # Extract fusion head parameters
        fusion_params = self._extract_fusion_head_params(model)
        if len(fusion_params) == 0:
            return None, None, None, None

        # Forward with temporal-only mask (T1V0L0)
        try:
            outputs_T = forward_fn(model, x_bkd, y_bkd, x_mark, y_mark, idx, modal_mask='T1V0L0')
            if isinstance(outputs_T, tuple):
                outputs_T = outputs_T[0]
            outputs_T = outputs_T[:, -y_bkd.shape[1]:, :]

            loss_T = F.mse_loss(
                outputs_T[:, target_start:target_end, target_var],
                y_bkd[:, target_start:target_end, target_var]
            )
        except Exception:
            return None, None, None, None

        # Compute g_T (temporal-only gradient, stop gradient)
        g_T, g_T_norm = self._compute_gradient_vector(loss_T, fusion_params, create_graph=False)
        if g_T is None or g_T_norm < 1e-8:
            return None, None, None, None
        g_T = g_T.detach()

        # Forward with full multimodal (use current config mask)
        try:
            outputs_all = forward_fn(model, x_bkd, y_bkd, x_mark, y_mark, idx, modal_mask=None)
            if isinstance(outputs_all, tuple):
                outputs_all = outputs_all[0]
            outputs_all = outputs_all[:, -y_bkd.shape[1]:, :]

            loss_all = F.mse_loss(
                outputs_all[:, target_start:target_end, target_var],
                y_bkd[:, target_start:target_end, target_var]
            )
        except Exception:
            return None, None, None, None

        # Compute g_all (multimodal gradient, keep graph)
        g_all, g_all_norm = self._compute_gradient_vector(loss_all, fusion_params, create_graph=True)
        if g_all is None or g_all_norm < 1e-8:
            return None, None, None, None

        # L_CGH = 1 - cos(g_all, sg(g_T))
        cos_sim = F.cosine_similarity(g_all.unsqueeze(0), g_T.unsqueeze(0))
        L_cgh = 1.0 - cos_sim

        # L_mag = [ρ - ||g_all|| / (||g_T|| + ε)]²_+
        rho = self.cgh_rho
        epsilon = 1e-8
        ratio = g_all_norm / (g_T_norm + epsilon)
        mag_term = rho - ratio
        L_mag = torch.relu(mag_term) ** 2

        return L_cgh, L_mag, g_T_norm, g_all_norm

    def build_poisoned_sample(self, x, y):
        x_b = x.unsqueeze(0)
        y_b = y.unsqueeze(0)
        x_p, y_p = self.build_poisoned_batch(x_b, y_b)
        return x_p.squeeze(0), y_p.squeeze(0)

    def update_trigger_generator(self, model, criterion=None, batch_size=32, device=None, forward_fn=None, return_detail=False):
        import numpy as np

        if not hasattr(self, "atk_sample_ids") or len(self.atk_sample_ids) == 0:
            if return_detail:
                return {
                    'trigger_total_loss': None,
                    'trigger_attack_region_loss': None,
                    'trigger_norm_loss': None,
                    'trigger_num_steps': 0,
                }
            return 0.0

        if device is None:
            device = next(model.parameters()).device

        optimizer = getattr(self, "optimizer", None)
        if optimizer is None:
            optimizer = getattr(self, "trigger_optimizer", None)
        if optimizer is None:
            optimizer = getattr(self, "tgr_optimizer", None)

        if optimizer is None:
            raise AttributeError(
                "Cannot find optimizer for trigger generator. "
                "Please set self.optimizer / self.trigger_optimizer / self.tgr_optimizer."
            )

        model.eval()
        self.trigger_generator.train()

        sample_ids = np.asarray(self.atk_sample_ids)
        np.random.shuffle(sample_ids)

        total_loss = 0.0
        attack_loss_total = 0.0
        norm_loss_total = 0.0
        steps = 0

        for beg in range(0, len(sample_ids), batch_size):
            batch_ids = sample_ids[beg:beg + batch_size]

            xs, ys = [], []
            x_marks, y_marks = [], []
            has_mark = False

            for sid in batch_ids:
                x, y, xm, ym, _ = self.dataset.get_clean_sample(int(sid))
                xs.append(x)
                ys.append(y)

                if xm is not None and ym is not None:
                    has_mark = True
                    x_marks.append(xm)
                    y_marks.append(ym)

            x = torch.stack(xs, dim=0).to(device)
            y = torch.stack(ys, dim=0).to(device)

            if has_mark:
                x_mark = torch.stack(x_marks, dim=0).to(device)
                y_mark = torch.stack(y_marks, dim=0).to(device)
            else:
                x_mark = None
                y_mark = None
            idx = torch.as_tensor(batch_ids, dtype=torch.long, device=device)

            x_bkd, y_bkd = self.build_poisoned_batch(x, y)

            optimizer.zero_grad()

            if forward_fn is not None:
                outputs = forward_fn(model, x_bkd, y_bkd, x_mark, y_mark, idx)
            else:
                try:
                    outputs = model(x_bkd, x_mark, None, y_mark)
                except TypeError:
                    try:
                        outputs = model(x_bkd, x_mark, y_mark)
                    except TypeError:
                        outputs = model(x_bkd)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            outputs = outputs[:, -y_bkd.shape[1]:, :]

            target_var = torch.as_tensor([self.primary_target_var], dtype=torch.long, device=device)

            target_start = int(self.target_start_in_pred)
            target_end = min(y_bkd.shape[1], target_start + self.pattern_len)

            loss_bkd = F.mse_loss(
                outputs[:, target_start:target_end, target_var],
                y_bkd[:, target_start:target_end, target_var]
            )

            loss = loss_bkd

            norm_loss = torch.zeros((), dtype=loss_bkd.dtype, device=device)
            lam_norm = getattr(self, "lam_norm", 0.0)
            if lam_norm is not None and lam_norm > 0:
                norm_loss = lam_norm * torch.mean(torch.abs(x_bkd - x))
                loss = loss + norm_loss

            # Cross-modal Gradient Harmonization (CGH) loss
            cgh_loss = torch.zeros((), dtype=loss_bkd.dtype, device=device)
            mag_loss = torch.zeros((), dtype=loss_bkd.dtype, device=device)
            if forward_fn is not None and self.lam_cgh > 0:
                L_cgh, L_mag, g_T_norm, g_all_norm = self._compute_cgh_loss(
                    model, x_bkd, y_bkd, x_mark, y_mark, idx,
                    forward_fn, target_var, target_start, target_end
                )
                if L_cgh is not None:
                    cgh_loss = self.lam_cgh * L_cgh
                    loss = loss + cgh_loss
                if L_mag is not None and self.lam_mag > 0:
                    mag_loss = self.lam_mag * L_mag
                    loss = loss + mag_loss

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            attack_loss_total += float(loss_bkd.detach().item())
            norm_loss_total += float(norm_loss.detach().item())
            steps += 1

        model.train()
        if return_detail:
            denom = max(steps, 1)
            return {
                'trigger_total_loss': total_loss / denom if steps > 0 else None,
                'trigger_attack_region_loss': attack_loss_total / denom if steps > 0 else None,
                'trigger_norm_loss': norm_loss_total / denom if steps > 0 else None,
                'trigger_num_steps': steps,
            }
        return total_loss / max(steps, 1)
