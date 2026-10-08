import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import yaml
from easydict import EasyDict as edict

from dataset import load_raw_data, TimeDataset
from forecast_models.PatchTST import Model as PatchTST


def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_attack_vars(train_data_seq, target_idx, extra_ratio=0.0):
    num_vars = train_data_seq.shape[1]
    feature_indices = [i for i in range(num_vars) if i != target_idx]

    extra_num = int(np.floor(len(feature_indices) * extra_ratio + 0.5))
    extra_num = max(0, min(extra_num, len(feature_indices)))

    if extra_num == 0:
        return [int(target_idx)]

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
        corr_scores.append((feat_idx, abs(corr)))

    corr_scores.sort(key=lambda x: x[1], reverse=True)
    selected_extra = [item[0] for item in corr_scores[:extra_num]]
    return [int(target_idx)] + [int(i) for i in selected_extra]


def unpack_batch(batch, device, seq_len):
    if len(batch) == 6:
        x, y, clean_y, idx, x_mark, y_mark = batch
        x_mark = x_mark.to(device)
        y_mark = y_mark.to(device)
    elif len(batch) == 4:
        x, y, clean_y, idx = batch
        x_mark = torch.zeros(x.shape[0], seq_len, 5, device=device)
        y_mark = torch.zeros(y.shape[0], y.shape[1], 5, device=device)
    else:
        raise ValueError(f'Unexpected batch format with len={len(batch)}')

    x = x.to(device)
    y = y.to(device)
    return x, y, x_mark, y_mark


def patch_importance_to_timestep_importance(patch_importance, seq_len, patch_len, stride):
    patch_importance = np.asarray(patch_importance, dtype=np.float64)
    num_patch = len(patch_importance)

    time_score = np.zeros(seq_len, dtype=np.float64)
    time_count = np.zeros(seq_len, dtype=np.float64)

    for p in range(num_patch):
        start = p * stride
        end = start + patch_len
        if start >= seq_len:
            continue
        end = min(end, seq_len)
        time_score[start:end] += patch_importance[p]
        time_count[start:end] += 1.0

    time_count[time_count == 0] = 1.0
    time_score = time_score / time_count
    return time_score


def locate_single_trigger_segment(time_importance, threshold_ratio=0.7,
                                  min_trigger_len=4, max_trigger_len=16,
                                  min_start_idx=0):
    score = np.asarray(time_importance, dtype=np.float64)
    peak_idx = int(np.argmax(score))
    peak_val = float(score[peak_idx])

    if peak_val <= 0:
        center = max(min_start_idx, len(score) // 2)
        start = max(min_start_idx, center - min_trigger_len // 2)
        end = min(len(score), start + min_trigger_len)
        return start, end

    threshold = peak_val * float(threshold_ratio)

    left = peak_idx
    while left - 1 >= min_start_idx and score[left - 1] >= threshold:
        left -= 1

    right = peak_idx
    while right + 1 < len(score) and score[right + 1] >= threshold:
        right += 1

    cur_len = right - left + 1

    if cur_len < min_trigger_len:
        need = min_trigger_len - cur_len
        ext_left = need // 2
        ext_right = need - ext_left
        left = max(min_start_idx, left - ext_left)
        right = min(len(score) - 1, right + ext_right)

        cur_len = right - left + 1
        if cur_len < min_trigger_len:
            if left == min_start_idx:
                right = min(len(score) - 1, left + min_trigger_len - 1)
            elif right == len(score) - 1:
                left = max(min_start_idx, right - min_trigger_len + 1)

    cur_len = right - left + 1
    if cur_len > max_trigger_len:
        half = max_trigger_len // 2
        left = max(min_start_idx, peak_idx - half)
        right = left + max_trigger_len - 1
        if right >= len(score):
            right = len(score) - 1
            left = max(min_start_idx, right - max_trigger_len + 1)

    return int(left), int(right + 1)


def _find_local_peaks(score, min_start_idx=0):
    """
    简单局部峰检测：
    峰值点需满足 score[i] >= score[i-1] 且 score[i] >= score[i+1]
    """
    score = np.asarray(score, dtype=np.float64)
    peaks = []

    for i in range(max(1, min_start_idx), len(score) - 1):
        if score[i] >= score[i - 1] and score[i] >= score[i + 1]:
            peaks.append(i)

    # 边界可选处理
    if len(score) > 0 and min_start_idx == 0:
        if len(score) >= 2 and score[0] >= score[1]:
            peaks = [0] + peaks
    if len(score) >= 2 and score[-1] >= score[-2]:
        peaks.append(len(score) - 1)

    return peaks


def _interval_distance(seg1, seg2):
    """
    seg = (start, end), 半开区间 [start, end)
    若重叠/相邻则距离 <= 0
    """
    s1, e1 = seg1
    s2, e2 = seg2
    if e1 <= s2:
        return s2 - e1
    elif e2 <= s1:
        return s1 - e2
    else:
        return -1


def locate_multi_trigger_segments(
    time_importance,
    threshold_ratio=0.7,
    min_trigger_len=4,
    max_trigger_len=16,
    min_start_idx=0,
    max_trigger_segments=3,
    min_segment_gap=4,
    multi_peak_ratio=0.75,
):
    """
    多片段 trigger 选取：
    1. 先找单片段主片段
    2. 再找局部峰
    3. 仅保留足够高的次峰
    4. 每个峰独立阈值扩展成候选片段
    5. 子片段长度不超过主片段长度
    6. 与已有片段间隔不足 min_segment_gap 的丢弃
    7. 最多保留 max_trigger_segments 段
    """
    score = np.asarray(time_importance, dtype=np.float64)

    main_start, main_end = locate_single_trigger_segment(
        score,
        threshold_ratio=threshold_ratio,
        min_trigger_len=min_trigger_len,
        max_trigger_len=max_trigger_len,
        min_start_idx=min_start_idx
    )
    main_segment = (main_start, main_end)
    main_len = main_end - main_start

    peak_idx = int(np.argmax(score))
    main_peak_val = float(score[peak_idx])

    if max_trigger_segments <= 1 or main_peak_val <= 0:
        return [main_segment], main_segment

    local_peaks = _find_local_peaks(score, min_start_idx=min_start_idx)

    # 按峰值从大到小排序
    peak_items = [(p, float(score[p])) for p in local_peaks]
    peak_items.sort(key=lambda x: x[1], reverse=True)

    segments = [main_segment]
    used_peaks = {peak_idx}

    for p_idx, p_val in peak_items:
        if len(segments) >= max_trigger_segments:
            break

        # 跳过主峰附近/主峰本身
        if abs(p_idx - peak_idx) < min_segment_gap:
            continue

        # 次峰必须足够高，否则宁缺毋滥
        if p_val < multi_peak_ratio * main_peak_val:
            continue

        # 以该峰为中心做阈值扩展
        threshold = p_val * float(threshold_ratio)

        left = p_idx
        while left - 1 >= min_start_idx and score[left - 1] >= threshold:
            left -= 1

        right = p_idx
        while right + 1 < len(score) and score[right + 1] >= threshold:
            right += 1

        cand_len = right - left + 1

        # 至少 min_trigger_len
        if cand_len < min_trigger_len:
            need = min_trigger_len - cand_len
            ext_left = need // 2
            ext_right = need - ext_left
            left = max(min_start_idx, left - ext_left)
            right = min(len(score) - 1, right + ext_right)

        # 子片段长度不能超过主片段长度，也不能超过 max_trigger_len
        cand_max_len = min(main_len, max_trigger_len)
        cand_len = right - left + 1
        if cand_len > cand_max_len:
            half = cand_max_len // 2
            left = max(min_start_idx, p_idx - half)
            right = left + cand_max_len - 1
            if right >= len(score):
                right = len(score) - 1
                left = max(min_start_idx, right - cand_max_len + 1)

        cand_seg = (int(left), int(right + 1))

        # 与已选片段之间要有足够间隔
        good = True
        for seg in segments:
            if _interval_distance(cand_seg, seg) < min_segment_gap:
                good = False
                break

        if not good:
            continue

        segments.append(cand_seg)
        used_peaks.add(p_idx)

    # 按时间排序
    segments = sorted(segments, key=lambda x: x[0])

    return segments, main_segment


class TriggerLocator:
    """
    用 PatchTST 训练一个 probe model，并基于 OT 的 target-aware attention
    定位 trigger 在历史输入中的位置与长度。

    当前版本支持：
    - 单连续片段
    - 多连续片段（默认可关闭）
    - 最后一层 attention
    - 多头平均
    - target 区间聚合
    - 使用 OT 决定最终 trigger 位置
    """

    def __init__(
        self,
        config,
        device,
        train_data_seq,
        test_data_seq,
        train_timestamps,
        test_timestamps,
        target_idx,
        data_columns=None,
    ):
        self.config = config
        self.device = device
        self.train_data_seq = train_data_seq
        self.test_data_seq = test_data_seq
        self.train_timestamps = train_timestamps
        self.test_timestamps = test_timestamps
        self.target_idx = int(target_idx)
        self.data_columns = data_columns

        self.seq_len = int(config.seq_len)
        self.pred_len = int(config.pred_len)
        self.points_per_day = int(config.points_per_day)
        self.sample_mode = str(getattr(config, 'sample_mode', 'generic'))
        self.sample_stride = int(getattr(config, 'sample_stride', 1))
        self.day_aligned = self.sample_mode == 'day_aligned'
        self.use_timestamps = bool(config.Dataset.use_timestamps)

        self.locator_epochs = int(getattr(config, 'locator_epochs', 20))
        self.locator_batch_size = int(getattr(config, 'locator_batch_size', 64))
        self.locator_lr = float(getattr(config, 'locator_lr', 1e-4))
        self.attn_threshold_ratio = float(getattr(config, 'attn_threshold_ratio', 0.7))
        self.min_trigger_len = int(getattr(config, 'min_trigger_len', 4))
        self.max_trigger_len = int(getattr(config, 'max_trigger_len', 16))
        self.target_start_in_pred = int(getattr(config, 'target_start_in_pred', 0))
        self.target_len = int(getattr(config, 'target_len', 7))
        self.extra_atk_ratio = float(getattr(config, 'extra_atk_ratio', 0.0))
        self.use_selected_vars_only = bool(getattr(config, 'locator_use_selected_vars_only', False))
        self.bef_tgr_len = int(getattr(config, 'bef_tgr_len', 6))

        # ---------- 多片段新增超参数 ----------
        self.multi_trigger = bool(getattr(config, 'multi_trigger', False))
        self.max_trigger_segments = int(getattr(config, 'max_trigger_segments', 3))
        self.max_trigger_segments = max(1, min(self.max_trigger_segments, 3))
        self.min_segment_gap = int(getattr(config, 'min_segment_gap', 4))
        self.multi_peak_ratio = float(getattr(config, 'multi_peak_ratio', 0.75))

        self.model = None
        self.train_set = None
        self.test_set = None
        self.train_loader = None
        self.test_loader = None
        self.selected_vars = None
        self.var_importance_time = None
        self.trigger_segments = None

        self._build_probe()

    def _build_patchtst_config(self):
        default_config = yaml.load(open('configs/default_config.yaml', 'r'), Loader=yaml.FullLoader)
        model_cfg = default_config['Model']['PatchTST'].copy()

        num_vars = self.train_data_seq.shape[1]
        model_cfg['task_name'] = 'long_term_forecast'
        model_cfg['seq_len'] = self.seq_len
        model_cfg['pred_len'] = self.pred_len
        model_cfg['enc_in'] = num_vars
        model_cfg['dec_in'] = num_vars
        model_cfg['c_out'] = num_vars
        model_cfg['freq'] = 't' if self.points_per_day == 96 else 'h'
        if 'label_len' in model_cfg:
            model_cfg['label_len'] = max(1, self.pred_len // 2)
        if 'embed' not in model_cfg:
            model_cfg['embed'] = 'timeF'
        return edict(model_cfg)

    def _build_probe(self):
        model_cfg = self._build_patchtst_config()

        self.train_set = TimeDataset(
            self.train_data_seq,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            timestamps=self.train_timestamps,
            points_per_day=self.points_per_day,
            sample_mode=self.sample_mode,
            sample_stride=self.sample_stride,
            target_idx=self.target_idx,
            use_timestamps=self.use_timestamps,
        )

        self.test_set = TimeDataset(
            self.test_data_seq,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            timestamps=self.test_timestamps,
            points_per_day=self.points_per_day,
            sample_mode=self.sample_mode,
            sample_stride=self.sample_stride,
            target_idx=self.target_idx,
            use_timestamps=self.use_timestamps,
        )

        self.train_loader = DataLoader(self.train_set, batch_size=self.locator_batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_set, batch_size=self.locator_batch_size, shuffle=False)

        self.model = PatchTST(model_cfg).to(self.device)

    def fit(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.locator_lr)

        print('[TriggerLocator] Start training PatchTST probe model...')
        for epoch in range(self.locator_epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batch = 0

            for batch in self.train_loader:
                x, y, x_mark, y_mark = unpack_batch(batch, self.device, self.seq_len)

                optimizer.zero_grad()
                outputs = self.model(x, x_mark, None, y_mark)
                outputs = outputs[:, -y.shape[1]:, :]

                loss = F.smooth_l1_loss(outputs[:, :, self.target_idx], y[:, :, self.target_idx])
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())
                n_batch += 1

            val_mae, val_rmse = self.evaluate()
            print(f'[TriggerLocator][Epoch {epoch:03d}] '
                  f'train_loss={epoch_loss / max(n_batch,1):.6f} '
                  f'val_mae={val_mae:.6f} val_rmse={val_rmse:.6f}')

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        preds = []
        trues = []

        for batch in self.test_loader:
            x, y, x_mark, y_mark = unpack_batch(batch, self.device, self.seq_len)
            outputs = self.model(x, x_mark, None, y_mark)
            outputs = outputs[:, -y.shape[1]:, :]

            preds.append(outputs[:, :, self.target_idx].detach().cpu().numpy())
            trues.append(y[:, :, self.target_idx].detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        mae = np.mean(np.abs(preds - trues))
        rmse = np.sqrt(np.mean((preds - trues) ** 2))
        return float(mae), float(rmse)

    @torch.no_grad()
    def collect_attention_importance(self, selected_vars=None):
        self.model.eval()

        num_vars = self.model.enc_in
        if selected_vars is None:
            selected_vars = list(range(num_vars))

        var_importance_patch_sum = {v: None for v in selected_vars}
        var_counter = {v: 0 for v in selected_vars}

        patch_len = getattr(self.model.patch_embedding, 'patch_len', 16)
        stride = getattr(self.model.patch_embedding, 'stride', 8)

        for batch in self.test_loader:
            x, y, x_mark, y_mark = unpack_batch(batch, self.device, self.seq_len)
            outputs, attn_info = self.model(x, x_mark, None, y_mark, return_attn=True)

            last_attn = attn_info['last_attn']
            n_vars = int(attn_info['n_vars'])
            patch_num = int(attn_info['patch_num'])

            if last_attn.ndim == 4:
                last_attn = last_attn.mean(dim=1)
            elif last_attn.ndim != 3:
                raise ValueError(f'Unexpected attn shape: {last_attn.shape}')

            B = x.shape[0]
            last_attn = last_attn.reshape(B, n_vars, patch_num, patch_num)

            rel_start = self.target_start_in_pred / max(1, self.pred_len)
            rel_end = (self.target_start_in_pred + self.target_len) / max(1, self.pred_len)

            q_start = int(np.floor(rel_start * patch_num))
            q_end = int(np.ceil(rel_end * patch_num))
            q_start = max(0, min(q_start, patch_num - 1))
            q_end = max(q_start + 1, min(q_end, patch_num))

            patch_importance = last_attn[:, :, q_start:q_end, :].mean(dim=0).mean(dim=1)

            for v in selected_vars:
                vec = patch_importance[v].detach().cpu().numpy()
                if var_importance_patch_sum[v] is None:
                    var_importance_patch_sum[v] = np.zeros_like(vec, dtype=np.float64)
                var_importance_patch_sum[v] += vec
                var_counter[v] += 1

        var_importance_time = {}
        for v in selected_vars:
            avg_patch_imp = var_importance_patch_sum[v] / max(var_counter[v], 1)
            time_imp = patch_importance_to_timestep_importance(
                avg_patch_imp,
                seq_len=self.seq_len,
                patch_len=patch_len,
                stride=stride
            )
            var_importance_time[v] = time_imp

        self.var_importance_time = var_importance_time
        return var_importance_time

    def locate(self):
        self.fit()

        if self.use_selected_vars_only:
            self.selected_vars = select_attack_vars(
                self.train_data_seq,
                self.target_idx,
                extra_ratio=self.extra_atk_ratio
            )
            print(f'[TriggerLocator] analyze selected vars only: {self.selected_vars}')
        else:
            self.selected_vars = list(range(self.train_data_seq.shape[1]))
            print(f'[TriggerLocator] analyze all vars: {self.selected_vars}')

        self.collect_attention_importance(selected_vars=self.selected_vars)

        trigger_segments = {}
        trigger_details = {}

        for var_idx in self.selected_vars:
            if self.multi_trigger:
                segments, main_segment = locate_multi_trigger_segments(
                    self.var_importance_time[var_idx],
                    threshold_ratio=self.attn_threshold_ratio,
                    min_trigger_len=self.min_trigger_len,
                    max_trigger_len=self.max_trigger_len,
                    min_start_idx=self.bef_tgr_len,
                    max_trigger_segments=self.max_trigger_segments,
                    min_segment_gap=self.min_segment_gap,
                    multi_peak_ratio=self.multi_peak_ratio,
                )
            else:
                main_segment = locate_single_trigger_segment(
                    self.var_importance_time[var_idx],
                    threshold_ratio=self.attn_threshold_ratio,
                    min_trigger_len=self.min_trigger_len,
                    max_trigger_len=self.max_trigger_len,
                    min_start_idx=self.bef_tgr_len,
                )
                segments = [main_segment]

            trigger_segments[var_idx] = segments
            trigger_details[var_idx] = {
                'main_segment': main_segment,
                'segments': segments,
                'num_segments': len(segments),
            }

            name = self.data_columns[var_idx] if self.data_columns is not None else f'var_{var_idx}'
            print(
                f'[TriggerLocator] var={var_idx} ({name}) -> '
                f'main_segment={main_segment}, segments={segments}, num_segments={len(segments)}'
            )

        self.trigger_segments = trigger_segments

        # 最终只用 OT 的 target-aware attention 决定 trigger
        ot_detail = trigger_details[self.target_idx]
        final_main_start, final_main_end = ot_detail['main_segment']

        result = {
            'trigger_start_in_hist': int(final_main_start),
            'trigger_end_in_hist': int(final_main_end),
            'trigger_len': int(final_main_end - final_main_start),

            # 多片段新增
            'mode': 'multi' if len(ot_detail['segments']) > 1 else 'single',
            'main_segment': ot_detail['main_segment'],
            'segments': ot_detail['segments'],
            'num_segments': len(ot_detail['segments']),

            'target_var': int(self.target_idx),
            'all_segments': trigger_segments,
            'all_importance': self.var_importance_time,
        }

        print(
            f'[TriggerLocator] final trigger for OT -> '
            f'mode={result["mode"]}, '
            f'main_segment={result["main_segment"]}, '
            f'segments={result["segments"]}, '
            f'num_segments={result["num_segments"]}'
        )
        return result
