import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _build_time_mark(timestamps):
    """
    简单时间特征编码。
    输入: pandas Series / DatetimeIndex / datetime list
    输出: [T, time_feat_dim]
    """
    ts = pd.to_datetime(timestamps)

    # 统一成 DatetimeIndex，避免 Series 没有 .month / .hour 属性的问题
    if isinstance(ts, pd.Series):
        ts = pd.DatetimeIndex(ts)
    elif not isinstance(ts, pd.DatetimeIndex):
        ts = pd.DatetimeIndex(ts)

    month = ts.month.astype(np.float32) / 12.0
    day = ts.day.astype(np.float32) / 31.0
    weekday = ts.weekday.astype(np.float32) / 6.0
    hour = ts.hour.astype(np.float32) / 23.0
    minute = ts.minute.astype(np.float32) / 59.0

    return np.stack([month, day, weekday, hour, minute], axis=1).astype(np.float32)



def _load_npz_data(dataset_config, data_filename, target_col, train_ratio):
    """
    PEMS-style .npz loader.

    File format: npz with key 'data', array shape [T, N] or [T, N, C]
    (only feature channel 0 is used). No timestamps, no explicit OT column;
    all N nodes are equivalent sensors.

    Returns the same 8-tuple contract as load_raw_data. The attack target
    node (target_idx) defaults to the last node (N-1); it can be overridden
    by setting target_col to an integer node index in the config.
    """
    npz = np.load(data_filename)
    key = 'data' if 'data' in npz else list(npz.keys())[0]
    data = np.asarray(npz[key])

    if data.ndim == 3:
        data = data[:, :, 0]
    elif data.ndim != 2:
        raise ValueError(f"Unsupported npz array shape {data.shape}; expected [T, N] or [T, N, C]")

    data = data.astype(np.float32)
    num_vars = data.shape[1]

    # target node: last node by default, or an integer index via target_col
    target_idx = num_vars - 1
    try:
        target_idx = int(target_col)
    except (TypeError, ValueError):
        target_idx = num_vars - 1
    if not (0 <= target_idx < num_vars):
        target_idx = num_vars - 1

    points_per_day = int(_cfg_get(dataset_config, "points_per_day", 96))
    split_idx = int(len(data) * train_ratio)
    split_idx = (split_idx // points_per_day) * points_per_day

    train_data = data[:split_idx]
    test_data = data[split_idx:]

    train_mean = train_data.mean(axis=0, keepdims=True)
    train_std = train_data.std(axis=0, keepdims=True)
    train_std[train_std < 1e-6] = 1.0

    train_data = (train_data - train_mean) / train_std
    test_data = (test_data - train_mean) / train_std

    cols = [f'node_{i}' for i in range(num_vars)]

    return (
        train_data.astype(np.float32),
        test_data.astype(np.float32),
        train_mean.squeeze(0).astype(np.float32),
        train_std.squeeze(0).astype(np.float32),
        None,   # train_timestamps
        None,   # test_timestamps
        target_idx,
        cols,
    )


def load_raw_data(dataset_config):
    """
    ETT-style csv loader.

    数据格式:
        第 1 列: date
        第 2 到倒数第 2 列: 特征列
        最后一列: OT, 目标列

    返回:
        train_data_seq: [T_train, N]
        test_data_seq:  [T_test, N]
        train_mean:     [N]
        train_std:      [N]
        train_timestamps
        test_timestamps
        target_idx
        columns
    """

    data_filename = (
            _cfg_get(dataset_config, "data_filename", None)
            or _cfg_get(dataset_config, "dataset", None)
            or _cfg_get(dataset_config, "filename", None)
            or _cfg_get(dataset_config, "data_path", None)
            or _cfg_get(dataset_config, "root_path", None)
    )

    if data_filename is None:
        raise ValueError(
            "Cannot find data filename in dataset_config. "
            "Please set data_filename / filename / data_path."
        )

    date_col = _cfg_get(dataset_config, "date_col", "date")
    target_col = _cfg_get(dataset_config, "target_col", "OT")
    train_ratio = float(_cfg_get(dataset_config, "train_ratio", 0.8))

    # ---- .npz branch (PEMS-style: no timestamps, no OT column) ----
    if str(data_filename).endswith('.npz'):
        return _load_npz_data(dataset_config, data_filename, target_col, train_ratio)

    if data_filename.endswith('.csv') or ('/' in data_filename) or ('\\' in data_filename):
        df = pd.read_csv(data_filename)
    else:
        df = pd.read_csv('./data/' + data_filename + '.csv')

    if date_col not in df.columns:
        raise ValueError(f"date_col={date_col} not found in csv columns: {df.columns.tolist()}")

    if target_col not in df.columns:
        raise ValueError(f"target_col={target_col} not found in csv columns: {df.columns.tolist()}")

    timestamps = pd.to_datetime(df[date_col])
    value_df = df.drop(columns=[date_col])

    # 保证 OT 在最后一列。如果原 csv 已经最后一列是 OT，这里不会改变语义。
    cols = [c for c in value_df.columns if c != target_col] + [target_col]
    value_df = value_df[cols]

    data = value_df.values.astype(np.float32)
    target_idx = len(cols) - 1

    points_per_day = int(_cfg_get(dataset_config, "points_per_day", 96))
    split_idx = int(len(data) * train_ratio)

    # 向下对齐到自然日边界
    split_idx = (split_idx // points_per_day) * points_per_day

    train_data = data[:split_idx]
    test_data = data[split_idx:]

    train_timestamps = timestamps.iloc[:split_idx].reset_index(drop=True)
    test_timestamps = timestamps.iloc[split_idx:].reset_index(drop=True)

    # 按变量标准化，比全局标准化更适合负荷/风光/气象混合数据。
    train_mean = train_data.mean(axis=0, keepdims=True)
    train_std = train_data.std(axis=0, keepdims=True)
    train_std[train_std < 1e-6] = 1.0

    train_data = (train_data - train_mean) / train_std
    test_data = (test_data - train_mean) / train_std

    return (
        train_data.astype(np.float32),
        test_data.astype(np.float32),
        train_mean.squeeze(0).astype(np.float32),
        train_std.squeeze(0).astype(np.float32),
        train_timestamps,
        test_timestamps,
        target_idx,
        cols,
    )


class TimeDataset(Dataset):
    """
    Time series forecasting dataset with configurable sample construction.

    返回:
        如果有 timestamps:
            x, y_poison_or_clean, y_clean, sample_id, x_mark, y_mark
        否则:
            x, y_poison_or_clean, y_clean, sample_id

    其中:
        x:       [seq_len, num_vars]
        y:       [pred_len, num_vars]
        x_mark:  [seq_len, time_feat_dim]
        y_mark:  [pred_len, time_feat_dim]
    """

    def __init__(
        self,
        data,
        num_for_hist=None,
        num_for_futr=None,
        timestamps=None,
        seq_len=None,
        pred_len=None,
        points_per_day=96,
        day_aligned=None,
        sample_mode='generic',
        sample_stride=1,
        target_idx=-1,
        use_timestamps=True,
    ):
        super().__init__()

        self.data = torch.as_tensor(data, dtype=torch.float32)
        if self.data.ndim != 2:
            raise ValueError(f"TimeDataset expects data shape [T, N], got {self.data.shape}")

        self.seq_len = int(seq_len if seq_len is not None else num_for_hist)
        self.pred_len = int(pred_len if pred_len is not None else num_for_futr)
        self.points_per_day = int(points_per_day)
        self.sample_mode = str(sample_mode).lower()
        if day_aligned is not None:
            self.sample_mode = 'day_aligned' if bool(day_aligned) else 'generic'
        if self.sample_mode not in {'generic', 'day_aligned'}:
            raise ValueError(f"Unsupported sample_mode={sample_mode}. Use 'generic' or 'day_aligned'.")
        self.day_aligned = self.sample_mode == 'day_aligned'
        self.sample_stride = int(sample_stride)
        if self.sample_stride <= 0:
            raise ValueError(f"sample_stride must be positive, got {sample_stride}")
        self.target_idx = int(target_idx)
        self.use_timestamps = bool(use_timestamps and timestamps is not None)

        self.timestamps = None
        self.time_marks = None

        if self.use_timestamps:
            self.timestamps = pd.to_datetime(timestamps).reset_index(drop=True)
            self.time_marks = torch.as_tensor(_build_time_mark(self.timestamps), dtype=torch.float32)

        self.sample_starts = self._build_sample_starts()

        # sample_id -> (poisoned_x, poisoned_y)
        self.poisoned_samples = {}

    def _is_day_boundary(self, idx):
        if not self.use_timestamps:
            return idx % self.points_per_day == 0

        ts = self.timestamps.iloc[idx]
        return ts.hour == 0 and ts.minute == 0

    def _build_sample_starts(self):
        max_start = len(self.data) - self.seq_len - self.pred_len
        if max_start < 0:
            raise ValueError(
                f"Data length {len(self.data)} is shorter than seq_len + pred_len = "
                f"{self.seq_len + self.pred_len}"
            )

        starts = []

        if self.sample_mode == 'day_aligned':
            for idx in range(0, max_start + 1):
                if self._is_day_boundary(idx):
                    starts.append(idx)
        else:
            starts = list(range(0, max_start + 1, self.sample_stride))

        if len(starts) == 0:
            raise ValueError(
                "No valid samples found. "
                "Please check seq_len, pred_len, sample_mode, sample_stride, points_per_day and date alignment."
            )

        return starts

    def __len__(self):
        return len(self.sample_starts)

    def clear_poisoned_samples(self):
        self.poisoned_samples = {}

    def set_poisoned_sample(self, sample_id, poisoned_x, poisoned_y):
        self.poisoned_samples[int(sample_id)] = (
            torch.as_tensor(poisoned_x, dtype=torch.float32).detach().cpu(),
            torch.as_tensor(poisoned_y, dtype=torch.float32).detach().cpu(),
        )

    def get_clean_sample(self, sample_id):
        sample_id = int(sample_id)
        start = self.sample_starts[sample_id]

        x_start = start
        x_end = start + self.seq_len
        y_start = x_end
        y_end = x_end + self.pred_len

        x = self.data[x_start:x_end].clone()
        y = self.data[y_start:y_end].clone()

        if self.use_timestamps:
            x_mark = self.time_marks[x_start:x_end].clone()
            y_mark = self.time_marks[y_start:y_end].clone()
            return x, y, x_mark, y_mark, start

        return x, y, None, None, start

    def __getitem__(self, sample_id):
        clean_x, clean_y, x_mark, y_mark, global_start = self.get_clean_sample(sample_id)

        if int(sample_id) in self.poisoned_samples:
            x, y = self.poisoned_samples[int(sample_id)]
            x = x.clone()
            y = y.clone()
        else:
            x = clean_x
            y = clean_y

        if self.use_timestamps:
            return x, y, clean_y, int(sample_id), x_mark, y_mark

        return x, y, clean_y, int(sample_id)


class AttackEvaluateSet(Dataset):
    """
    测试阶段动态注入 trigger。

    约定:
        trigger 放在输入窗口末尾
        target 放在预测窗口开头

    注意:
        这里不修改底层 test dataset。
    """

    def __init__(self, base_dataset, attacker):
        super().__init__()
        self.base_dataset = base_dataset
        self.attacker = attacker

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        return self.base_dataset[idx]

    def collate_fn(self, batch):
        has_mark = len(batch[0]) == 6

        if has_mark:
            xs, ys, clean_ys, sample_ids, x_marks, y_marks = zip(*batch)
            x_marks = torch.stack(x_marks, dim=0)
            y_marks = torch.stack(y_marks, dim=0)
        else:
            xs, ys, clean_ys, sample_ids = zip(*batch)

        xs = torch.stack(xs, dim=0)
        clean_ys = torch.stack(clean_ys, dim=0)
        ys = clean_ys.clone()
        sample_ids = torch.as_tensor(sample_ids, dtype=torch.long)

        xs, ys = self.attacker.build_poisoned_batch(xs, ys)

        if has_mark:
            return xs, ys, clean_ys, sample_ids, x_marks, y_marks

        return xs, ys, clean_ys, sample_ids
