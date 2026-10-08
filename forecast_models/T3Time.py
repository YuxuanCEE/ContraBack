from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


class Normalize(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):
        if mode == 'norm':
            dims = tuple(range(1, x.ndim - 1))
            self.mean = torch.mean(x, dim=dims, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=dims, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        if mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
            return x * self.stdev + self.mean
        raise NotImplementedError


class CrossModalLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, q, k, v):
        q_proj = self.q_proj(q)
        k_proj = self.k_proj(k)
        v_proj = self.v_proj(v)
        scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / sqrt(q_proj.shape[-1])
        attn = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(self.dropout(attn), v_proj)
        q = self.norm_attn(q + self.dropout(self.out_proj(attn_out)))
        return self.norm_ffn(q + self.dropout(self.ffn(q)))


class AdaptiveDynamicHeadsCMA(nn.Module):
    def __init__(self, num_heads, channel):
        super().__init__()
        self.num_heads = num_heads
        self.channel = channel
        self.gate_mlp = nn.Sequential(
            nn.Linear(num_heads * channel, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_heads),
        )

    def forward(self, cma_outputs):
        B, C, N = cma_outputs[0].shape
        H = self.num_heads
        combined = torch.cat(cma_outputs, dim=1)
        gates = torch.softmax(self.gate_mlp(combined.permute(0, 2, 1)), dim=-1)
        combined_heads = combined.view(B, H, C, N).permute(0, 1, 3, 2)
        return (combined_heads * gates.permute(0, 2, 1).unsqueeze(-1)).sum(dim=1).permute(0, 2, 1)


class RichHorizonGate(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.Sigmoid(),
        )

    def forward(self, embedding, horizon):
        B, C, _ = embedding.size()
        pooled = embedding.mean(dim=2)
        horizon_tensor = torch.full((B, 1), float(horizon) / 1000.0, device=embedding.device)
        return self.gate_mlp(torch.cat([pooled, horizon_tensor], dim=1)).unsqueeze(-1)


class FrequencyAttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.freq_attention = nn.Sequential(
            nn.Linear(embed_dim, max(1, embed_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(1, embed_dim // 2), 1),
        )

    def forward(self, freq_enc_out):
        attn = torch.softmax(self.freq_attention(freq_enc_out), dim=1)
        return (freq_enc_out * attn).sum(dim=1)


class PromptEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_name = str(getattr(config, 'dataset_name', 'custom'))
        self.d_llm = int(config.d_llm)
        self.max_length = int(getattr(config, 'prompt_max_length', 512))
        self.value_round = int(getattr(config, 'prompt_value_round', 4))
        self.content = getattr(config, 'content', 'A custom multivariate time series forecasting dataset.')
        self._init_gpt2(config)

    def _init_gpt2(self, config):
        import transformers
        from transformers import GPT2Model, GPT2Tokenizer

        transformers.logging.set_verbosity_error()
        model_path = getattr(config, 'llm_model_path', './pretrained_models/gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = GPT2Model.from_pretrained(
            model_path,
            local_files_only=True,
            use_safetensors=True,
        )
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.pad_token = '[PAD]'
        for param in self.model.parameters():
            param.requires_grad = False

    def _interval_text(self):
        name = self.data_name.lower()
        if name in {'etth1', 'etth2', 'electricity', 'traffic'}:
            return 'every hour'
        if name in {'ettm1', 'ettm2'}:
            return 'every 15 minutes'
        if name == 'weather':
            return 'every 10 minutes'
        if name in {'exchange_rate', 'stock_data'}:
            return 'every day'
        return 'at each time step'

    def _build_prompts(self, x_enc):
        B, L, N = x_enc.shape
        interval = self._interval_text()
        prompts = []
        for b in range(B):
            for n in range(N):
                values = x_enc[b, :, n].detach().cpu().tolist()
                values = [round(float(v), self.value_round) for v in values]
                values_str = ', '.join(str(v) for v in values)
                trend = float(torch.diff(x_enc[b, :, n]).sum().detach().cpu())
                prompts.append(
                    f'Dataset description: {self.content} '
                    f'From step 1 to step {L}, the values were {values_str} {interval}. '
                    f'The total trend value was {trend:.6f}.'
                )
        return prompts

    @torch.no_grad()
    def generate_embeddings(self, x_enc):
        B, _, N = x_enc.shape
        prompts = self._build_prompts(x_enc)
        inputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(x_enc.device)
        outputs = self.model(**inputs).last_hidden_state
        lengths = inputs.attention_mask.sum(dim=1) - 1
        last_token = outputs[torch.arange(outputs.shape[0], device=x_enc.device), lengths]
        return last_token.reshape(B, N, self.d_llm).permute(0, 2, 1)


class Model(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self._ensure_config_defaults()
        self.task_name = config.task_name
        self.channel = int(config.channel)
        self.num_nodes = int(config.num_nodes)
        self.seq_len = int(config.seq_len)
        self.pred_len = int(config.pred_len)
        self.dropout_n = float(config.dropout_n)
        self.d_llm = int(config.d_llm)
        self.e_layer = int(config.e_layer)
        self.d_layer = int(config.d_layer)
        self.d_ff = int(config.d_ff)
        self.head = int(config.head)
        self.num_cma_heads = int(config.num_cma_heads)

        self.normalize_layers = Normalize(self.num_nodes, affine=False)
        self.length_to_feature = nn.Linear(self.seq_len, self.channel)
        self.ts_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.channel,
                nhead=self.head,
                batch_first=True,
                norm_first=True,
                dropout=self.dropout_n,
            ),
            num_layers=self.e_layer,
        )
        self.freq_token_proj = nn.Linear(1, self.channel)
        self.freq_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.channel,
                nhead=self.head,
                batch_first=True,
                norm_first=True,
                dropout=self.dropout_n,
            ),
            num_layers=1,
        )
        self.freq_pool = FrequencyAttentionPooling(self.channel)
        self.rich_horizon_gate = RichHorizonGate(self.channel)
        self.prompt_embedder = PromptEmbedding(config)
        self.prompt_proj = nn.Linear(self.d_llm, self.num_nodes)
        self.cma_heads = nn.ModuleList([
            CrossModalLayer(self.num_nodes, self.d_ff, self.dropout_n)
            for _ in range(self.num_cma_heads)
        ])
        self.adaptive_dynamic_heads_cma = AdaptiveDynamicHeadsCMA(self.num_cma_heads, self.channel)
        self.residual_alpha = nn.Parameter(torch.ones(self.channel) * 0.5)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.channel,
                nhead=self.head,
                batch_first=True,
                norm_first=True,
                dropout=self.dropout_n,
            ),
            num_layers=self.d_layer,
        )
        self.c_to_length = nn.Linear(self.channel, self.pred_len)

    def _ensure_config_defaults(self):
        defaults = {
            'task_name': 'long_term_forecast',
            'channel': 32,
            'num_nodes': getattr(self.config, 'enc_in', 1),
            'd_llm': 768,
            'e_layer': 1,
            'd_layer': 1,
            'd_ff': 32,
            'head': 8,
            'dropout_n': 0.2,
            'num_cma_heads': 4,
            'llm_model_path': './pretrained_models/gpt2',
            'prompt_max_length': 512,
            'prompt_value_round': 4,
            'dataset_name': 'custom',
            'content': 'A custom multivariate time series forecasting dataset.',
        }
        for key, value in defaults.items():
            if not hasattr(self.config, key):
                setattr(self.config, key, value)
        self.config.head = self._valid_head_count(int(self.config.channel), int(self.config.head))

    @staticmethod
    def _valid_head_count(dim, heads):
        heads = max(1, min(heads, dim))
        while dim % heads != 0 and heads > 1:
            heads -= 1
        return heads

    def param_num(self):
        return sum(param.nelement() for param in self.parameters())

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def frequency_domain_processing(self, input_data):
        freq_mag = torch.fft.rfft(input_data, dim=-1).abs()
        B, N, Lf = freq_mag.shape
        freq_tokens = freq_mag.unsqueeze(-1).reshape(B * N, Lf, 1)
        freq_tokens = self.freq_token_proj(freq_tokens)
        freq_enc_out = self.freq_encoder(freq_tokens)
        freq_enc_out = self.freq_pool(freq_enc_out)
        return freq_enc_out.reshape(B, N, self.channel)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name not in {'long_term_forecast', 'short_term_forecast'}:
            return None

        x_enc = x_enc.float()
        embeddings = self.prompt_embedder.generate_embeddings(x_enc)
        x_enc = self.normalize_layers(x_enc, 'norm')
        x = x_enc.permute(0, 2, 1)

        freq_enc_out = self.frequency_domain_processing(x)
        x = self.length_to_feature(x)
        enc_out = self.ts_encoder(x).permute(0, 2, 1)

        gate = self.rich_horizon_gate(enc_out, self.pred_len)
        enc_out = gate * freq_enc_out.permute(0, 2, 1) + (1 - gate) * enc_out

        embeddings = self.prompt_proj(embeddings.permute(0, 2, 1)).permute(0, 2, 1)
        cma_outputs = [head(enc_out, embeddings, embeddings) for head in self.cma_heads]
        fused = self.adaptive_dynamic_heads_cma(cma_outputs)

        alpha = self.residual_alpha.view(1, self.channel, 1)
        cross_out = alpha * fused + (1 - alpha) * enc_out
        cross_out = cross_out.permute(0, 2, 1)
        dec_out = self.decoder(cross_out, cross_out)
        dec_out = self.c_to_length(dec_out).permute(0, 2, 1)
        return self.normalize_layers(dec_out, 'denorm')
