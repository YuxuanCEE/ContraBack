from math import sqrt

import torch
import torch.nn as nn

from .layers.Embed import PatchEmbedding


class Normalize(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):
        if mode == 'norm':
            self.mean = torch.mean(x, dim=tuple(range(1, x.ndim - 1)), keepdim=True).detach()
            self.stdev = torch.sqrt(
                torch.var(x, dim=tuple(range(1, x.ndim - 1)), keepdim=True, unbiased=False) + self.eps
            ).detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        if mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
            return x * self.stdev + self.mean
        raise NotImplementedError


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return self.dropout(x)


class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.configs = configs
        self._ensure_config_defaults()
        self.task_name = configs.task_name
        self.pred_len = int(configs.pred_len)
        self.seq_len = int(configs.seq_len)
        self.d_ff = int(configs.d_ff)
        self.top_k = int(configs.top_k)
        self.d_llm = int(configs.llm_dim)
        self.patch_len = int(configs.patch_len)
        self.stride = int(configs.stride)

        self._init_llm(configs)

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.pad_token = '[PAD]'

        for param in self.llm_model.parameters():
            param.requires_grad = False

        self.description = getattr(
            configs,
            'content',
            'A custom multivariate time series forecasting dataset.'
        )
        self.dropout = nn.Dropout(configs.dropout)
        self.patch_embedding = PatchEmbedding(
            configs.d_model,
            self.patch_len,
            self.stride,
            int(getattr(configs, 'padding', 0)),
            configs.dropout,
        )

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = int(configs.num_tokens)
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((self.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name in {'long_term_forecast', 'short_term_forecast'}:
            self.output_projection = FlattenHead(
                configs.enc_in,
                self.head_nf,
                self.pred_len,
                head_dropout=configs.dropout,
            )
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def _ensure_config_defaults(self):
        defaults = {
            'task_name': 'long_term_forecast',
            'd_model': 128,
            'd_ff': 768,
            'n_heads': 8,
            'dropout': 0.1,
            'patch_len': 16,
            'stride': 8,
            'padding': 0,
            'top_k': 5,
            'llm_model': 'GPT2',
            'llm_dim': 768,
            'llm_layers': 1,
            'llm_model_path': './pretrained_models/gpt2',
            'num_tokens': 1000,
            'content': 'A custom multivariate time series forecasting dataset.',
        }
        for key, value in defaults.items():
            if not hasattr(self.configs, key):
                setattr(self.configs, key, value)

    def _init_llm(self, configs):
        import transformers
        from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

        transformers.logging.set_verbosity_error()
        llm_model = str(configs.llm_model).upper()
        if llm_model != 'GPT2':
            raise ValueError("This integration currently supports llm_model='GPT2'.")

        model_path = getattr(configs, 'llm_model_path', './pretrained_models/gpt2')
        gpt2_config = GPT2Config.from_pretrained(model_path)
        gpt2_config.num_hidden_layers = int(configs.llm_layers)
        gpt2_config.output_attentions = True
        gpt2_config.output_hidden_states = True
        self.llm_model = GPT2Model.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            config=gpt2_config,
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in {'long_term_forecast', 'short_term_forecast'}:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_prompt = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_prompt, dim=1)[0]
        max_values = torch.max(x_prompt, dim=1)[0]
        medians = torch.median(x_prompt, dim=1).values
        lags = self.calculate_lags(x_prompt)
        trends = x_prompt.diff(dim=1).sum(dim=1)

        prompts = []
        for b in range(x_prompt.shape[0]):
            trend = 'upward' if trends[b] > 0 else 'downward'
            prompt = (
                f'<|start_prompt|>Dataset description: {self.description} '
                f'Task description: forecast the next {self.pred_len} steps given the previous {self.seq_len} steps information; '
                'Input statistics: '
                f'min value {min_values[b].tolist()[0]}, '
                f'max value {max_values[b].tolist()[0]}, '
                f'median value {medians[b].tolist()[0]}, '
                f'the trend of input is {trend}, '
                f'top {self.top_k} lags are : {lags[b].tolist()}<|<end_prompt>|>'
            )
            prompts.append(prompt)

        prompt_ids = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=int(getattr(self.configs, 'prompt_max_length', 2048)),
        ).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_ids.to(x_enc.device))

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        x_patch = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_patch)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        llm_input = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llm_input).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]
        dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        return self.normalize_layers(dec_out, 'denorm')

    def calculate_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        corr = torch.fft.irfft(q_fft * torch.conj(k_fft), dim=-1)
        mean_value = torch.mean(corr, dim=1)
        top_k = min(self.top_k, mean_value.shape[-1])
        _, lags = torch.topk(mean_value, top_k, dim=-1)
        return lags


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads
        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)
        out = self.reprogramming(target_embedding, source_embedding, value_embedding)
        return self.out_projection(out.reshape(B, L, -1))

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        scale = 1. / sqrt(target_embedding.shape[-1])
        scores = torch.einsum('blhe,she->bhls', target_embedding, source_embedding)
        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        return torch.einsum('bhls,she->blhe', attn, value_embedding)
