import torch
from torch import nn
from typing import Optional, Tuple, List
import math
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel

class KVCache():
    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
    
    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            # The shape of the key_cache is [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            return self.key_cache[0].shape[-2]

    def update(self, key: torch.Tensor, value: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            # If we never added anything to the KV-Cache of this layer, let's create it.
            self.key_cache.append(key)
            self.value_cache.append(value)
        else:
            # ... otherwise we concatenate the new keys with the existing ones.
            # each tensor has shape: [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value], dim=-2)
        # ... and then we return all the existing keys + the new ones.
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

class GemmaConfig():
    def __init__(
            self,
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim=256,
            max_position_embeddings=8192,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            attention_bias=False,
            attention_dropout=0.0,
            pad_token_id=None,
            **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id

class PaliGemmaConfig():
    def __init__(self, vision_config=None, text_config=None, ignore_index=-100, image_token_index=256000, vocab_size=257152, projection_dim=2048, hidden_size=2048, pad_token_id=None, **kwargs):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config
        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
        
    def norm_(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self.norm_(x.float())
        # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim  # set to the head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # calculate the theta according to the formula theta_i = base ^ (2i/dim) where i = 0, 1, ..., dim//2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [batch_size, num_attention_heads, seq_len, head_size]
        # self.inv_freq.to(x.device)

        # copy the inv_freq tensor for batch in the sequence
        # [dim//2] -> [1, dim//2, 1] -> [batch_size, head_dim // 2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        
        # [batch_size, seq_len] -> [batch_size, 1, seq_len]
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # multiply each theta by the position (which is the argument of sin and cos functions)
            # freqs: [batch_size, head_dim // 2, 1] @ [batch_size, 1, seq_len] -> [batch_size, seq_len, head_dim//2]
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # emb: [batch_size, seq_len, head_dim]
            emb = torch.cat((freqs, freqs), dim=-1)
            # cos, sin [batch_size, seq_len, head_dim]
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    
def rotate_half(x):
    # Build the [-x2, x1, -x4, x3, ...] tensor for the sin part of the positional encoding.
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim) # Add the head dimension
    sin = sin.unsqueeze(unsqueeze_dim) # Add the head dimension
    # Apply the formula (34) of the Rotary Positional Encoding paper.
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class GemmaMLP(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
    def forward(self, x):
        # 这是 Gemma / LLaMA 系列模型中 SwiGLU 激活函数的等效实现
        # Equivalent to:
        # y = self.gate_proj(x) # [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
        # y = torch.gelu(y, approximate="tanh") # [Batch_Size, Seq_Len, Intermediate_Size]
        # j = self.up_proj(x) # [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
        # z = y * j # [Batch_Size, Seq_Len, Intermediate_Size]
        # z = self.down_proj(z) # [Batch_Size, Seq_Len, Intermediate_Size] -> [Batch_Size, Seq_Len, Hidden_Size]
        return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        assert self.hidden_size % self.num_heads == 0
        
        # number of heads = 8
        # hidden_size = 1024
        # head_dim = 1024 / 8 = 128
        # wq: [1024, 8 * 128] = [1024, 1024]
        # wk: [1024, 1 * 128] = [1024, 128]
        # wv: [1024, 1 * 128] = [1024, 128]
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = GemmaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.rope_theta)

    def forward(self,
                hidden_state: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                kv_cache: Optional[KVCache] = None,
                **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_state.size()
        # [batch_size, seq_len, hidden_size] = [batch_size, seq_len, num_heads_Q * head_dim]
        q = self.q_proj(hidden_state)
        # [batch_size, seq_len, num_heads_KV * head_dim]
        k = self.k_proj(hidden_state)
        # [batch_size, seq_len, num_heads_KV * head_dim]
        v = self.v_proj(hidden_state)

        # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, num_heads_Q, head_dim] -> [batch_size, num_heads_Q, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, num_heads_KV, head_dim] -> [batch_size, num_heads_KV, seq_len, head_dim]
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # [batch_size, seq_len, head_dim], [batch_size, seq_len, head_dim]
        cos, sin = self.rotary_emb(v, position_ids, seq_len=None)
        # [batch_size, num_heads_Q, seq_len, head_dim], [batch_size, num_heads_KV, seq_len, head_dim]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v, self.layer_idx)

        # repeat the key and values to match the number of heads of the query
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        # [batch_size, num_heads_Q, seq_len, head_dim] * [batch_size, num_heads_Q, head_dim, seq_len] -> [batch_size, num_heads_Q, seq_len, seq_len]
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        assert attention_mask is not None
        attn_weights = attn_weights + attention_mask

        # Apply the softmax
        # [Batch_Size, Num_Heads_Q, Seq_Len_Q, Seq_Len_KV]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # [batch_size, num_heads_Q, seq_len, seq_len] * [batch_size, num_heads_Q, seq_len, head_dim] -> [batch_size, num_heads_Q, seq_len, head_dim]
        attn_output = torch.matmul(attn_weights, v)
        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        # [batch_size, num_heads_Q, seq_len, head_dim] -> [batch_size, seq_len, num_heads_Q, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [batch_size, seq_len, num_heads_Q, head_dim] -> [batch_size, seq_len, num_heads_Q * head_dim]
        attn_output = attn_output.view(batch_size, seq_len, -1)
        # [batch_size, seq_len, num_heads_Q * head_dim] -> [batch_size, seq_len, hidden_size]
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
        


class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GemmaAttention(config, layer_idx)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(self, 
                hidden_state: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                kv_cache: Optional[KVCache] = None,):
        residual = hidden_state
        # [batch_size, seq_len, hidden_size]
        hidden_state = self.input_layernorm(hidden_state)
        # [batch_size, seq_len, hidden_size]
        hidden_state, _ = self.self_attn(hidden_state, attention_mask, position_ids, kv_cache)
        # [batch_size, seq_len, hidden_size]
        hidden_state = residual + hidden_state

        # [batch_size, seq_len, hidden_size]
        residual = hidden_state
        # [batch_size, seq_len, hidden_size]
        hidden_state = self.post_attention_layernorm(hidden_state)
        # [batch_size, seq_len, hidden_size]
        hidden_state = self.mlp(hidden_state)
        # [batch_size, seq_len, hidden_size]
        hidden_state = residual + hidden_state
        
        return hidden_state
    

class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens
    
    def forward(self, 
                attention_mask: Optional[torch.Tensor] = None, 
                position_ids: Optional[torch.LongTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None, 
                kv_cache: Optional[KVCache] = None,) -> torch.FloatTensor:
        # [batch_size, seq_len, hidden_size]
        hidden_states = inputs_embeds
        # [batch_size, seq_len, hidden_size]
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        for layer in self.layers:
            # [batch_size, seq_len, hidden_size]
            hidden_states = layer(hidden_states, attention_mask, position_ids, kv_cache)

        # [batch_size, seq_len, hidden_size]
        hidden_states = self.norm(hidden_states)
        return hidden_states


class GemmaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, 
                attention_mask: Optional[torch.Tensor] = None, 
                position_ids: Optional[torch.LongTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None, 
                kv_cache: Optional[KVCache] = None,) -> Tuple:
        # input_embeds: [Batch_Size, Seq_Len, Hidden_Size]
        # outputs: [Batch_Size, Seq_Len, Hidden_Size]
        outputs = self.model(attention_mask, position_ids, inputs_embeds, kv_cache)
        hidden_states = outputs
        logits = self.lm_head(hidden_states).float()

        return_data = {"logits": logits}
        if kv_cache is not None:
            # return the updated cache
            return_data["kv_cache"] = kv_cache
        return return_data

class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, image_features):
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, projection_dim]
        hidden_states = self.linear(image_features)
        return hidden_states

class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        return self.language_model.tie_weights()
    
    def _merge_ids_with_image_features(self, image_features: torch.Tensor, inputs_embeds: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, kv_cache: Optional[KVCache] = None):
        _, _, embed_dim = image_features.shape
        batch_size, seq_len = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        # image_features.shape: [batch_size, num_image_tokens, embed_dim]
        scaled_image_features = image_features / math.sqrt(self.config.hidden_size)
        
        # combine the embeddings of image tokens, the text tokens and mask out all the padding tokens
        final_embedding = torch.zeros(batch_size, seq_len, embed_dim, dtype=dtype, device=device)
        
        # shape: [batch_size, seq_len]. true for text tokens
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        # shape: [batch_size, seq_len]. true for image tokens
        image_mask = (input_ids == self.config.image_token_index)
        # shape: [batch_size, seq_len]. true for padding tokens
        pad_mask = (input_ids == self.pad_token_id)

        # we need to expand the masks to the embedding dimension otherwise we cannot use them in torch.where
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # add the text embedding
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)
        # insert image embeddings. we cannot use torch.where because the sequence length of scaled_image_features is not equal to sequence length of final embeddngs
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)
        # zero out padding tokens
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        ## create the attention mask ##

        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        # min_dtype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0:
            # do not mask any token, because we are in the prefill phase
            # this only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            # since we are generating tokens, the query must be one single token
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            # also in this case we do not need to mask anything , since each query should be able to attend all previous
            # this only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        # add head dimension
        # [batch_size, q_len, kv_len] -> [batch_size, num_heads_q, q_len, kv_len]
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # the position of the query is just last position
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # create a position_ids based on the size of the attention_mask
            # for masked tokens, use number 1 as position.
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)
        return final_embedding, causal_mask, position_ids
    

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        assert torch.all(attention_mask == 1), "The input cannot be padded"
        # 1. extra the input embeddings
        # shape: (batch_size, seq_len, hidden_size)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 2. merge text and images
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Seq_Len, hidden_size]
        image_features = self.multi_modal_projector(selected_image_feature)

        # 3. merge the embeddings of text tokens and image tokens
        inputs_embeds, attention_mask, position_ids = self._merge_ids_with_image_features(image_features, inputs_embeds, input_ids, attention_mask, kv_cache)

        outputs = self.language_model(attention_mask=attention_mask, position_ids=position_ids, inputs_embeds=inputs_embeds, kv_cache=kv_cache)
        return outputs