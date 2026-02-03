import torch 
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel


class GemmaConfig:
    def __init__(self, vocab_size, hidden_size, intermediate_size, num_hidden_layers, num_attention_heads, num_key_value_heads, 
                 head_dim=256, max_position_embeddings=8192, rms_nrom_eps=1e-6,
                 rope_theta=10000.0, attention_bias=False, attention_dropout=0.0, pad_token_id=None, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_nrom_eps = rms_nrom_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id
        self.is_encoder_decoder = False


class PaliGemmaConfig:
    def __init__(self, vision_config=None, text_config=None, ignore_index=-100, image_token_index=256000, vocab_size=257152, projection_dim=2048, hidden_size=2048, pad_token_id=None, **kwargs):
        super.__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id
        self.vision_config = vision_config
        self.is_encoder_decoder = False

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config
        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim


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
        padding_mask = (input_ids == self.pad_token_id)

        # we need to expand the masks to the embedding dimension otherwise we cannot use them in torch.where
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        padding_mask_expanded = padding_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # add the text embedding
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)
        # insert image embeddings. we cannot use torch.where because the sequence length of scaled_image_features is not equal to sequence length of final embeddngs
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)
        # zero out padding tokens
        final_embedding = torch.where(padding_mask_expanded, torch.zeros_like(final_embedding), final_embedding)
    
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
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 2. merge text and images
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        selected_image_feature = self.vision_tower(pixel_values.to(input_embeds.dtype))
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Seq_Len, hidden_size]
        image_features = self.multi_modal_projector(selected_image_feature)

        # 3. merge the embeddings of text tokens and image tokens
        inputs_embeds, attention_mask, position_ids = self._merge_ids_with_image_features(image_features, input_embeds, input_ids, attention_mask, kv_cache)

        outputs = self.language_model(attention_mask=attention_mask, position_ids=position_ids, inputs_embeds=inputs_embeds, kv_cache=kv_cache)
        return outputs