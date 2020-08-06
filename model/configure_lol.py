import logging
from typing import Callable, Union
from transformers.configuration_utils import PretrainedConfig

logger = logging.getLogger(__name__)

LOL_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "lol-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/xxx-base-uncased-config.json",
    "lol-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/xxx-large-uncased-config.json",
}


class LOLConfig(PretrainedConfig):
  model_type = 'lol'

  def __init__(self,
    #Reformer Config
    attention_head_size = 64,
    attn_layers = ["local", "lsh", "local", "lsh", "local", "lsh"],
    axial_norm_std = 1.0,
    axial_pos_embds = True,
    axial_pos_shape = [64, 64],
    axial_pos_embds_dim = [64, 192],
    chunk_size_lm_head = 0,
    chunk_size_feed_forward = 0,
    eos_token_id = 2,
    feed_forward_size = 512,
    hash_seed = None,
    is_decoder = False,
    local_num_chunks_before = 1,
    local_num_chunks_after = 0,
    local_attention_probs_dropout_prob = 0.05,
    local_attn_chunk_length = 64,
    lsh_attn_chunk_length = 64,
    lsh_attention_probs_dropout_prob = 0.0,
    lsh_num_chunks_before = 1,
    lsh_num_chunks_after = 0,
    num_buckets = None,
    num_hashes = 1,

    # Electra Config
    embedding_size=128,
    num_hidden_layers=12,
    intermediate_size=1024,

    attention_probs_dropout_prob=0.1,
    type_vocab_size=2,
    summary_type="first",
    summary_use_proj=True,
    summary_activation="gelu",
    summary_last_dropout=0.1,

    # Common Config
    vocab_size=30522,
    hidden_size=256,
    max_position_embeddings=4096,
    pad_token_id=0,
    num_attention_heads=2,
    layer_norm_eps=1e-12,
    initializer_range=0.02,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    **kwargs
    ):
    super().__init__(pad_token_id=pad_token_id, eos_token_id=eos_token_id, is_decoder=is_decoder, **kwargs)
    #Common Config
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.layer_norm_eps = layer_norm_eps
    self.num_attention_heads = num_attention_heads

    # Reformer Config
    self.hash_seed = hash_seed
    self.attention_head_size = attention_head_size
    self.num_hashes = num_hashes
    self.num_hidden_layers = len(attn_layers)
    self.num_buckets = tuple(num_buckets) if isinstance(num_buckets, list) else num_buckets
    self.lsh_attn_chunk_length = lsh_attn_chunk_length
    self.local_attn_chunk_length = local_attn_chunk_length
    self.lsh_num_chunks_after = lsh_num_chunks_after
    self.lsh_num_chunks_before = lsh_num_chunks_before
    self.local_num_chunks_after = local_num_chunks_after
    self.local_num_chunks_before = local_num_chunks_before
    self.hidden_act = hidden_act
    self.feed_forward_size = feed_forward_size
    self.lsh_attention_probs_dropout_prob = lsh_attention_probs_dropout_prob
    self.local_attention_probs_dropout_prob = local_attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.initializer_range = initializer_range
    self.axial_pos_embds = axial_pos_embds
    self.axial_pos_shape = tuple(axial_pos_shape)
    self.axial_pos_embds_dim = tuple(axial_pos_embds_dim)
    self.axial_norm_std = axial_norm_std
    self.chunk_size_lm_head = chunk_size_lm_head
    self.chunk_size_feed_forward = chunk_size_feed_forward
    self.attn_layers = attn_layers

    # Electra Config
    self.embedding_size = embedding_size
    self.num_hidden_layers = num_hidden_layers
    self.intermediate_size = intermediate_size
    self.hidden_act = hidden_act
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range

    self.summary_type = summary_type
    self.summary_use_proj = summary_use_proj
    self.summary_activation = summary_activation
    self.summary_last_dropout = summary_last_dropout
