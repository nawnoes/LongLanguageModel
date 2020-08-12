# coding=utf-8
# Copyright 2020 SeongHwan Kim Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LOL model. """

####################################################
# In this template, replace all the XXX (various casings) with your model name
####################################################


import logging
import os
import sys
from collections import namedtuple
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import List, Optional, Tuple



import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from model.configure_lol import LOLConfig

from transformers.activations import gelu, gelu_fast, gelu_new, swish, get_activation
from transformers.configuration_electra import ElectraConfig
from transformers.configuration_reformer import ReformerConfig
from transformers.file_utils import (
    ModelOutput,
    DUMMY_INPUTS,
    DUMMY_MASK,
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_callable,
    replace_return_docstrings,

)
from transformers.modeling_outputs import (
    BaseModelOutput,
    CausalLMOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel, apply_chunking_to_forward
from transformers.modeling_bert import BertEmbeddings, BertEncoder, BertLayerNorm, BertPreTrainedModel

logger = logging.getLogger(__name__)

_CONFIG_FOR_DOC = "ElectraConfig"
_TOKENIZER_FOR_DOC = "ElectraTokenizer"
"""
reformer 사용 함
"""
def _stable_argsort(vector, dim):
    # this function scales the vector so that torch.argsort is stable.
    # torch.argsort is not stable on its own
    scale_offset = torch.arange(vector.shape[dim], device=vector.device).view(1, 1, -1)
    scale_offset = scale_offset.expand(vector.shape)
    scaled_vector = vector.shape[dim] * vector + (scale_offset % vector.shape[dim])
    return torch.argsort(scaled_vector, dim=dim)


def _get_least_common_mult_chunk_len(config):
    attn_types = config.attn_layers
    attn_types_set = set(attn_types)
    if len(attn_types_set) == 1 and attn_types[0] == "lsh":
        return config.lsh_attn_chunk_length
    elif len(attn_types_set) == 1 and attn_types[0] == "local":
        return config.local_attn_chunk_length
    elif len(attn_types_set) == 2 and attn_types_set == set(["lsh", "local"]):
        return np.lcm(config.lsh_attn_chunk_length, config.local_attn_chunk_length)
    else:
        raise NotImplementedError(
            "Only attn layer types 'lsh' and 'local' exist, but `config.attn_layers`: {}. Select attn layer types from ['lsh', 'local'] only.".format(
                config.attn_layers
            )
        )


def _get_min_chunk_len(config):
    attn_types = config.attn_layers
    attn_types_set = set(attn_types)
    if len(attn_types_set) == 1 and attn_types[0] == "lsh":
        return config.lsh_attn_chunk_length
    elif len(attn_types_set) == 1 and attn_types[0] == "local":
        return config.local_attn_chunk_length
    elif len(attn_types_set) == 2 and attn_types_set == set(["lsh", "local"]):
        return min(config.lsh_attn_chunk_length, config.local_attn_chunk_length)
    else:
        raise NotImplementedError(
            "Only attn layer types 'lsh' and 'local' exist, but `config.attn_layers`: {}. Select attn layer types from ['lsh', 'local'] only.".format(
                config.attn_layers
            )
        )
"""
reformer AxialPositionEmbedding
"""
class AxialPositionEmbeddings(nn.Module):
    """Constructs axial position embeddings. Useful for very long input
    sequences to save memory and time.
    """

    def __init__(self, config):
        super().__init__()
        self.axial_pos_shape = config.axial_pos_shape
        self.axial_pos_embds_dim = config.axial_pos_embds_dim
        self.dropout = config.hidden_dropout_prob

        self.least_common_mult_chunk_length = _get_least_common_mult_chunk_len(config)
        self.weights = nn.ParameterList()

        assert (
            sum(self.axial_pos_embds_dim) == config.hidden_size
        ), "Make sure that config.axial_pos_embds factors: {} sum to config.hidden_size: {}".format(
            self.axial_pos_embds_dim, config.hidden_size
        )

        # create weights
        for axis, axial_pos_embd_dim in enumerate(self.axial_pos_embds_dim):
            # create expanded shapes
            ax_shape = [1] * len(self.axial_pos_shape)
            ax_shape[axis] = self.axial_pos_shape[axis]
            ax_shape = tuple(ax_shape) + (axial_pos_embd_dim,)

            # create tensor and init
            self.weights.append(nn.Parameter(torch.ones(ax_shape, dtype=torch.float32)))

    def forward(self, position_ids):
        # broadcast weights to correct shape
        batch_size = position_ids.shape[0]
        sequence_length = position_ids.shape[1]

        broadcasted_weights = [
            weight.expand((batch_size,) + self.axial_pos_shape + weight.shape[-1:]) for weight in self.weights
        ]

        if self.training is True:
            assert (
                reduce(mul, self.axial_pos_shape) == sequence_length
            ), "If training, make sure that config.axial_pos_shape factors: {} multiply to sequence length. Got prod({}) != sequence_length: {}. You might want to consider padding your sequence length to {} or changing config.axial_pos_shape.".format(
                self.axial_pos_shape, self.axial_pos_shape, sequence_length, reduce(mul, self.axial_pos_shape)
            )
            if self.dropout > 0:
                weights = torch.cat(broadcasted_weights, dim=-1)
                # permute weights so that 2D correctly drops dims 1 and 2
                transposed_weights = weights.transpose(2, 1)
                # drop entire matrix of last two dims (prev dims 1 and 2)
                dropped_transposed_weights = nn.functional.dropout2d(
                    transposed_weights, p=self.dropout, training=self.training
                )
                dropped_weights = dropped_transposed_weights.transpose(2, 1)

                position_encodings = torch.reshape(dropped_weights, (batch_size, sequence_length, -1))

            else:
                position_encodings = torch.cat(
                    [torch.reshape(weight, (batch_size, sequence_length, -1)) for weight in broadcasted_weights],
                    dim=-1,
                )

        else:
            assert (
                reduce(mul, self.axial_pos_shape) >= sequence_length
            ), "Make sure that config.axial_pos_shape factors: {} multiply at least to max(sequence_length, least_common_mult_chunk_length): max({}, {})".format(
                self.axial_pos_shape, sequence_length, self.least_common_mult_chunk_length,
            )

            # compute how many columns are needed
            max_position_id = position_ids.max().item()
            required_pos_encodings_columns = -(-(max_position_id + 1) // self.axial_pos_shape[1])

            # cut to columns that are needed
            position_encodings = torch.cat(
                [weight[:, :required_pos_encodings_columns] for weight in broadcasted_weights], dim=-1
            )
            position_encodings = torch.reshape(position_encodings, (batch_size, -1, position_encodings.shape[-1]))

            # select correct position encodings
            position_encodings = torch.cat(
                [
                    torch.index_select(position_encodings[i], 0, position_ids[i]).unsqueeze(0)
                    for i in range(batch_size)
                ],
                dim=0,
            )

        return position_encodings

class PositionEmbeddings(nn.Module):
    """Constructs conventional position embeddings of shape `[max_pos_embeddings, hidden_size]`.
    """

    def __init__(self, config):
        super().__init__()
        self.dropout = config.hidden_dropout_prob
        self.embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)

    def forward(self, position_ids):
        position_embeddings = self.embedding(position_ids)
        position_embeddings = nn.functional.dropout(position_embeddings, p=self.dropout, training=self.training)
        return position_embeddings

class LOLEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.max_position_embeddings = config.max_position_embeddings
        self.dropout = config.hidden_dropout_prob

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = (
            AxialPositionEmbeddings(config) if config.axial_pos_embds else PositionEmbeddings(config)
        )

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None, start_idx_pos_encodings=0):
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        else:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device

        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = torch.arange(
                start_idx_pos_encodings, start_idx_pos_encodings + seq_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        assert (
            position_ids.shape[-1] <= self.max_position_embeddings
        ), "Sequence Length: {} has to be larger equal than config.max_position_embeddings: {}".format(
            position_ids.shape[-1], self.max_position_embeddings
        )

        # dropout
        embeddings = nn.functional.dropout(inputs_embeds, p=self.dropout, training=self.training)

        # add positional embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = embeddings + position_embeddings
        return embeddings

class LOLDiscriminatorPredictions(nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_prediction = nn.Linear(config.hidden_size, 1)
        self.config = config

    def forward(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = get_activation(self.config.hidden_act)(hidden_states)
        logits = self.dense_prediction(hidden_states).squeeze()

        return logits


class LOLGeneratorPredictions(nn.Module):
    """Prediction module for the generator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()

        self.LayerNorm = BertLayerNorm(config.embedding_size)
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)

    def forward(self, generator_hidden_states):
        hidden_states = self.dense(generator_hidden_states)
        hidden_states = get_activation("gelu")(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


class LOLPreTrainedModel(BertPreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = LOLConfig
    # load_tf_weights = load_tf_weights_in_electra
    base_model_prefix = "lol"

@dataclass
class LOLForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.ElectraForPreTrainingModel`.

    Args:
        loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss of the ELECTRA objective.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`):
            Prediction scores of the head (scores for each token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class LOLModel(LOLPreTrainedModel):

    config_class = LOLConfig

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = LOLEmbeddings(config)

        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(config.embedding_size, config.hidden_size)

        self.encoder = BertEncoder(config)
        self.config = config
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # @add_start_docstrings_to_callable(ELECTRA_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="google/electra-small-discriminator",
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        hidden_states = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        if hasattr(self, "embeddings_project"):
            hidden_states = self.embeddings_project(hidden_states)

        hidden_states = self.encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return hidden_states

class LOLClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class LOLForPreTraining(LOLPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.electra = LOLModel(config)
        self.discriminator_predictions = LOLDiscriminatorPredictions(config)
        self.init_weights()

    # @add_start_docstrings_to_callable(ELECTRA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=LOLForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`, defaults to :obj:`None`):
            Labels for computing the ELECTRA loss. Input should be a sequence of tokens (see :obj:`input_ids` docstring)
            Indices should be in ``[0, 1]``.
            ``0`` indicates the token is an original token,
            ``1`` indicates the token was replaced.

    Returns:

    Examples::

        >>> from transformers import ElectraTokenizer, ElectraForPreTraining
        >>> import torch

        >>> tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
        >>> model = ElectraForPreTraining.from_pretrained('google/electra-small-discriminator')

        >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        >>> logits = model(input_ids).logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        discriminator_sequence_output = discriminator_hidden_states[0]

        logits = self.discriminator_predictions(discriminator_sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1, discriminator_sequence_output.shape[1]) == 1
                active_logits = logits.view(-1, discriminator_sequence_output.shape[1])[active_loss]
                active_labels = labels[active_loss]
                loss = loss_fct(active_logits, active_labels.float())
            else:
                loss = loss_fct(logits.view(-1, discriminator_sequence_output.shape[1]), labels.float())

        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return LOLForPreTrainingOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )

