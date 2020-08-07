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

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from model.configure_lol import LOLConfig

from transformers.activations import gelu, gelu_fast, gelu_new, swish
from transformers.configuration_reformer import ReformerConfig
from transformers.file_utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_callable,
)
from transformers.modeling_outputs import CausalLMOutput, MaskedLMOutput, QuestionAnsweringModelOutput, SequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel, apply_chunking_to_forward


