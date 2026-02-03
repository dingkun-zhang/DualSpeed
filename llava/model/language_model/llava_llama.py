#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

import torch.nn.functional as F
import random
import wandb
from llava.constants import IGNORE_INDEX


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    # [DualSpeed]
    def initialize_mode_isolator(self):
        self.mode_isolator = nn.Linear(self.model.config.hidden_size, 4, bias=False)

    # [DualSpeed]
    def is_pretraining_stage(self):
        if not hasattr(self, 'is_pretraining_stage_'):
            self.is_pretraining_stage_ = self.training and not any(
                param.requires_grad
                for layer in self.model.layers
                for param in layer.parameters()
            )
        return self.is_pretraining_stage_

    def get_model(self):
        return self.model

    # [DualSpeed]
    def distillation_loss(self, student_logits, teacher_logits, T=1.0):
        return F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction="batchmean"
        ) * (T ** 2)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # [DualSpeed] Core Framework
        slow_mode_ratio = 0.1
        pruning_ratio = 0.9
        random_value = random.random()
        if self.training and not self.is_pretraining_stage() and random_value < slow_mode_ratio:
            # [DualSpeed] Slow-Mode

            input_ids_ = input_ids
            position_ids_ = position_ids
            attention_mask_ = attention_mask
            past_key_values_ = past_key_values
            labels_ = labels
            inputs_embeds_ = inputs_embeds

            # [DualSpeed] Teacher forward
            with torch.no_grad():
                if inputs_embeds_ is None:
                    (
                        input_ids,
                        position_ids,
                        attention_mask,
                        past_key_values,
                        inputs_embeds,
                        labels
                    ) = self.prepare_inputs_labels_for_multimodal(
                        input_ids_.clone(),
                        position_ids_.clone() if position_ids_ is not None else None,
                        attention_mask_.clone() if attention_mask_ is not None else None,
                        past_key_values_.clone() if past_key_values_ is not None else None,
                        labels_.clone(),
                        images,
                        image_sizes,
                        prune=True,
                        pruning_ratio=pruning_ratio
                    )
                outputs_teacher = super().forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=True,
                    return_dict=True
                )
                labels_batch = labels
                logits_teacher_batch = []
                for i, labels in enumerate(labels_batch):
                    LABEL_token_indexes = [i-1 for i, e in enumerate(labels) if e != IGNORE_INDEX and i-1 >= 0]
                    logits_teacher = outputs_teacher['logits'][i][LABEL_token_indexes]
                    logits_teacher_batch.append(logits_teacher.detach())
                # teacher ce loss only for monitoring
                teacher_ce_loss = outputs_teacher["loss"] if isinstance(outputs_teacher, dict) else outputs_teacher[0]

            # [DualSpeed] Student forward
            if inputs_embeds_ is None:
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids_.clone(),
                    position_ids_.clone() if position_ids_ is not None else None,
                    attention_mask_.clone() if attention_mask_ is not None else None,
                    past_key_values_.clone() if past_key_values_ is not None else None,
                    labels_.clone(),
                    images,
                    image_sizes,
                    prune=False
                )
            outputs_student = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=True
            )
            labels_batch = labels
            logits_student_batch = []
            for i, labels in enumerate(labels_batch):
                LABEL_token_indexes = [i-1 for i, e in enumerate(labels) if e != IGNORE_INDEX and i-1 >= 0]
                logits_student = outputs_student['logits'][i][LABEL_token_indexes]
                logits_student_batch.append(logits_student)
            student_ce_loss = outputs_student["loss"] if isinstance(outputs_student, dict) else outputs_student[0]

            # [DualSpeed] Calculate self-distillation loss
            T, alpha = 1, 1
            logits_s_all, logits_t_all = [], []
            for logits_s, logits_t in zip(logits_student_batch, logits_teacher_batch):
                min_len = min(len(logits_s), len(logits_t))
                logits_s = logits_s[:min_len]
                logits_t = logits_t[:min_len]
                logits_s_all.append(logits_s)
                logits_t_all.append(logits_t)
            logits_s_all = torch.cat(logits_s_all, dim=0)
            logits_t_all = torch.cat(logits_t_all, dim=0)
            distill_loss = self.distillation_loss(logits_s_all, logits_t_all, T=T)
            total_loss = student_ce_loss + alpha * distill_loss
        
            wandb.log({
                "teacher_ce_loss": teacher_ce_loss.item(),
                "student_ce_loss": student_ce_loss.item(),
                "distill_loss": distill_loss.item()
            })

            if isinstance(outputs_student, dict):
                outputs_student["loss"] = total_loss
            else:
                outputs_student[0] = total_loss

            return outputs_student
            
        # [DualSpeed] Fast-Mode
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                prune=self.training,
                pruning_ratio=pruning_ratio
            )
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        if self.training:
            ce_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            wandb.log({
                "teacher_ce_loss": ce_loss.item(),
                "student_ce_loss": float('nan'),
                "distill_loss": float('nan')
            })
        return outputs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
