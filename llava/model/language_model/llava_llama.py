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
import sys

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from torch.nn import CrossEntropyLoss
import torch.distributed as dist
try:
    import wandb
except:
    pass

w1 = 1
w2 = 1
w3 = 1
w4 = 1
w5 = 1
w6 = 1

def build_2layer_mlp_vision_decoder(llm_hidden_dim, img_cls_dim):
    return nn.Sequential(
        nn.Linear(llm_hidden_dim, 2048),  # First layer reduces size from 4096 to 2048
        nn.GELU(),              # Activation for first layer
        nn.Linear(2048, img_cls_dim),  # Second layer reduces size from 2048 to 1024
        nn.GELU()               # Activation for second layer
    )

def build_2layer_mlp_txt_decoder(llm_hidden_dim, txt_cls_dim):
    return nn.Sequential(
        nn.Linear(llm_hidden_dim, 2048),  # First layer reduces size from 4096 to 2048
        nn.GELU(),              # Activation for first layer
        nn.Linear(2048, txt_cls_dim),  # Second layer reduces size from 2048 to 1024
        nn.GELU()               # Activation for second layer
    )

class BroadcastModel(nn.Module):
    def __init__(self, llm_hidden_dim, txt_cls_dim, img_cls_dim):
        super().__init__()

        # Create decoders once as module attributes
        self.text_decoder = build_2layer_mlp_txt_decoder(llm_hidden_dim, txt_cls_dim)
        self.img_decoder = build_2layer_mlp_vision_decoder(llm_hidden_dim, img_cls_dim)

    def compute_combined_loss(self, pred, target, temperature=0.5):
        # MSE Loss
        mse_loss = F.mse_loss(pred, target)
        
        # Cosine Loss (1 - cosine_similarity to make it a loss)
        cosine_loss = 1 - F.cosine_similarity(pred, target, dim=-1).mean()
        
        # KL Loss with temperature scaling
        pred_scaled = F.log_softmax(pred / temperature, dim=-1)
        target_scaled = F.softmax(target / temperature, dim=-1)
        kl_loss = F.kl_div(pred_scaled, target_scaled, reduction='batchmean') * (temperature ** 2)
        
        # Combine losses with weights
        #return mse_loss + 0.5 * cosine_loss + 0.1 * kl_loss
        return mse_loss

    def compute_broadcast_losses(self, avg_all_tokens_embeddings, avg_text_embeddings, avg_img_embeddings, 
                          txt_input_cls_embeds, img_input_cls_embeds):
        
        losses = {}
        
        # 1. All tokens -> Text CLS (using text decoder)
        if txt_input_cls_embeds is not None and not torch.all(img_input_cls_embeds == 0):
            proj_all_to_txt = self.text_decoder(avg_all_tokens_embeddings)
            losses['all_to_txt'] = self.compute_combined_loss(proj_all_to_txt, txt_input_cls_embeds)
        else:
            losses['all_to_txt'] = 0
            
        # 2. All tokens -> Image CLS (using image decoder)
        if img_input_cls_embeds is not None and not torch.all(img_input_cls_embeds == 0):
            proj_all_to_img = self.img_decoder(avg_all_tokens_embeddings)
            losses['all_to_img'] = self.compute_combined_loss(proj_all_to_img, img_input_cls_embeds)
        else:
            losses['all_to_img'] = 0
            
        # 3. Text tokens -> Text CLS (using text decoder)
        if txt_input_cls_embeds is not None and not torch.all(img_input_cls_embeds == 0):
            proj_txt_to_txt = self.text_decoder(avg_text_embeddings)
            losses['txt_to_txt'] = self.compute_combined_loss(proj_txt_to_txt, txt_input_cls_embeds)
        else:
            losses['txt_to_txt'] = 0
            
        # 4. Image tokens -> Image CLS (using image decoder)
        if img_input_cls_embeds is not None and not torch.all(img_input_cls_embeds == 0):
            proj_img_to_img = self.img_decoder(avg_img_embeddings)
            losses['img_to_img'] = self.compute_combined_loss(proj_img_to_img, img_input_cls_embeds)
        else:
            losses['img_to_img'] = 0
            
        # 5a. Text tokens -> Image CLS (using image decoder)
        if img_input_cls_embeds is not None and not torch.all(img_input_cls_embeds == 0):
            proj_txt_to_img = self.img_decoder(avg_text_embeddings)
            losses['txt_to_img'] = self.compute_combined_loss(proj_txt_to_img, img_input_cls_embeds)
        else:
            losses['txt_to_img'] = 0
            
        # 5b. Image tokens -> Text CLS (using text decoder)
        if txt_input_cls_embeds is not None and not torch.all(img_input_cls_embeds == 0):
            proj_img_to_txt = self.text_decoder(avg_img_embeddings)
            losses['img_to_txt'] = self.compute_combined_loss(proj_img_to_txt, txt_input_cls_embeds)
        else:
            losses['img_to_txt'] = 0
        
        # Combine all losses with weights
        total_broadcast_loss = (
            losses['all_to_txt'] * w1 +
            losses['all_to_img'] * w2 +
            losses['txt_to_txt'] * w3 +
            losses['img_to_img'] * w4 +
            losses['txt_to_img'] * w5 +
            losses['img_to_txt'] * w6     
        )
        return total_broadcast_loss, losses

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

    def get_model(self):
        return self.model
    
    def _forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        img_input_cls_embeds: Optional[torch.FloatTensor] = None,
        txt_input_cls_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        # Convert attention_mask to float for computation
        attention_mask = attention_mask.float()  # (batch_size, sequence_length)

        # Expand attention mask for broadcasting
        expanded_attention_mask = attention_mask.unsqueeze(-1)  # (batch_size, sequence_length, 1)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        layer_states = outputs[-1][1:]

        layer_states16 = layer_states[15]

        layer_states24 = layer_states[23]

        llm_hidden_dim = layer_states16.shape[-1]
        txt_cls_dim = txt_input_cls_embeds.shape[-1]
        img_cls_dim = img_input_cls_embeds.shape[-1]

        # Apply mask to layer_states12
        masked_layer_states16 = layer_states16 * expanded_attention_mask  # (batch_size, sequence_length, embedding_dim)
        masked_layer_states24 = layer_states24 * expanded_attention_mask

#FOR PRETRAINING


        # Extract image tokens (index 1 to 576 inclusive)
        #img_tokens16 = masked_layer_states16[:, 1:577, :]  # Image tokens (1–576)
        # Extract text tokens (index 577 to the end)
        #text_tokens16 = masked_layer_states16[:, 577:, :]  # Text tokens (577 onward)

        # Extract image tokens (index 1 to 576 inclusive)
        #img_tokens24 = masked_layer_states24[:, 1:577, :]  # Image tokens (1–576)
        # Extract text tokens (index 577 to the end)
        #text_tokens24 = masked_layer_states24[:, 577:, :]  # Text tokens (577 onward)

        # Get corresponding masks for image and text tokens
        #img_mask = expanded_attention_mask[:, 1:577, :]  # Mask for image tokens
        #text_mask = expanded_attention_mask[:, 577:, :]  # Mask for text tokens


#FOR FINETUNING


        # Extract image tokens (index 1 to 576 inclusive)
        img_tokens16 = masked_layer_states16[:, 35:611, :]  # Image tokens (35–611)
        # Extract text tokens (index 577 to the end)
        text_tokens16 = masked_layer_states16[:, 611:, :]  # Text tokens (611 onward)
        # Extract image tokens (index 1 to 576 inclusive)
        img_tokens24 = masked_layer_states24[:, 35:611, :]  # Image tokens (1–576)
        # Extract text tokens (index 577 to the end)
        text_tokens24 = masked_layer_states24[:, 611:, :]  # Text tokens (577 onward)
        # Get corresponding masks for image and text tokens
        img_mask = expanded_attention_mask[:, 35:611, :]  # Mask for image tokens
        text_mask = expanded_attention_mask[:, 611:, :]  # Mask for text tokens



        # Sum embeddings for image and text tokens
        img_token_sum16 = (img_tokens16 * img_mask).sum(dim=1)  # (batch_size, embedding_dim)
        text_token_sum16 = (text_tokens16 * text_mask).sum(dim=1)  # (batch_size, embedding_dim)

        # Sum embeddings for image and text tokens
        img_token_sum24 = (img_tokens24 * img_mask).sum(dim=1)  # (batch_size, embedding_dim)
        text_token_sum24 = (text_tokens24 * text_mask).sum(dim=1)  # (batch_size, embedding_dim)

        # Count valid tokens (non-padding) for image and text
        valid_img_token_counts = img_mask[:, :, 0].sum(dim=1, keepdim=True)  # (batch_size, 1)
        valid_text_token_counts = text_mask[:, :, 0].sum(dim=1, keepdim=True)  # (batch_size, 1)
        # Total valid token counts and total sum
        total_valid_token_counts = valid_img_token_counts + valid_text_token_counts  # (batch_size, 1)
        total_token_sum16 = img_token_sum16 + text_token_sum16  # (batch_size, embedding_dim)
        total_token_sum24 = img_token_sum24 + text_token_sum24  # (batch_size, embedding_dim)

        # Compute the average embeddings
        # Avoid division by zero using a small epsilon
        epsilon = 1e-8
        avg_img_embeddings16 = img_token_sum16 / (valid_img_token_counts + epsilon)  # (batch_size, embedding_dim)
        avg_text_embeddings16 = text_token_sum16 / (valid_text_token_counts + epsilon)  # (batch_size, embedding_dim)
        avg_all_tokens_embeddings16 = total_token_sum16 / (total_valid_token_counts + epsilon)  # (batch_size, embedding_dim)

        avg_img_embeddings24 = img_token_sum24 / (valid_img_token_counts + epsilon)  # (batch_size, embedding_dim)
        avg_text_embeddings24 = text_token_sum24 / (valid_text_token_counts + epsilon)  # (batch_size, embedding_dim)
        avg_all_tokens_embeddings24 = total_token_sum24 / (total_valid_token_counts + epsilon)  # (batch_size, embedding_dim)

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        text_loss = None
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            text_loss = loss_fct(shift_logits, shift_labels)

            # First, create the model (do this once during your main model initialization)
            self.broadcast_model = BroadcastModel(llm_hidden_dim, txt_cls_dim, img_cls_dim).to(hidden_states.device)

            # Then when computing losses, call it like this:
            total_broadcast_loss16, loss_dict = self.broadcast_model.compute_broadcast_losses(
                avg_all_tokens_embeddings16,
                avg_text_embeddings16,
                avg_img_embeddings16,
                txt_input_cls_embeds,
                img_input_cls_embeds
            )

        #loss = text_loss + total_broadcast_loss16
        loss = text_loss

        try:
            if dist.get_rank() == 0:
                if loss > text_loss:
                    log_dict = {
                        "total loss": loss,
                        "text_cross_entropy_loss": text_loss,
                        "all_to_text_cls_loss": loss_dict['all_to_txt'],
                        "all_to_image_cls_loss": loss_dict['all_to_img'],
                        "text_to_text_cls_loss": loss_dict['txt_to_txt'],
                        "image_to_image_cls_loss": loss_dict['img_to_img'],
                        "text_to_image_cls_loss": loss_dict['txt_to_img'],
                        "image_to_text_cls_loss": loss_dict['img_to_txt'],
                        "total_broadcast_loss": total_broadcast_loss16,
                    }
                    filtered_log_dict = {key: value for key, value in log_dict.items() if value > 0}
                    wandb.log(filtered_log_dict)
                else:
                    wandb.log({
                        "total loss": loss,
                        "text_cross_entropy_loss": text_loss,
                        "all_to_text_cls_loss": loss_dict['all_to_txt'],
                        "all_to_image_cls_loss": loss_dict['all_to_img'],
                        "text_to_text_cls_loss": loss_dict['txt_to_txt'],
                        "image_to_image_cls_loss": loss_dict['img_to_img'],
                        "text_to_image_cls_loss": loss_dict['txt_to_img'],
                        "image_to_text_cls_loss": loss_dict['img_to_txt'],
                        "total_broadcast_loss": total_broadcast_loss16,
                    })

                self.steps += 1
        except:
            pass
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )    
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        img_input_cls_embeds: Optional[torch.FloatTensor] = None,
        txt_input_cls_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                img_input_cls_embeds,
                txt_input_cls_embeds
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        return self._forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            img_input_cls_embeds= img_input_cls_embeds,
            txt_input_cls_embeds= txt_input_cls_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

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
                _,
                img_input_cls_embeds, 
                txt_input_cls_embeds
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
