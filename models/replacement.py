from __future__ import annotations
import os
# 获取当前工作目录
current_directory = os.getcwd()
# 获取上一层目录
parent_directory = os.path.dirname(current_directory)
import sys
sys.path.append(parent_directory)
import math
import os.path
from typing import Any, Callable, Dict, List, Optional, Union
import torch, gc
from functools import partial
import argparse
from torch import tensor
from diffusers import LMSDiscreteScheduler, UNet2DConditionModel, AutoencoderKL, DDPMScheduler, DDIMScheduler, \
	DPMSolverMultistepScheduler, StableDiffusionPipeline
from transformers import AutoTokenizer, CLIPTextModel, CLIPImageProcessor
from diffusers.pipelines.stable_diffusion import (
	StableDiffusionPipelineOutput,
	StableDiffusionSafetyChecker,
)
from diffusers import StableDiffusionAttendAndExcitePipeline
from diffusers.models.attention_processor import AttnProcessor2_0
import logging
from diffusers.loaders import TextualInversionLoaderMixin
from copy import deepcopy
import copy
import numpy as npd
from matplotlib import pyplot as plt
import collections
# from peft import LoraConfig
# from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
# from safetensors.torch import load_file as load_safetensors
from datetime import datetime
logger = logging.getLogger("global_logger")
is_debug = True
LOG_dir = "logs"
# Create log filename with timestamp using the existing LOG_dir
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_dir, f"attention_log_{timestamp}.txt")
print(f"log_file:{log_file}")
os.makedirs(LOG_dir, exist_ok=True)
def get_peft_kwargs(rank_dict, network_alpha_dict, peft_state_dict, is_unet=True):
    rank_pattern = {}
    alpha_pattern = {}
    r = lora_alpha = list(rank_dict.values())[0]
    if len(set(rank_dict.values())) > 1:
        # get the rank occuring the most number of times
        r = collections.Counter(rank_dict.values()).most_common()[0][0]
        # for modules with rank different from the most occuring rank, add it to the rank_pattern
        rank_pattern = dict(filter(lambda x: x[1] != r, rank_dict.items()))
        rank_pattern = {k.split(".lora_B.")[0]: v for k, v in rank_pattern.items()}
    if network_alpha_dict is not None and len(network_alpha_dict) > 0:
        if len(set(network_alpha_dict.values())) > 1:
            # get the alpha occuring the most number of times
            lora_alpha = collections.Counter(network_alpha_dict.values()).most_common()[0][0]
            # for modules with alpha different from the most occuring alpha, add it to the alpha_pattern
            alpha_pattern = dict(filter(lambda x: x[1] != lora_alpha, network_alpha_dict.items()))
            if is_unet:
                alpha_pattern = {
                    ".".join(k.split(".lora_A.")[0].split(".")).replace(".alpha", ""): v
                    for k, v in alpha_pattern.items()
                }
            else:
                alpha_pattern = {".".join(k.split(".down.")[0].split(".")[:-1]): v for k, v in alpha_pattern.items()}
        else:
            lora_alpha = set(network_alpha_dict.values()).pop()
    # layer names without the Diffusers specific
    target_modules = list({name.split(".lora")[0] for name in peft_state_dict.keys()})
    lora_config_kwargs = {
        "r": r,
        "lora_alpha": lora_alpha,
        "rank_pattern": rank_pattern,
        "alpha_pattern": alpha_pattern,
        "target_modules": target_modules,
    }
    return lora_config_kwargs
def check_and_detach(tensor):
	if tensor.requires_grad:
		return tensor.detach()
	else:
		return tensor
class my_attn_processor:
	"""
	Attention processor for free guidance.
	"""
	def __init__(self, func, masks:dict[int,torch.Tensor], lenOfTokens, place_in_unet=None, is_vis = False, step_vis=10, guidance_step_interval = None, folder_path=None):
		self.guidance_func = func
		self.tokens = range(lenOfTokens)
		self.masks = masks
		self.place_in_unet = place_in_unet
		self.cur_step = 0
		self.is_vis = is_vis
		self.step_vis = step_vis
		self.folder_path = folder_path
		if guidance_step_interval is not None:
			self.guidance_step_interval = range(guidance_step_interval[0], guidance_step_interval[1])
		else:
			self.guidance_step_interval = None
	
	def visualize_attention(self, attention_probs, step, is_original=True):
		"""可视化注意力图
		Args:
			attention_probs: 注意力概率 [B, HW, N] 其中N可能是HW(self)或lenOfTokens(cross)
			step: 当前步数
			is_original: 是否为原始注意力图（True为原始，False为编辑后）
		"""
		if (step != 1) and (step % self.step_vis != 0):
			return
			
		import matplotlib.pyplot as plt
		import os
		import math
		
		# Use folder_path instead of hardcoded directory
		if self.folder_path is None:
			save_dir = f"attention_vis/step_{step}"
		else:
			save_dir = os.path.join(self.folder_path, f"attention_vis/step_{step}")
		os.makedirs(save_dir, exist_ok=True)
		
		is_self = "attn1" in self.place_in_unet
		B, HW, N = attention_probs.shape
		H = W = int(math.sqrt(HW))
		
		status = "orig" if is_original else "edited"
		
		if is_self:  # self-attention [B, 1024, 1024]
			# flag = (self.place_in_unet != "up_blocks.3.attentions.2.transformer_blocks.0.attn1.processor")
			# if flag:  # skip the vis
			# 	return
			from sklearn.decomposition import PCA
			import numpy as np
			import torchvision.transforms as T
			from PIL import Image
			pca = PCA(n_components=3)
			
			# Average across batch dimension first
			attention_mean = attention_probs.mean(0).detach().cpu().numpy()  # Shape: [1024, 1024]
			attention_pca = pca.fit_transform(attention_mean)  # Shape: [1024, 3]
			
			# Reshape to image dimensions
			attention_pca = attention_pca.reshape(H, W, 3)
			
			# Normalize and convert to image
			pca_img_min = attention_pca.min(axis=(0, 1), keepdims=True)
			pca_img_max = attention_pca.max(axis=(0, 1), keepdims=True)
			pca_img_normalized = (attention_pca - pca_img_min) / (pca_img_max - pca_img_min + 1e-5)
			pca_img_uint8 = (pca_img_normalized * 255).astype(np.uint8)
			pca_image = Image.fromarray(pca_img_uint8)
			pca_image = T.Resize(512, interpolation=T.InterpolationMode.NEAREST)(pca_image)
			pca_image.save(os.path.join(save_dir, f"step{step}_{self.place_in_unet}self_{status}_pca.png"))
        
			
		else:  # cross-attention [B, 1024, lenOfTokens]
			# 计算网格布局
			# return
			num_tokens = len(self.tokens)
			grid_size = math.ceil(math.sqrt(num_tokens))
			
			# 创建大图
			fig = plt.figure(figsize=(4 * grid_size, 4 * grid_size))
			fig.suptitle(f"Cross Attention ({status}) - {self.place_in_unet} - Step {step} - {num_tokens} Tokens", fontsize=16)
			
			# 绘制每个token的注意力图
			for idx, token_idx in enumerate(self.tokens):
				attn_map = attention_probs.mean(0)[:, token_idx].reshape(H, W)  # [32, 32]
				
				ax = plt.subplot(grid_size, grid_size, idx + 1)
				im = ax.imshow(attn_map.cpu().detach().numpy())
				ax.set_title(f"Token {token_idx}")
				ax.axis('off')
				plt.colorbar(im, ax=ax)
			
			plt.tight_layout()
			plt.savefig(f"{save_dir}/step{step}_{self.place_in_unet}_cross_{status}.png")
			plt.close()
	def __call__(
		self,
		attn,
		hidden_states: torch.FloatTensor,
		encoder_hidden_states: Optional[torch.FloatTensor] = None,
		attention_mask: Optional[torch.FloatTensor] = None,
		temb: Optional[torch.FloatTensor] = None,
		scale: float = 1.0,
	) -> torch.Tensor:
		batch_size = hidden_states.shape[0]
		
		# Update call count and current step
		self.cur_step += 1
		
		residual = hidden_states
		query = attn.to_q(hidden_states)
		if encoder_hidden_states is None:
			encoder_hidden_states = hidden_states
		key = attn.to_k(encoder_hidden_states)
		value = attn.to_v(encoder_hidden_states)
		query = attn.head_to_batch_dim(query)
		key = attn.head_to_batch_dim(key)
		value = attn.head_to_batch_dim(value)
		attention_probs = attn.get_attention_scores(query, key, attention_mask)
		if self.is_vis:
			# 可视化原始注意力
			self.visualize_attention(attention_probs, self.cur_step, is_original=True)
			
			with open(log_file, "a") as f:
				f.write(f"Place in UNet: {self.place_in_unet}\n")
				f.write(f"Before guidance - attention_probs shape: {attention_probs.shape} in step {self.cur_step}\n")
		
		if self.guidance_step_interval is not None:
			if self.cur_step not in self.guidance_step_interval:
				# print(f"Not in guidance_step_interval: {self.cur_step}")
				hidden_states = torch.bmm(attention_probs, value)
				hidden_states = attn.batch_to_head_dim(hidden_states)
				# Linear projection
				hidden_states = attn.to_out[0](hidden_states)
				# Dropout
				hidden_states = attn.to_out[1](hidden_states)
				if attn.residual_connection:
					hidden_states = hidden_states + residual
				hidden_states = hidden_states / attn.rescale_output_factor
				return hidden_states
		
		# determine this layer’s spatial resolution
		BHW = attention_probs.shape[1]     # e.g. 64*64=4096
		H = int(math.sqrt(BHW))            # gives 64, 32, 16, or 8
		# look up the correct mask
		region = self.masks[H]             # your {64:mask64, 32:mask32,…}
		# now apply guidance to only that region
		attention_probs = self.guidance_func(
			attention_probs.unsqueeze(0),
			region,
			self.tokens
		).squeeze(0)
		
		if self.is_vis:
			# 可视化编辑后的注意力
			self.visualize_attention(attention_probs, self.cur_step, is_original=False)
			
			with open(log_file, "a") as f:
				f.write(f"After guidance - attention_probs shape: {attention_probs.shape} in step {self.cur_step}\n\n")
		
		hidden_states = torch.bmm(attention_probs, value)
		hidden_states = attn.batch_to_head_dim(hidden_states)
		# Linear projection
		hidden_states = attn.to_out[0](hidden_states)
		# Dropout
		hidden_states = attn.to_out[1](hidden_states)
		if attn.residual_connection:
			hidden_states = hidden_states + residual
		hidden_states = hidden_states / attn.rescale_output_factor
		return hidden_states
class SelfAttentionStore(AttnProcessor2_0):
	"""
	Stores self-attention maps during original generation for later blending.
	"""
	def __init__(self, place_in_unet=None):
		super().__init__()
		self.place_in_unet = place_in_unet
		self.stored_attentions = {}  # Store by timestep
		self.cur_step = 0
	
	def __call__(
		self,
		attn: torch.nn.Module,
		hidden_states: torch.FloatTensor,
		encoder_hidden_states: Optional[torch.FloatTensor] = None,
		attention_mask: Optional[torch.FloatTensor] = None,
		temb: Optional[torch.FloatTensor] = None,
		scale: float = 1.0
	) -> torch.FloatTensor:
		# Standard attention computation
		residual = hidden_states
		
		if attn.spatial_norm is not None:
			hidden_states = attn.spatial_norm(hidden_states, temb)
		input_ndim = hidden_states.ndim
		if input_ndim == 4:
			batch_size, channel, height, width = hidden_states.shape
			hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
		
		batch_size, sequence_length, _ = hidden_states.shape
		
		if attention_mask is not None:
			attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
			attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
		
		if attn.group_norm is not None:
			hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
		
		query = attn.to_q(hidden_states, scale=scale)
		key = attn.to_k(hidden_states, scale=scale)
		value = attn.to_v(hidden_states, scale=scale)
		
		inner_dim = key.shape[-1]
		head_dim = inner_dim // attn.heads
		
		query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
		key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
		value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
		
		# Compute attention scores
		attention_scores = torch.matmul(query, key.transpose(-1, -2)) * attn.scale
		if attention_mask is not None:
			attention_scores = attention_scores + attention_mask
		
		# Store the attention scores BEFORE softmax
		self.stored_attentions[self.cur_step] = attention_scores.detach().clone()
		self.cur_step += 1
		
		# Continue with standard attention
		attention_probs = attention_scores.softmax(dim=-1)
		hidden_states = torch.matmul(attention_probs, value)
		hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
		hidden_states = attn.to_out[0](hidden_states, scale=scale)
		hidden_states = attn.to_out[1](hidden_states)
		
		if input_ndim == 4:
			hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
		
		if attn.residual_connection:
			hidden_states = hidden_states + residual
		
		hidden_states = hidden_states / attn.rescale_output_factor
		
		return hidden_states


class SelfAttentionBlend(AttnProcessor2_0):
	"""
	Blends edited self-attention with stored original maps after step 30.
	"""
	def __init__(self, stored_attentions, blend_mask, place_in_unet=None, start_blending_step=30):
		super().__init__()
		self.stored_attentions = stored_attentions
		self.blend_mask = blend_mask  # Smooth mask: 1.0 inside edit region, 0.0 outside
		self.place_in_unet = place_in_unet
		self.start_blending_step = start_blending_step
		self.cur_step = 0
	
	def __call__(
		self,
		attn: torch.nn.Module,
		hidden_states: torch.FloatTensor,
		encoder_hidden_states: Optional[torch.FloatTensor] = None,
		attention_mask: Optional[torch.FloatTensor] = None,
		temb: Optional[torch.FloatTensor] = None,
		scale: float = 1.0
	) -> torch.FloatTensor:
		# Standard attention computation up to attention scores
		residual = hidden_states
		
		if attn.spatial_norm is not None:
			hidden_states = attn.spatial_norm(hidden_states, temb)
		input_ndim = hidden_states.ndim
		if input_ndim == 4:
			batch_size, channel, height, width = hidden_states.shape
			hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
		
		batch_size, sequence_length, _ = hidden_states.shape
		
		if attention_mask is not None:
			attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
			attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
		
		if attn.group_norm is not None:
			hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
		
		query = attn.to_q(hidden_states, scale=scale)
		key = attn.to_k(hidden_states, scale=scale)
		value = attn.to_v(hidden_states, scale=scale)
		
		inner_dim = key.shape[-1]
		head_dim = inner_dim // attn.heads
		
		query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
		key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
		value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
		
		# Compute current attention scores
		attention_scores = torch.matmul(query, key.transpose(-1, -2)) * attn.scale
		if attention_mask is not None:
			attention_scores = attention_scores + attention_mask
		
		# Blend with stored attention if after start_blending_step
		diffusion_step = self.cur_step // 16  # Approximate diffusion step
		if diffusion_step >= self.start_blending_step and self.cur_step in self.stored_attentions:
			original_scores = self.stored_attentions[self.cur_step]
			
			# Get the appropriate blend mask for this resolution
			import math
			H = int(math.sqrt(sequence_length))
			if H in self.blend_mask:
				mask = self.blend_mask[H]  # Should be [H*H, 1] or similar
				# Reshape mask for attention: [1, 1, HW, 1] to broadcast
				mask = mask.view(1, 1, sequence_length, 1).to(attention_scores.device)
				
				# Blend: use edited attention in edit region, original outside
				# mask is 1.0 in edit region, 0.0 outside
				attention_scores = mask * attention_scores + (1 - mask) * original_scores
				
				if diffusion_step == self.start_blending_step and self.cur_step % 16 == 0:
					print(f"Step {diffusion_step}: Starting self-attention blending at {self.place_in_unet}")
		
		self.cur_step += 1
		
		# Continue with standard attention
		attention_probs = attention_scores.softmax(dim=-1)
		hidden_states = torch.matmul(attention_probs, value)
		hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
		hidden_states = attn.to_out[0](hidden_states, scale=scale)
		hidden_states = attn.to_out[1](hidden_states)
		
		if input_ndim == 4:
			hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
		
		if attn.residual_connection:
			hidden_states = hidden_states + residual
		
		hidden_states = hidden_states / attn.rescale_output_factor
		
		return hidden_states


class my_attn_processor2(AttnProcessor2_0):
	"""
	Attention processor for free guidance.
	"""
	def __init__(self, func, masks:dict[int,torch.Tensor], lenOfTokens, place_in_unet=None, is_vis = False, step_vis=10, guidance_step_interval = None, folder_path=None, text_encoder=None, tokenizer=None):
		self.guidance_func = func
		self.tokens = range(lenOfTokens)
		self.masks = masks
		self.place_in_unet = place_in_unet
		self.cur_step = 0
		# Add timestep tracking
		self.current_diffusion_step = 0
		# Initialize visualization and text encoder attributes
		self.is_vis = is_vis
		self.step_vis = step_vis
		self.folder_path = folder_path
		self.text_encoder = text_encoder
		self.tokenizer = tokenizer
		if guidance_step_interval is not None:
			self.guidance_step_interval = range(guidance_step_interval[0], guidance_step_interval[1])
		else:
			self.guidance_step_interval = None
		
	def reset_timestep(self):
		"""Reset timestep counter for new generation"""
		self.current_diffusion_step = 0
		
	def update_timestep(self, step):
		"""Update current diffusion timestep (0-49)"""
		self.current_diffusion_step = step
	
	def expand_value_tensor(self, value, add_tokens):
		"""Expand value tensor to include embeddings for new tokens"""
		if not add_tokens or self.text_encoder is None or self.tokenizer is None:
			return value
			
		# Get the dimensions we need to match
		device = value.device
		dtype = value.dtype
		batch_size, seq_len, hidden_dim = value.shape
		
		# Get real text encoder embeddings for the new tokens
		num_new_tokens = len(add_tokens)
		
		# Tokenize the new tokens and get their embeddings from the text encoder
		token_ids = self.tokenizer(
			add_tokens, 
			return_tensors="pt", 
			padding=True, 
			truncation=True,
			add_special_tokens=False
		)["input_ids"].to(device)
		
		# Get proper text encoder embeddings using the full pipeline
		# This gives us the same embeddings that the original prompt uses
		with torch.no_grad():
			# Encode the add_tokens through the full text encoder pipeline
			text_input = self.tokenizer(
				add_tokens,
				padding="max_length",
				max_length=self.tokenizer.model_max_length,
				truncation=True,
				return_tensors="pt",
			).to(device)
			
			# Get the full text encoder output (not just token embeddings)
			text_encoder_output = self.text_encoder(text_input.input_ids)[0]  # [batch, seq_len, 768]
			
			# Extract embeddings for our specific tokens (skip <start_of_text> token at index 0)
			# For each add_token, get its corresponding embedding
			new_text_embeddings = []
			for i, token in enumerate(add_tokens):
				# Find where this token appears in the tokenized sequence
				token_id = self.tokenizer.convert_tokens_to_ids(token)
				# Get the embedding at position 1 (after <start_of_text>)
				token_embedding = text_encoder_output[0, 1 + i, :]  # [768]
				new_text_embeddings.append(token_embedding)
			
			# Stack all token embeddings: [num_tokens, 768]
			stacked_embeddings = torch.stack(new_text_embeddings, dim=0)
		
		# The value tensor in cross-attention should match the text encoder output dimensions
		# If there's still a mismatch, we need to understand why
		if stacked_embeddings.shape[-1] != hidden_dim:
			# This suggests the value tensor has been processed through some projection layer
			# We should use the same projection that was used for the original text embeddings
			# For now, create a learned projection that we can optimize
			if not hasattr(self, '_text_to_value_projection'):
				self._text_to_value_projection = torch.nn.Linear(768, hidden_dim, device=device, dtype=dtype)
			projected_embeddings = self._text_to_value_projection(stacked_embeddings)
		else:
			projected_embeddings = stacked_embeddings
		
		# Reshape to match expected dimensions [batch_size, num_new_tokens, hidden_dim]
		new_embeddings = projected_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
		
		# Concatenate with original values
		expanded_value = torch.cat([value, new_embeddings], dim=1)
		return expanded_value
		
	def visualize_attention(self, attention_probs, step, is_original=True):
		"""可视化注意力图
		Args:
			attention_probs: 注意力概率 [B, heads, HW, N] for my_attn_processor2
			step: 当前步数
			is_original: 是否为原始注意力图（True为原始，False为编辑后）
		"""
		if (step != 1) and (step % self.step_vis != 0):
			return
			
		import matplotlib.pyplot as plt
		import os
		import math
		
		# Use folder_path instead of hardcoded directory
		if self.folder_path is None:
			save_dir = f"attention_vis/step_{step}"
		else:
			save_dir = os.path.join(self.folder_path, f"attention_vis/step_{step}")
		os.makedirs(save_dir, exist_ok=True)
		
		is_self = "attn1" in self.place_in_unet
		
		# Handle 4D tensors from my_attn_processor2: [B, heads, HW, N]
		if len(attention_probs.shape) == 4:
			B, heads, HW, N = attention_probs.shape
			# Average across heads for visualization
			attention_probs = attention_probs.mean(dim=1)  # [B, HW, N]
		else:
			B, HW, N = attention_probs.shape
		H = W = int(math.sqrt(HW))
		
		status = "orig" if is_original else "edited"
		
		if is_self:  # self-attention [B, 1024, 1024]
			# 只保存第一个batch的注意力图
			attn_map = attention_probs[0].reshape(H, W, H, W)  # [32, 32, 32, 32]
			# 取中心点的注意力分布
			center_h, center_w = H//2, W//2
			center_attn = attn_map[center_h, center_w].reshape(H, W)  # [32, 32]
			
			plt.figure(figsize=(8, 8))
			plt.imshow(center_attn.cpu().detach().numpy())
			plt.colorbar()
			plt.title(f"Self Attention ({status}) - {self.place_in_unet} - Step {step}")
			plt.savefig(f"{save_dir}/step{step}_{self.place_in_unet}_self_{status}.png")
			plt.close()
			
		else:  # cross-attention [B, 1024, lenOfTokens]
			# Use actual tensor dimensions instead of self.tokens to catch injected tokens
			actual_num_tokens = N  # This will be 78 if butterfly was injected, 77 if not
			tokens_to_show = range(actual_num_tokens)  # Show ALL tokens including injected ones
			
			grid_size = math.ceil(math.sqrt(actual_num_tokens))
			fig = plt.figure(figsize=(4 * grid_size, 4 * grid_size))
			fig.suptitle(f"Cross Attention ({status}) - {self.place_in_unet} - Step {step} - {actual_num_tokens} Tokens", fontsize=16)
			for idx, token_idx in enumerate(tokens_to_show):
				if token_idx >= N:  # Safety check
					break
				attn_map = attention_probs[0, :, token_idx].reshape(H, W)  # [32, 32]
				
				ax = plt.subplot(grid_size, grid_size, idx + 1)
				im = ax.imshow(attn_map.cpu().detach().numpy())
				ax.set_title(f"Token {token_idx}")
				ax.axis('off')
				plt.colorbar(im, ax=ax)
			
			plt.tight_layout()
			plt.savefig(f"{save_dir}/step{step}_{self.place_in_unet}_cross_{status}.png")
			plt.close()
	def __call__(
		self,
		attn: torch.nn.Module,
		hidden_states: torch.FloatTensor,
		encoder_hidden_states: Optional[torch.FloatTensor] = None,
		attention_mask: Optional[torch.FloatTensor] = None,
		temb: Optional[torch.FloatTensor] = None,
		scale: float = 1.0
	) -> torch.FloatTensor:
		residual = hidden_states
		
		if attn.spatial_norm is not None:
			hidden_states = attn.spatial_norm(hidden_states, temb)
		input_ndim = hidden_states.ndim
		if input_ndim == 4:
			batch_size, channel, height, width = hidden_states.shape
			hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
		batch_size, sequence_length, _ = (
			hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
		)
		if attention_mask is not None:
			attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
			attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
		if attn.group_norm is not None:
			hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
		query = attn.to_q(hidden_states, scale=scale)
		if encoder_hidden_states is None:
			encoder_hidden_states = hidden_states
		elif attn.norm_cross:
			encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
		key = attn.to_k(encoder_hidden_states, scale=scale)
		value = attn.to_v(encoder_hidden_states, scale=scale)
		
		# Store original value for potential expansion
		original_value = value
		
		inner_dim = key.shape[-1]
		head_dim = inner_dim // attn.heads
		query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
		key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
		value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
		# 计算注意力分数
		attention_scores = torch.matmul(query, key.transpose(-1, -2)) * attn.scale
		if attention_mask is not None:
			attention_scores = attention_scores + attention_mask
		
		# Update call count and current step
		self.cur_step += 1
		
		
		# APPLY GUIDANCE TO RAW SCORES BEFORE SOFTMAX!
		# Always apply guidance (we already filter to attn2 layers in init_model)
		
		# Check if we should apply guidance in this step
		if self.guidance_step_interval is not None:
			if self.cur_step not in self.guidance_step_interval:
				# Skip guidance, just apply softmax and continue
				attention_probs = attention_scores.softmax(dim=-1)
				hidden_states = torch.matmul(attention_probs, value)
				hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
				hidden_states = attn.to_out[0](hidden_states, scale=scale)
				hidden_states = attn.to_out[1](hidden_states)
				if input_ndim == 4:
					hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
				if attn.residual_connection:
					hidden_states = hidden_states + residual
				hidden_states = hidden_states / attn.rescale_output_factor
				return hidden_states
		
		if self.is_vis:
			with open(log_file, "a") as f:
				f.write(f"Place in UNet: {self.place_in_unet}\n")
				f.write(f"Before guidance - attention_scores shape: {attention_scores.shape}\n")
		
		# 1) reshape to [batch*heads, HW, tokens]  
		seq_len = attention_scores.shape[-2]
		attention_scores_reshaped = attention_scores.view(-1, seq_len, attention_scores.shape[-1])
		# 2) pick the right H×H mask
		BHW = seq_len
		H   = int(math.sqrt(BHW))
		region = self.masks[H]  # your dict of {64:mask64,32:mask32,…}
		# 3) apply guidance to RAW SCORES - may return expanded dimensions and token info
		# The guidance function will use the global denoising tracker directly
		guidance_result = self.guidance_func(
			attention_scores_reshaped.unsqueeze(0),
			region,
			self.tokens
		)
		
		# Handle different return types (old functions return tensor, new ones return tuple)
		if isinstance(guidance_result, tuple):
			processed_scores = guidance_result[0].squeeze(0)
			injected_tokens = guidance_result[1] if len(guidance_result) > 1 else []
		else:
			processed_scores = guidance_result.squeeze(0)
			injected_tokens = []
			
		# 4) reshape back to [batch, heads, HW, tokens] - may have expanded token dimension
		attention_scores = processed_scores.view(batch_size, attn.heads, -1, processed_scores.shape[-1])
		
		# No need to expand value tensor - extended prompt handles it naturally
		
		if self.is_vis:
			with open(log_file, "a") as f:
				f.write(f"After guidance - attention_scores shape: {attention_scores.shape}\n\n")
		
		# NOW apply softmax to the modified scores
		attention_probs = attention_scores.softmax(dim=-1)
		
		# Debug removed - processor2 is now working
		
		if self.is_vis:
			self.visualize_attention(attention_probs, self.cur_step, is_original=False)
		
		# Check dimension consistency (should match now with value expansion)
		value_token_count = value.shape[-2]  # Number of tokens in value (may be expanded)
		attention_token_count = attention_probs.shape[-1]  # Number of tokens in attention
		
		if attention_token_count != value_token_count:
			# This should not happen with proper value expansion
			raise RuntimeError(f"Dimension mismatch - attention={attention_token_count} vs value={value_token_count}")
		
		# Use the SAME computation method as working my_attn_processor  
		# Reshape attention_probs to match bmm requirements: [batch*heads, seq_len, seq_len]
		attention_probs_bmm = attention_probs.reshape(-1, attention_probs.shape[-2], attention_probs.shape[-1])
		value_bmm = value.reshape(-1, value.shape[-2], value.shape[-1])
		
		# Ensure dtype consistency for BMM operation
		attention_probs_bmm = attention_probs_bmm.to(value_bmm.dtype)
		
		hidden_states = torch.bmm(attention_probs_bmm, value_bmm)
		hidden_states = attn.batch_to_head_dim(hidden_states)
		# linear proj (NO scale parameter like working version)
		hidden_states = attn.to_out[0](hidden_states)
		# dropout
		hidden_states = attn.to_out[1](hidden_states)
		if input_ndim == 4:
			hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
		if attn.residual_connection:
			hidden_states = hidden_states + residual
		hidden_states = hidden_states / attn.rescale_output_factor
		return hidden_states

