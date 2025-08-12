"""
Instant Attention Mask generation adapted from InstDiffEdit
for use with ControlNet scribble editing pipeline
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageChops
from torchvision import transforms as tfms
import cv2
from typing import List, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity

# AttentionControl base class
class AttentionControl:
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return 0  # We don't use LOW_RESOURCE mode
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError
    
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        # Only process conditional attention (second half)
        h = attn.shape[0]
        attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):
    """Stores attention maps during diffusion process."""
    
    @staticmethod
    def get_empty_store():
        return {
            "down_cross": [], "mid_cross": [], "up_cross": [],
            "down_self": [], "mid_self": [], "up_self": []
        }
    
    def __init__(self):
        super().__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        # Store attention maps - handle different tensor shapes
        if len(attn.shape) == 3:  # [batch*heads, seq, seq]
            seq_len = attn.shape[1]
        elif len(attn.shape) == 4:  # [batch, heads, seq, seq]
            seq_len = attn.shape[2]
        else:
            seq_len = attn.shape[-2] if len(attn.shape) > 1 else 0
            
        if seq_len <= 32 ** 2 and seq_len > 0:  # Avoid memory overhead
            # Clone to avoid in-place modifications
            self.step_store[key].append(attn.clone().detach().cpu())
        return attn
    
    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                if key in self.step_store:
                    # Ensure both lists have same length
                    min_len = min(len(self.attention_store[key]), len(self.step_store[key]))
                    for i in range(min_len):
                        if i < len(self.step_store[key]):
                            # Move to same device and accumulate
                            device = self.attention_store[key][i].device
                            step_tensor = self.step_store[key][i].to(device)
                            self.attention_store[key][i] = self.attention_store[key][i] + step_tensor
        self.step_store = self.get_empty_store()
    
    def get_average_attention(self):
        average_attention = {
            key: [item / self.cur_step for item in self.attention_store[key]] 
            for key in self.attention_store
        }
        return average_attention
    
    def reset(self):
        super().reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


def register_attention_control(pipe, controller):
    """Register attention control hooks in the pipeline's UNet model for modern diffusers."""
    
    # Store original forward methods
    original_forwards = {}
    
    def register_forward_hook(module, place_in_unet):
        """Create a forward hook for modern diffusers Attention modules."""
        original_forward = module.forward
        original_forwards[id(module)] = original_forward
        
        def custom_forward(hidden_states, encoder_hidden_states=None, attention_mask=None, 
                          temb=None, scale=1.0, **cross_attention_kwargs):
            # Determine if this is cross-attention
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else hidden_states
            
            batch_size, sequence_length, _ = (
                hidden_states.shape if hidden_states.ndim == 3 else 
                (hidden_states.shape[0], hidden_states.shape[1], hidden_states.shape[2])
            )
            
            # Get query, key, value projections
            if hasattr(module, 'to_q'):
                query = module.to_q(hidden_states)
                key = module.to_k(encoder_hidden_states)
                value = module.to_v(encoder_hidden_states)
            else:
                # For combined qkv projection
                qkv = module.to_qkv(hidden_states)
                query, key, value = qkv.chunk(3, dim=-1)
            
            # Get attention dimension info
            inner_dim = key.shape[-1]
            head_dim = inner_dim // module.heads if hasattr(module, 'heads') else getattr(module, 'head_dim', 64)
            heads = inner_dim // head_dim
            
            # Reshape for attention computation
            query = query.view(batch_size, -1, heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, heads, head_dim).transpose(1, 2)
            
            # Compute attention scores
            if hasattr(module, 'scale'):
                attention_scores = torch.matmul(query, key.transpose(-1, -2)) * module.scale
            else:
                attention_scores = torch.matmul(query, key.transpose(-1, -2)) / (head_dim ** 0.5)
            
            # Apply attention mask if provided
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            
            # Get attention probabilities
            attention_probs = attention_scores.softmax(dim=-1)
            
            # Reshape for controller: [batch*heads, seq, seq]
            batch_size_heads = batch_size * heads
            attention_probs_reshaped = attention_probs.view(batch_size_heads, -1, attention_probs.shape[-1])
            
            # Apply attention control
            attention_probs_reshaped = controller(attention_probs_reshaped, is_cross, place_in_unet)
            
            # Reshape back
            attention_probs = attention_probs_reshaped.view(batch_size, heads, -1, attention_probs_reshaped.shape[-1])
            
            # Apply attention to values
            hidden_states = torch.matmul(attention_probs, value)
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim)
            
            # Apply output projection
            if hasattr(module, 'to_out'):
                if isinstance(module.to_out, torch.nn.ModuleList):
                    hidden_states = module.to_out[0](hidden_states)
                    hidden_states = module.to_out[1](hidden_states)  # dropout
                else:
                    hidden_states = module.to_out(hidden_states)
            
            return hidden_states
        
        module.forward = custom_forward
    
    def register_recr(module, module_name, count, place_in_unet):
        """Recursively register hooks for attention modules."""
        # Debug: Print module info
        if hasattr(module, '__class__'):
            class_name = module.__class__.__name__
            
            # Check for Attention module (modern diffusers naming)
            if class_name in ['Attention', 'CrossAttention', 'MemoryEfficientCrossAttention']:
                print(f"Found attention module: {module_name} ({class_name})")
                # Register all attention modules - let the forward function determine cross vs self
                register_forward_hook(module, place_in_unet)
                return count + 1
        
        # Recursively process children
        for name, child in module.named_children():
            full_name = f"{module_name}.{name}" if module_name else name
            count = register_recr(child, full_name, count, place_in_unet)
        
        return count
    
    # Reset controller
    controller.reset()
    
    # Register hooks in UNet blocks
    cross_att_count = 0
    
    # Process down blocks
    if hasattr(pipe.unet, 'down_blocks'):
        for i, block in enumerate(pipe.unet.down_blocks):
            cross_att_count = register_recr(block, f"down_blocks.{i}", cross_att_count, "down")
    
    # Process mid block
    if hasattr(pipe.unet, 'mid_block'):
        cross_att_count = register_recr(pipe.unet.mid_block, "mid_block", cross_att_count, "mid")
    
    # Process up blocks
    if hasattr(pipe.unet, 'up_blocks'):
        for i, block in enumerate(pipe.unet.up_blocks):
            cross_att_count = register_recr(block, f"up_blocks.{i}", cross_att_count, "up")
    
    controller.num_att_layers = cross_att_count
    print(f"Registered {cross_att_count} attention layers")


def aggregate_attention(batch_size, attention_store, res: int, 
                        from_where: List[str], is_cross: bool, select: int):
    """Aggregate attention maps from specified layers."""
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    
    for location in from_where:
        key = f"{location}_{'cross' if is_cross else 'self'}"
        if key in attention_maps:
            for item in attention_maps[key]:
                # Handle different attention tensor shapes
                if len(item.shape) == 3:  # [batch*heads, seq, seq]
                    seq_len = item.shape[1]
                elif len(item.shape) == 4:  # [batch, heads, seq, seq] 
                    seq_len = item.shape[2]
                else:
                    continue
                
                if seq_len == num_pixels:
                    # Move to CPU for processing
                    item_cpu = item.cpu()
                    
                    if len(item_cpu.shape) == 3:  # [batch*heads, seq, seq]
                        # Reshape to [batch*heads, res, res, num_tokens]
                        heads = item_cpu.shape[0] // batch_size if batch_size > 1 else item_cpu.shape[0]
                        cross_maps = item_cpu.reshape(heads, res, res, item_cpu.shape[-1])
                        
                    elif len(item_cpu.shape) == 4:  # [batch, heads, seq, seq]
                        # Reshape to [batch, heads, res, res, num_tokens]
                        cross_maps = item_cpu.reshape(item_cpu.shape[0], item_cpu.shape[1], res, res, item_cpu.shape[-1])
                        # Average over batch dimension if needed
                        if batch_size > 1:
                            cross_maps = cross_maps.mean(dim=0)
                        cross_maps = cross_maps.view(-1, res, res, cross_maps.shape[-1])
                    
                    out.append(cross_maps)
    
    if len(out) == 0:
        # Return empty tensor if no attention maps found
        print(f"Warning: No attention maps found for {from_where} {'cross' if is_cross else 'self'}")
        return torch.zeros(res, res, 77)  # Default token length
    
    # Concatenate along heads dimension
    out = torch.cat(out, dim=0)
    
    # Average over heads dimension
    if len(out.shape) == 4:  # [heads, res, res, tokens]
        out = out.mean(dim=0)  # [res, res, tokens]
    
    return out


def cal_sim(images, base_index=0):
    """Calculate cosine similarity between attention maps."""
    base = np.array(images[base_index]).flatten()
    image = images.copy()
    image.pop(base_index)
    scores = []
    max_score = 0
    max_index = 0
    
    for i, im in enumerate(image):
        im = np.array(im).flatten()
        sims = cosine_similarity(im.reshape(1, -1), base.reshape(1, -1))
        scores.append(float(sims[0]))
        if sims > max_score:
            max_score = sims
            max_index = i + 1
    
    scores.insert(base_index, 1.0)
    return max_index, scores


def get_position(images, position=[], th1=0.9, th2=0.6):
    """Classify tokens as foreground, background, or neutral."""
    indexx, _ = cal_sim(images[:-1])
    _, scores = cal_sim(images[1:-1], indexx-1)
    position.append(0)
    
    for score in scores:
        if score >= th1:
            position.append(1)  # Foreground
        elif score <= th2:
            position.append(-1)  # Background
        else:
            position.append(0)  # Neutral
    
    position.append(0)
    return position


def token_combination(tokens, prompt_list, attention_maps, tokenizer, lens=64):
    """Combine attention maps for multi-token words."""
    flag = 0
    j = 0
    attention_token = ''
    attens = []
    images = []
    
    if len(tokens) > len(prompt_list) + 2:
        flag = 1
    
    if flag == 0:
        # Simple case: one token per word
        for i in range(len(tokens)):
            image = attention_maps[:, :, i]
            image = 255 * image / image.max()
            image = Image.fromarray(image.numpy().astype(np.uint8)).resize((lens, lens))
            images.append(image)
    else:
        # Complex case: multiple tokens per word
        for i in range(len(tokens)):
            if i == 0 or i == len(tokens)-1:
                image = attention_maps[:, :, i]
            else:
                inst = tokenizer.decode(int(tokens[i]))
                if inst == prompt_list[j]:
                    image = attention_maps[:, :, i]
                else:
                    attention_token = attention_token + inst
                    attens.append(attention_maps[:, :, i])
                    if attention_token.lower() == prompt_list[j].lower():
                        image = torch.stack(attens, 0).mean(0)
                        attention_token = ''
                        attens = []
                    else:
                        continue
                j = j + 1
            
            image = 255 * image / image.max()
            image = Image.fromarray(image.numpy().astype(np.uint8)).resize((lens, lens))
            images.append(image)
    
    # Normalize images
    for i in range(len(images)):
        if i == 0:
            images[i] = ImageChops.invert(images[i])
        bi_diff = tfms.ToTensor()(images[i])[0]
        bi_diff = (bi_diff - bi_diff.min()) / (bi_diff.max() - bi_diff.min()) * 255.
        bi_diff = np.array(bi_diff).astype(np.uint8)
        bi_diff = Image.fromarray(bi_diff.astype(np.uint8))
        images[i] = bi_diff
    
    return images


def refine(images, position):
    """Refine attention maps by separating foreground and background."""
    back = None
    objects = None
    back_number = sum(k == -1 for k in position)
    objects_number = sum(k == 1 for k in position)
    
    for inst in range(len(images)):
        if position[inst] == 1:
            if objects is None:
                objects = tfms.ToTensor()(images[inst])
            else:
                objects = objects + tfms.ToTensor()(images[inst])
        elif position[inst] == -1:
            if back is None:
                back = tfms.ToTensor()(images[inst])
            else:
                back = back + tfms.ToTensor()(images[inst])
    
    if back is None:
        objects = objects / objects_number
        objects = (objects - objects.min()) / (objects.max() - objects.min())
        img = objects[0]
    else:
        back = back / back_number
        back = (back - back.min()) / (back.max() - back.min())
        objects = objects / objects_number
        objects = (objects - objects.min()) / (objects.max() - objects.min())
        img = objects[0] - back[0]
        img = torch.clamp(img, min=0.0)
    
    return img


def post_deal(img, th):
    """Post-process the mask with Gaussian blur and thresholding."""
    img = np.array(img)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = (img - img.min()) / (img.max() - img.min())
    mask_diff = Image.fromarray((img * 255.).astype(np.uint8))
    img = (img - img.min()) / (img.max() - img.min())
    img = np.array(img > th).astype(np.float32)
    mask_image = Image.fromarray((img * 255.).astype(np.uint8))
    return mask_diff, mask_image


def get_mask(position, batch_size, attention_store, prompt, tokenizer, 
             th=0.5, res=16, from_where=("up", "down")):
    """Generate mask from attention maps."""
    tokens = tokenizer.encode(prompt)
    prompt_list = prompt.split(' ')
    attention_maps = aggregate_attention(batch_size, attention_store, res, from_where, True, 0)
    images = token_combination(tokens, prompt_list, attention_maps, tokenizer)
    
    if len(images) == 3:
        # Simple case with few tokens
        img = tfms.ToTensor()(images[1])[0]
    else:
        # Complex case: classify and refine
        if position == []:
            position = get_position(images, position)
        img = refine(images, position)
    
    mask_diff, mask_image = post_deal(img, th)
    return mask_diff, mask_image, position


def use_mask(x_t, mask, or_latent):
    """Apply mask to blend generated and original latents."""
    # Resize mask to match latent dimensions
    mask = tfms.ToTensor()(mask.resize((x_t.shape[-2], x_t.shape[-1])))
    mask = mask.expand(x_t.shape[1], -1, -1).unsqueeze(0)
    
    # Move to same device
    device = x_t.device
    mask = mask.float().to(device)
    or_latent = or_latent.to(device)
    
    # Blend: keep original outside mask, use generated inside mask
    x_t = or_latent + mask * (x_t - or_latent)
    return x_t