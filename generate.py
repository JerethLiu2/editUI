#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TextCenGen: Attention-Guided Text-Centric Background Adaptation for Text-to-Image Generation
ICML 2025
"""

import os
current_directory = os.getcwd()
import sys
sys.path.append(current_directory)
import argparse
import torch
torch.set_grad_enabled(False)
from diffusers import AutoPipelineForText2Image, DiffusionPipeline
from datetime import datetime
import json
from models.replacement import my_attn_processor2
from utils.frame import Frame
from utils.guidance_replacement import replace_attn_with_move_object_against_single_point
from utils.ptp_utils import fill_tensor
from utils.vis_utils import save_images
from functools import partial
from diffusers import DPMSolverMultistepScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
import numpy as np
from PIL import Image
from utils.ptp_utils import fill_tensor
from utils.guidance_replacement import focus_and_squelch, focus_and_squelch_with_injection

class DenoisingStepTracker:
    """Global tracker for denoising steps during Stable Diffusion inference."""
    def __init__(self):
        self.current_step = 0
        self.total_steps = 0
        
    def reset(self, total_steps):
        """Reset tracker for new generation."""
        self.current_step = 0  # Will be incremented to 1 on first UNet call
        self.total_steps = total_steps
        
    def increment(self):
        """Increment step counter once per denoising timestep."""
        self.current_step += 1
        
    def get_step(self):
        """Get current denoising step."""
        return self.current_step

# Global tracker instance
denoising_tracker = DenoisingStepTracker()

def extract_substring(s):
    last_slash_index = s.rfind('/')
    if last_slash_index == -1:
        last_slash_index = 0
    else:
        last_slash_index += 1
    dot_index = s.find('.', last_slash_index)
    if dot_index == -1:
        dot_index = len(s)
    return s[last_slash_index:dot_index]


#skip self-attention
def create_masks_from_add_image(add_image_path):
    """Create masks for regional guidance from drawn add region image."""
    import cv2
    from scipy.ndimage import binary_fill_holes
    from PIL import Image as PILImage
    
    # Load and process the add image
    add_image = PILImage.open(add_image_path)
    print(f"Loaded add image: {add_image.size}, mode: {add_image.mode}")
    
    # Convert to grayscale for edge detection
    add_array = np.array(add_image.convert('L'))
    
    # Apply edge detection to find drawn regions
    edges = cv2.Canny(add_array, 10, 50)
    
    if edges.sum() == 0:
        print("Canny didn't detect edges. Trying threshold-based detection...")
        _, edges = cv2.threshold(255 - add_array, 10, 255, cv2.THRESH_BINARY)
    
    # Fill the interior of the drawn shape
    mask_2d = binary_fill_holes(edges).astype(np.float32)
    
    print(f"Add region mask coverage: {mask_2d.mean():.2%} of image")
    
    # Build masks for each U-Net cross-attn scale
    scales = [64, 32, 16, 8]
    masks = {}
    
    for s in scales:
        # Resize mask to match this scale
        mask_pil = PILImage.fromarray((mask_2d * 255).astype(np.uint8))
        mask_resized = mask_pil.resize((s, s), PILImage.NEAREST)
        mask_tensor = torch.from_numpy(np.array(mask_resized) / 255.0).float().to("cuda")
        
        # Invert mask (0 where we want to add, 1 where we want to preserve)
        masks[s] = 1.0 - mask_tensor
    
    return masks

def create_smooth_blend_masks(masks: dict[int, torch.Tensor], sigma=1.0):
    """Create smooth blend masks using Gaussian blur for natural transitions."""
    import torch.nn.functional as F
    smooth_masks = {}
    
    for size, mask in masks.items():
        # mask is [1, H, W, 1] binary mask
        mask_2d = mask[0, :, :, 0]  # [H, W]
        
        # Create Gaussian kernel for smoothing
        # Larger sigma = smoother transition
        kernel_size = max(3, int(2 * sigma) * 2 + 1)  # Ensure odd
        
        # Simple box blur as approximation (can replace with proper Gaussian)
        mask_expanded = mask_2d.unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]
        
        # Apply multiple passes of average pooling for smooth transition
        for _ in range(3):
            mask_expanded = F.avg_pool2d(
                F.pad(mask_expanded, (1, 1, 1, 1), mode='replicate'),
                kernel_size=3, stride=1, padding=0
            )
        
        # Ensure values are between 0 and 1
        mask_expanded = torch.clamp(mask_expanded, 0, 1)
        smooth_masks[size] = mask_expanded.squeeze(0).squeeze(0)  # [H, W]
    
    return smooth_masks

def init_model_store_self_attention(pipe, prompt):
    """Initialize model to store self-attention maps during original generation."""
    from models.replacement import SelfAttentionStore
    
    attn_procs = {}
    self_attention_stores = {}
    
    for name in pipe.unet.attn_processors.keys():
        if "attn1" in name:  # Self-attention layers
            store = SelfAttentionStore(place_in_unet=name)
            attn_procs[name] = store
            self_attention_stores[name] = store
        else:
            attn_procs[name] = pipe.unet.attn_processors[name]
    
    pipe.unet.set_attn_processor(attn_procs)
    return self_attention_stores

def init_model_with_self_attention_blend(pipe, prompt, guidance_func, masks: dict[int,torch.Tensor], 
                                         self_attention_stores=None, folder_path=None, is_vis=False, 
                                         step_vis=10, guidance_step_interval=None):
    """Initialize model with both cross-attention guidance and self-attention blending."""
    from models.replacement import SelfAttentionBlend
    
    attn_procs = {}
    tokens = pipe.tokenizer.tokenize(prompt)
    
    # Create smooth blend masks for self-attention
    smooth_masks = create_smooth_blend_masks(masks) if self_attention_stores else {}
    
    for name in pipe.unet.attn_processors.keys():
        if "attn2" in name:  # Cross-attention
            vis_folder = os.path.join(folder_path, "visualizations") if folder_path else None
            attn_procs[name] = my_attn_processor2(
                guidance_func,
                masks,
                len(tokens),
                place_in_unet=name,
                is_vis=is_vis,
                step_vis=step_vis,
                guidance_step_interval=guidance_step_interval,
                folder_path=vis_folder,
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer
            )
        elif "attn1" in name and self_attention_stores and name in self_attention_stores:
            # Self-attention with blending
            attn_procs[name] = SelfAttentionBlend(
                stored_attentions=self_attention_stores[name].stored_attentions,
                blend_mask=smooth_masks,
                place_in_unet=name,
                start_blending_step=30
            )
        else:
            attn_procs[name] = pipe.unet.attn_processors[name]
    
    pipe.unet.set_attn_processor(attn_procs)

# Removed old pipeline wrapper - using manual diffusion loop instead

def init_model(pipe,prompt,guidance_func,masks: dict[int,torch.Tensor], folder_path=None, is_vis=False, step_vis=10, guidance_step_interval=None):
    """Original init_model for backward compatibility."""
    attn_procs={}
    tokens=pipe.tokenizer.tokenize(prompt)
    for name in pipe.unet.attn_processors.keys():
        if "attn2" in name:
            vis_folder = os.path.join(folder_path, "visualizations") if folder_path else None
            attn_procs[name]=my_attn_processor2(
                guidance_func,
                masks,
                len(tokens),
                place_in_unet=name,
                is_vis=is_vis,
                step_vis=step_vis,
                guidance_step_interval=guidance_step_interval,
                folder_path=vis_folder,
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer
            )
        else:
            attn_procs[name]=pipe.unet.attn_processors[name]
    pipe.unet.set_attn_processor(attn_procs)


parser = argparse.ArgumentParser()
parser.add_argument('--num_inference_steps', type=int, default=50, help='The number of inference steps for pipe')
parser.add_argument('--height', type=int, default=512, help='The height for pipe')
parser.add_argument('--width', type=int, default=512, help='The width for pipe')

# parser.add_argument('--move_factor', type=float, default=2, help='The proportion of repulsive movement')
# parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for soft thresholding attention maps')
# parser.add_argument('--sharpness', type=float, default=1, help='Sharpness parameter for soft thresholding')
# parser.add_argument('--region_exclusion', type=float, default=0.75, help='Region exclusion strength')
# parser.add_argument('--theta', type=float, default=0.25, help='Conflict detection threshold')
# parser.add_argument('--repulsive_force', type=float, default=14)
# parser.add_argument('--margin_force', type=float, default=0.4)
parser.add_argument('--path', type=str, default="./")


parser.add_argument('--prompt', type=str, default="A person wearing a white cotton t-shirt and blue jeans")
parser.add_argument('--seed', type=int, default=-1, help='Seed for generation (-1 for random)')

parser.add_argument('--1', type=int, default=56775)
parser.add_argument('--box', nargs=4, type=int, metavar=('PX0','PY0','PX1','PY1'), help="512×512 px user box coords")
parser.add_argument('--add_image', type=str, default=None, help="path to drawn add region image (alternative to --box)")
parser.add_argument('--add_prompt',    type=str, default=None, help="text to ADD inside the box/region")
parser.add_argument('--scribble_path', type=str, default=None, help="path to scribble image")
parser.add_argument('--scribble_prompt', type=str, default=None, help="what the scribble should become")
parser.add_argument('--latent_path', type=str, default=None, help="path to saved latent tensor (.pt file) to skip initial generation")


args = parser.parse_args()
# model_id = os.path.expanduser(
#     "~/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5"
#     "/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
# )
model_id = "runwayml/stable-diffusion-v1-5"
device="cuda"

# Use DiffusionPipeline for the anime model
pipe = DiffusionPipeline.from_pretrained(model_id, 
    torch_dtype=torch.float16
).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

# guidance_func = partial(replace_attn_with_move_object_against_single_point,f_repl=args.repulsive_force,f_margin=args.margin_force,clamp=args.move_factor,threshold=args.threshold,sharpness=args.sharpness,region_exclusion=args.region_exclusion,theta=args.theta)

# Generate random seed if -1 is provided
if args.seed == -1:
    import random
    seed = random.randint(0, 2**32 - 1)
    print(f"Generated random seed: {seed}")
else:
    seed = args.seed

generator = torch.manual_seed(seed)
negative_prompt = "blurry, low quality, bad anatomy, bad hands, missing fingers, extra digits, cropped, worst quality, low resolution, text, watermark"


input_params = {
    "prompt": args.prompt,
    "seed": args.seed,
    "box": args.box,
    "add_image": args.add_image,
    "add_prompt": args.add_prompt,
    "scribble_path": args.scribble_path,
    "scribble_prompt": args.scribble_prompt,
    "latent_path": args.latent_path
}
prompt = args.prompt
# init_model(pipe,prompt,guidance_func,region)
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
outdir=extract_substring(args.path)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
folder_path = ('{}/output/{}/{}_{}').format(current_directory, outdir,prompt[:50], current_time)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# — Phase 1: Load or Generate latent + original image —

if args.latent_path:
    # Load existing latent from file
    print(f"Loading latent from {args.latent_path}...")
    init_latent = torch.load(args.latent_path).to(pipe.device)
    print(f"Loaded latent shape: {init_latent.shape}")
    
    # Decode the loaded latent to get the original image for reference
    arr = pipe.decode_latents(init_latent)[0]
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    img_np = (arr * 255.0).round().clip(0,255).astype("uint8")
    orig_img = Image.fromarray(img_np)
    
    # Save the decoded image for reference
    os.makedirs(folder_path, exist_ok=True)
    save_images(
        [orig_img],
        folder=folder_path,
        input_params=input_params,
        titles=["loaded_original"],
        left_top=None,
        right_bottom=None,
        save_with_draw_frame=False,
        save_combined=False
    )
    print(f"Saved loaded original image to {folder_path}/loaded_original.png")
    
else:
    # Generate new latent from prompt
    # 1a) Skip self-attention storage for now (memory intensive)
    self_attention_stores = None
    # if args.add_prompt or args.remove_prompt:
    #     print("Installing self-attention storage for original generation...")
    #     self_attention_stores = init_model_store_self_attention(pipe, prompt)

    # 1b) Run first pass to get latents (no attention hooks yet)  
    print("Generating original image...")
    output: StableDiffusionPipelineOutput = pipe(
        prompt,
        num_inference_steps=args.num_inference_steps,
        generator=torch.manual_seed(args.seed),
        negative_prompt=negative_prompt,
        output_type="latent",
        return_dict=False,
    )
    init_latent = output[0]  # shape [1, C, H, W]
    init_latent = init_latent.detach()

    # 2) Save the latent to disk for the edit phase
    os.makedirs(folder_path, exist_ok=True)
    torch.save(init_latent, os.path.join(folder_path, "init_latent.pt"))

    # 3) Decode & save the "original" reference image
    arr = pipe.decode_latents(init_latent)[0]
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    # the pipeline returns floats in [0,1], so scale back to 0–255:
    img_np = (arr * 255.0).round().clip(0,255).astype("uint8")
    orig_img = Image.fromarray(img_np)
    save_images(
        [orig_img],
        folder=folder_path,
        input_params=input_params,
        titles=["original"],
        left_top=None,
        right_bottom=None,
        save_with_draw_frame=False,
        save_combined=False
    )
    print(f"Saved original image to {folder_path}/original.png")

# Debug: Report self-attention storage (disabled for now)
# if self_attention_stores:
#     total_stored = sum(len(store.stored_attentions) for store in self_attention_stores.values())
#     print(f"Stored {total_stored} self-attention maps across {len(self_attention_stores)} layers")

if (args.box or args.add_image) and args.add_prompt:
    # Handle both box coordinates and drawn images
    if args.box:
        # Original box-based approach
        px0, py0, px1, py1 = args.box
        print(f"Using box coordinates: ({px0}, {py0}) to ({px1}, {py1})")

        # A) convert 512px→64px grid:
        def px_to_grid(px, img=512, grid=64): return int(px * grid / img)
        base_frame = Frame(
            px_to_grid(px0), px_to_grid(py0),
            px_to_grid(px1), px_to_grid(py1),
            deep_h=64
        )

        # B) build masks for each U-Net cross-attn scale:
        scales = [64, 32, 16, 8]
        masks = {}
        for s in scales:
            sf = s / base_frame.deep_h
            f = Frame(
                int(base_frame.x * sf),
                int(base_frame.y * sf),
                int(base_frame.a * sf),
                int(base_frame.b * sf),
                deep_h=s
            )
            # INVERT the mask - fill_tensor seems to return 0 inside, 1 outside
            masks[s] = 1.0 - fill_tensor(f.x, f.y, f.a, f.b, s, s).to("cuda")
    
    elif args.add_image:
        # New drawn image approach
        print(f"Using drawn add image: {args.add_image}")
        masks = create_masks_from_add_image(args.add_image)
        # No box coordinates for save_images in this case
        px0 = py0 = px1 = py1 = None
    
    for s, m in masks.items():
        print(f"scale={s}, mask cover fraction={m.float().mean().item():.3f}")

    
    # 4c) Reload the latent (DEBUG: skip file I/O to test if that's the issue)
    # init_latent = torch.load(os.path.join(folder_path,"init_latent.pt"))
    # Use the original latent directly without save/load

    # Token analysis  
    print(f"\n=== TOKEN ANALYSIS ===")
    print(f"Original prompt: '{prompt}'")
    if args.add_prompt:
        print(f"Add prompt: '{args.add_prompt}'")
        add_tokens = pipe.tokenizer.tokenize(args.add_prompt)
        print(f"Add tokens (tokenizer): {add_tokens}")
    print("=== END TOKEN ANALYSIS ===\n")

    
    # ADD functionality starts here (if args.add_prompt):
    if args.add_prompt:
        def simple_regional_guidance(attn, region, all_tokens, current_step=0):
            """
            Simple approach: boost butterfly attention in the edit region
            No temporal logic, no blending - just gentle regional focus
            """
            # Use global denoising tracker for debug prints
            current_step = denoising_tracker.get_step()
            
            
            # attn: [B, HW, C] where C is number of butterfly tokens
            out = attn.clone()
            
            # Get attention shape
            if len(out.shape) == 3:
                B, HW, C = out.shape
            elif len(out.shape) == 4:
                B, heads, HW, C = out.shape
            else:
                return out, []
            
            # Strong regional restriction: huge boost inside, strong suppression outside
            region_mask = region.flatten()[:HW]
            boost_strength = 3.0  # Original boost value
            suppression_strength = 0.5  # Original suppression value
            
            # Apply strong regional restriction to ALL tokens
            outside_region = 1.0 - region_mask
            for idx in range(C):
                if len(out.shape) == 3:
                    # Huge boost inside region, strong suppression outside
                    out[:, :, idx] = (out[:, :, idx] * boost_strength * region_mask.unsqueeze(0) + 
                                    out[:, :, idx] * suppression_strength * outside_region.unsqueeze(0))
                else:
                    # For 4D tensor
                    region_expanded = region_mask.unsqueeze(0).unsqueeze(0).expand(B, heads, HW)
                    outside_expanded = outside_region.unsqueeze(0).unsqueeze(0).expand(B, heads, HW)
                    out[:, :, :, idx] = (out[:, :, :, idx] * boost_strength * region_expanded + 
                                    out[:, :, :, idx] * suppression_strength * outside_expanded)
            
            return out, []  # Return tuple for my_attn_processor2

        # ADD functionality: only run if add_prompt is specified
        print(f"ADD APPROACH: Using add_prompt='{args.add_prompt}' with saved init_latent")
        
        # Load init_latent (final apple image)
        latents = init_latent.clone()
        
        # Initialize scheduler
        pipe.scheduler.set_timesteps(args.num_inference_steps, device=pipe.device)
        timesteps = pipe.scheduler.timesteps
        
        # Install attention processors with simple regional guidance for ADD PROMPT ONLY
        init_model(pipe, args.add_prompt, simple_regional_guidance, masks, folder_path=folder_path)
        
        # Encode ONLY the add prompt as you requested
        add_input = pipe.tokenizer(
            [args.add_prompt],
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(pipe.device)
        
        # DEBUG: Show exact token positions for visualization
        add_tokens = pipe.tokenizer.convert_ids_to_tokens(add_input.input_ids[0])
        print("=== ADD PROMPT TOKEN POSITIONS ===")
        for i, token in enumerate(add_tokens[:10]):  # Show first 10 tokens
            print(f"Token {i}: '{token}'")
        print("=== END TOKEN POSITIONS ===")
        
        with torch.no_grad():
            add_embeddings = pipe.text_encoder(add_input.input_ids)[0]
            
        # Encode negative prompt  
        uncond_input = pipe.tokenizer(
            [negative_prompt],
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(pipe.device)
        
        with torch.no_grad():
            uncond_embeddings = pipe.text_encoder(uncond_input.input_ids)[0]
        
        text_embeddings = torch.cat([uncond_embeddings, add_embeddings])
        guidance_scale = 7.5  # Standard CFG for SD1.5
        
        # Initialize step tracker (scheduler already set above)
        denoising_tracker.reset(args.num_inference_steps)
        
        # Run full denoising with butterfly prompt and regional guidance
        print(f"Running {args.num_inference_steps} steps with guidance...")
        
        for i, t in enumerate(timesteps):
            denoising_tracker.increment()
            current_step = denoising_tracker.get_step()
            
            # Steps 1-25: Keep latent unchanged (preserve apple)
            if current_step <= 25:
                if current_step == 25:
                    print("Step 25: Adding noise to preserved latent")
                    generator = torch.Generator(device=pipe.device)
                    generator.manual_seed(args.seed)
                    noise = torch.randn(latents.shape, generator=generator, device=pipe.device, dtype=torch.float16)
                    # Add less noise to preserve apple structure better
                    # Use full noise for clear butterfly generation
                    latents = pipe.scheduler.add_noise(latents, noise, t)
                # Skip denoising for steps 1-24, start denoising at step 25
                continue
            
            # Steps 25-50: Denoise with butterfly prompt + guidance
            if current_step % 5 == 0:
                print(f"Injection step {current_step}/50")
            
            # Standard diffusion step with butterfly embeddings
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            
            
            with torch.no_grad():
                noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
        
        edited_latent = latents.detach()
        print("Simple approach complete!")
        
        # Decode the edited latent to get the final image
        arr = pipe.decode_latents(edited_latent)[0]
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        img_np = (arr * 255.0).round().clip(0,255).astype("uint8")
        edited = Image.fromarray(img_np)


        # 4g) Save edited image with box overlay (if using box coordinates)
        if px0 is not None:  # Box coordinates available
            save_images([edited], folder=folder_path, input_params=input_params, titles=['edited'],
                left_top=(px0, py0), right_bottom=(px1, py1),
                save_with_draw_frame=True)
        else:  # Using drawn image, no box overlay
            save_images([edited], folder=folder_path, input_params=input_params, titles=['edited'],
                save_with_draw_frame=False)
        print(f"Saved edited image to {folder_path}/edited.png")

# SCRIBBLE functionality starts here (moved outside of box/add block)
if args.scribble_path and args.scribble_prompt:
    print(f"\nSCRIBBLE EDIT: Loading scribble from '{args.scribble_path}'")
    print(f"Scribble prompt: '{args.scribble_prompt}'")
    
    # Import ControlNet and scribble function
    from diffusers import ControlNetModel
    from scribble_edit import scribble_edit_with_attention_mask
    print("Using automatic attention mask generation with hybrid masking")
    
    # Load ControlNet model for scribble (v1.1 - improved version)
    print("Loading ControlNet scribble model v1.1...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_scribble",
        torch_dtype=torch.float16
    ).to(pipe.device)
    
    # Load scribble image
    scribble_image = Image.open(args.scribble_path)
    print(f"Loaded scribble image: {scribble_image.size}, mode: {scribble_image.mode}")
    
    # Debug: Check what we actually loaded
    scribble_array_test = np.array(scribble_image)
    print(f"Raw image stats: shape={scribble_array_test.shape}, min={scribble_array_test.min()}, max={scribble_array_test.max()}")
    if scribble_image.mode == 'RGBA':
        print("Image has alpha channel - converting to RGB with white background")
        # Create white background
        white_bg = Image.new('RGB', scribble_image.size, (255, 255, 255))
        # Paste image using alpha channel as mask
        white_bg.paste(scribble_image, mask=scribble_image.split()[3])  # 3 is the alpha channel
        scribble_image = white_bg
    
    # Run scribble edit using hybrid attention mask approach
    edited_latent = scribble_edit_with_attention_mask(
        pipe=pipe,
        controlnet=controlnet,
        init_latent=init_latent,
        scribble_image=scribble_image,
        scribble_prompt=args.scribble_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        noise_injection_step=1,  # Start ControlNet from beginning for full ribbon generation
        controlnet_conditioning_scale=2.5,  # Higher scale to force ControlNet to follow constraint more strictly
        mask_threshold=0.5,  # Threshold for attention mask
        debug_folder=folder_path
    )
    
    # Decode the edited latent - ensure dtype consistency
    edited_latent = edited_latent.to(pipe.vae.dtype)
    arr = pipe.decode_latents(edited_latent)[0]
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    img_np = (arr * 255.0).round().clip(0,255).astype("uint8")
    scribble_edited = Image.fromarray(img_np)
    
    # Save scribble-edited image
    save_images([scribble_edited], folder=folder_path, input_params=input_params, 
                titles=['scribble_edited'], save_with_draw_frame=False)
    print(f"Saved scribble-edited image to {folder_path}/scribble_edited.png")
    
    # Also save the scribble input for reference
    save_images([scribble_image], folder=folder_path, input_params=input_params, 
                titles=['scribble_input'], save_with_draw_frame=False)
    print(f"Saved scribble input to {folder_path}/scribble_input.png")
    
# End of functionality
if not args.add_prompt and not args.scribble_path:
    print("No add prompt or scribble specified!")
sys.exit(0)