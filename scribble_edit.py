"""
Scribble editing with ControlNet and InstDiffEdit automatic attention mask generation
"""

import torch
import numpy as np
from PIL import Image
from diffusers import ControlNetModel
import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import instant attention mask components
from instant_attention_mask import (
    AttentionStore, 
    register_attention_control,
    get_mask, 
    use_mask
)

# Define DenoisingStepTracker locally
class DenoisingStepTracker:
    """Global tracker for denoising steps during Stable Diffusion inference."""
    def __init__(self):
        self.current_step = 0
        self.total_steps = 0
        
    def reset(self, total_steps):
        """Reset tracker for new generation."""
        self.current_step = 0
        self.total_steps = total_steps
        
    def increment(self):
        """Increment step counter once per denoising timestep."""
        self.current_step += 1
        
    def get_step(self):
        """Get current denoising step."""
        return self.current_step

# Global denoising tracker
denoising_tracker = DenoisingStepTracker()


def create_controlnet_constraint_mask(scribble_image):
    """
    Create ControlNet constraint mask using edge detection + morphological operations
    (similar to the original scribble_edit.py approach).
    """
    import cv2
    from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes, gaussian_filter
    
    # Convert to grayscale for edge detection
    scribble_array = np.array(scribble_image.convert('L'))
    
    # Apply edge detection
    edges = cv2.Canny(scribble_array, 10, 50)
    
    if edges.sum() == 0:
        print("Canny didn't detect edges. Trying threshold-based detection...")
        _, edges = cv2.threshold(255 - scribble_array, 10, 255, cv2.THRESH_BINARY)
    
    # Convert to float mask
    control_mask_2d = (edges > 0).astype(np.float32)
    
    # Control mask: MORE generous dilation with less erosion to capture full ribbon area
    dilation_size = 20  # Much more generous to capture full generation area
    erosion_size = 12   # Less erosion to keep mask larger
    control_mask_2d = binary_dilation(control_mask_2d, iterations=dilation_size).astype(np.float32)
    control_mask_2d = binary_erosion(control_mask_2d, iterations=erosion_size).astype(np.float32)
    control_mask_2d = binary_fill_holes(control_mask_2d).astype(np.float32)
    
    # Minimal smoothing to preserve sharpness
    control_mask_2d = gaussian_filter(control_mask_2d, sigma=0.3)
    
    # Convert to tensor format for ControlNet: [1, 1, 512, 512]
    control_mask = torch.from_numpy(control_mask_2d).to(torch.float16)
    control_mask = control_mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    
    print(f"ControlNet constraint mask coverage: {control_mask.mean().item():.2%} of image")
    
    return control_mask


def apply_iam_blending(generated_latent, iam_mask, original_latent):
    """
    Apply IAM-based blending to combine generated and original latents.
    
    Args:
        generated_latent: Latent with ControlNet-generated content
        iam_mask: InstDiffEdit attention mask (PIL Image)
        original_latent: Original preserved latent
    """
    # Resize IAM mask to match latent dimensions
    from torchvision import transforms as tfms
    
    # Convert PIL mask to tensor and resize to latent dimensions
    mask_tensor = tfms.ToTensor()(iam_mask.resize((generated_latent.shape[-2], generated_latent.shape[-1])))
    
    # FINAL BOOST: Boost all mask values to 1.0 as the very last step before applying
    # This keeps the same mask boundaries but maximizes strength
    mask_tensor = torch.where(mask_tensor > 0.1, 1.0, mask_tensor)
    
    # Expand mask to match latent channels and batch dimensions
    mask_tensor = mask_tensor.expand(generated_latent.shape[1], -1, -1).unsqueeze(0)  # [1, C, H, W]
    
    # Move to same device and ensure correct dtype
    device = generated_latent.device
    mask_tensor = mask_tensor.float().to(device)
    original_latent = original_latent.to(device)
    
    # Blend: use generated content where IAM indicates attention, original elsewhere
    blended_latent = original_latent + mask_tensor * (generated_latent - original_latent)
    
    return blended_latent


def combine_constraint_and_iam_masks(constraint_mask, iam_mask, max_distance=20, falloff_strength=2.0):
    """
    HYBRID APPROACH: Use constraint as guide but allow IAM to extend beyond it
    - Keep ALL IAM attention (it knows where ribbon effects should be)
    - Weight it higher near constraint area, lower but still present when further away
    - Never completely filter out IAM regions
    
    Args:
        constraint_mask: ControlNet constraint mask tensor [1, 1, H, W] 
        iam_mask: InstDiffEdit attention mask (PIL Image)
        max_distance: Maximum distance for weighting falloff (pixels)
        falloff_strength: How quickly strength falls off with distance
    """
    from torchvision import transforms as tfms
    import torch.nn.functional as F
    from scipy.ndimage import distance_transform_edt, label
    import numpy as np
    
    # Convert IAM to tensor and resize to match constraint mask
    iam_tensor = tfms.ToTensor()(iam_mask)  # [1, H, W]
    iam_tensor = F.interpolate(iam_tensor.unsqueeze(0), 
                              size=(constraint_mask.shape[2], constraint_mask.shape[3]), 
                              mode='bilinear', align_corners=False)  # [1, 1, H, W]
    
    # Convert to numpy for processing
    constraint_np = constraint_mask.squeeze().cpu().numpy()
    iam_np = iam_tensor.squeeze().cpu().numpy()
    
    # NEW HYBRID APPROACH: Allow IAM to extend beyond constraint but weight it by distance
    constraint_binary = (constraint_np > 0.3).astype(bool)
    
    if constraint_binary.sum() == 0:
        # If no constraint mask, just return IAM as-is
        return iam_mask
    
    # Calculate distance from constraint for weighting (not filtering!)
    distance_from_constraint = distance_transform_edt(~constraint_binary)
    
    # Create distance-based weights that never go to zero:
    # - 1.0 inside constraint area  
    # - Gradual falloff outside, but minimum 30% weight to preserve IAM
    distance_weights = np.ones_like(distance_from_constraint)
    
    # Outside constraint area, apply falloff but keep minimum weight
    outside_constraint = distance_from_constraint > 0
    if max_distance > 0:
        max_dist_clipped = np.minimum(distance_from_constraint, max_distance)
        falloff = 1.0 - (max_dist_clipped / max_distance) ** falloff_strength
        distance_weights[outside_constraint] = np.maximum(falloff[outside_constraint], 0.1)  # Min 10% weight
    
    # Combine: IAM attention weighted by distance from constraint (NO filtering!)
    combined_np = iam_np * distance_weights
    
    # Also ensure we always keep the constraint area at reasonable strength  
    combined_np = np.maximum(combined_np, constraint_np * 0.5)
    
    # Convert back to PIL Image
    combined_tensor = torch.from_numpy(combined_np).float()
    combined_pil = tfms.ToPILImage()(combined_tensor)
    
    print(f"Hybrid masking: {(combined_np > 0.1).sum()} pixels in final mask (allows IAM beyond constraint)")
    
    return combined_pil


def scribble_edit_with_attention_mask(
    pipe,
    controlnet,
    init_latent,
    scribble_image,
    scribble_prompt,
    negative_prompt="",
    num_inference_steps=50,
    seed=42,
    noise_injection_step=1,
    controlnet_conditioning_scale=1.5,
    mask_threshold=0.5,
    debug_folder=None
):
    """
    Edit image with scribble using ControlNet with hybrid masking approach:
    
    1. ControlNet Constraint: Uses edge detection on scribble to constrain WHERE ControlNet generates
    2. IAM Blending: Uses InstDiffEdit attention masks to precisely control WHAT gets blended
    
    This combines the best of both approaches for optimal results.
    """
    
    # Prepare text embeddings for scribble prompt
    text_input = pipe.tokenizer(
        [scribble_prompt],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).to(pipe.device)
    
    with torch.no_grad():
        text_embeddings = pipe.text_encoder(text_input.input_ids)[0]
    
    # Prepare uncond embeddings
    uncond_input = pipe.tokenizer(
        [negative_prompt],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).to(pipe.device)
    
    with torch.no_grad():
        uncond_embeddings = pipe.text_encoder(uncond_input.input_ids)[0]
    
    # For CFG
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
    # Prepare scribble control image
    control_image = scribble_image.resize((512, 512))
    
    # Convert to RGB if not already
    if control_image.mode != 'RGB':
        control_image = control_image.convert('RGB')
    
    # Convert to numpy array
    control_image = np.array(control_image)
    
    # Convert to tensor and normalize
    control_image = torch.from_numpy(control_image).float() / 255.0
    
    # Rearrange from HWC to BCHW format
    control_image = control_image.permute(2, 0, 1).unsqueeze(0)
    control_image = control_image.to(pipe.device, dtype=torch.float16)
    
    # Create ControlNet constraint mask from the scribble drawing
    print("Creating ControlNet constraint mask from scribble...")
    control_mask = create_controlnet_constraint_mask(scribble_image)
    control_mask = control_mask.to(pipe.device)
    
    # Save ControlNet constraint mask for debugging
    if debug_folder:
        import torchvision.transforms as T
        control_mask_pil = T.ToPILImage()(control_mask.squeeze().cpu())
        control_mask_pil.save(f"{debug_folder}/controlnet_constraint_mask.png")
        print(f"Saved ControlNet constraint mask to {debug_folder}/controlnet_constraint_mask.png")
    
    # Initialize AttentionStore for capturing attention maps
    print("Initializing attention tracking...")
    attention_controller = AttentionStore()
    register_attention_control(pipe, attention_controller)
    
    # Initialize scheduler
    pipe.scheduler.set_timesteps(num_inference_steps, device=pipe.device)
    timesteps = pipe.scheduler.timesteps
    
    # Start with preserved latent
    latents = init_latent.clone()
    
    # Generator for reproducible noise
    generator = torch.Generator(device=pipe.device)
    generator.manual_seed(seed)
    
    # Reset step tracker
    denoising_tracker.reset(num_inference_steps)
    
    guidance_scale = 12.0  # Higher CFG for anime models
    
    # Position tracking for mask refinement
    position = []
    
    # Store masks for visualization
    all_masks = []
    
    print(f"Running scribble edit with structure preservation + automatic attention mask generation...")
    print(f"Scribble prompt: '{scribble_prompt}'")
    
    # Generate noise for later use
    noise = torch.randn(latents.shape, generator=generator, device=pipe.device, dtype=torch.float16)
    
    for i, t in enumerate(timesteps):
        denoising_tracker.increment()
        current_step = denoising_tracker.get_step()
        
        # NO STRUCTURE PRESERVATION - Start ControlNet from step 1 for full ribbon generation
        if current_step == 1:
            print("Step 1: Adding noise and starting ControlNet immediately")
            generator = torch.Generator(device=pipe.device)
            generator.manual_seed(seed)
            noise = torch.randn(latents.shape, generator=generator, device=pipe.device, dtype=torch.float16)
            latents = pipe.scheduler.add_noise(latents, noise, t)
        
        # Apply ControlNet influence for all steps
        if current_step % 5 == 0:
            print(f"ControlNet step {current_step}/{num_inference_steps}")
        
        # Store original latent with noise for this timestep
        or_latents = init_latent.clone()
        or_latents = pipe.scheduler.add_noise(or_latents, noise, t)
        
        # Expand latents for classifier-free guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        
        # Apply ControlNet constraint mask MORE aggressively - completely zero out areas outside mask
        expanded_control_mask = control_mask.expand_as(control_image)
        masked_control_image = torch.where(expanded_control_mask > 0.5, control_image, torch.zeros_like(control_image))
        
        # Get ControlNet conditioning - ensure dtype consistency
        down_block_res_samples, mid_block_res_sample = controlnet(
            latent_model_input.to(controlnet.dtype),
            t,
            encoder_hidden_states=text_embeddings.to(controlnet.dtype),
            controlnet_cond=masked_control_image.to(controlnet.dtype),
            conditioning_scale=controlnet_conditioning_scale,
            return_dict=False,
        )
        
        # Apply to UNet with ControlNet guidance - ensure dtype consistency
        with torch.no_grad():
            noise_pred = pipe.unet(
                latent_model_input.to(pipe.unet.dtype),
                t,
                encoder_hidden_states=text_embeddings.to(pipe.unet.dtype),
                down_block_additional_residuals=[d.to(pipe.unet.dtype) for d in down_block_res_samples],
                mid_block_additional_residual=mid_block_res_sample.to(pipe.unet.dtype),
            ).sample
        
        # Perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Scheduler step
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Generate attention mask for this step
        if current_step > 5:  # Let attention maps stabilize first few steps
            mask_diff, mask_image, position = get_mask(
                position=position,
                batch_size=1,
                attention_store=attention_controller,
                prompt=scribble_prompt,
                tokenizer=pipe.tokenizer,
                th=mask_threshold,
                res=16,  # Resolution for attention aggregation
                from_where=("up", "down")
            )
            
            # Combine constraint mask with IAM - tighten to prevent artifacts
            combined_mask = combine_constraint_and_iam_masks(
                control_mask, mask_image, 
                max_distance=25,  # Much tighter distance limit
                falloff_strength=2.0  # Sharper falloff
            )
            
            # Use combined mask for precise blending
            latents = apply_iam_blending(latents, combined_mask, or_latents)
            
            # Save combined mask for debugging (every 10 steps)
            if debug_folder and current_step % 10 == 0:
                combined_mask.save(f"{debug_folder}/combined_mask_step_{current_step}.png")
                print(f"Saved combined mask for step {current_step}")
            
            all_masks.append((mask_diff, mask_image, combined_mask))
    
    # Save final masks
    if debug_folder and len(all_masks) > 0:
        final_mask_diff, final_mask_binary, final_combined_mask = all_masks[-1]
        # Save final binary mask (IAM)
        final_mask_binary.save(f"{debug_folder}/final_IAM_binary.png")
        # Save final continuous mask (attention heatmap)
        final_mask_diff.save(f"{debug_folder}/final_IAM_heatmap.png")
        # Save final combined mask
        final_combined_mask.save(f"{debug_folder}/final_combined_mask.png")
        
        print(f"Saved final masks to {debug_folder}/")
        print(f"  - final_IAM_binary.png: Raw attention mask")
        print(f"  - final_IAM_heatmap.png: Attention heatmap")
        print(f"  - final_combined_mask.png: Final mask after filtering + weighting")
        
        # Skip evolution grid visualization to reduce debug clutter
    
    print("Scribble edit with attention mask complete!")
    
    return latents


# Optional: Function to run final inpainting pass for edge cleanup
def cleanup_with_inpainting(pipe, original_image, edited_latent, final_mask, prompt, seed=42):
    """
    Run a final inpainting pass to clean up edges using the final attention mask.
    This is optional but can improve edge quality.
    """
    from diffusers import StableDiffusionInpaintPipeline
    
    # Decode edited latent to image
    with torch.no_grad():
        edited_image = pipe.vae.decode(edited_latent / pipe.vae.config.scaling_factor).sample
    edited_image = (edited_image / 2 + 0.5).clamp(0, 1)
    edited_image = edited_image.cpu().permute(0, 2, 3, 1).numpy()[0]
    edited_image = Image.fromarray((edited_image * 255).round().astype("uint8"))
    
    # Initialize inpainting pipeline if not already available
    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16
    ).to(pipe.device)
    
    # Run inpainting
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    result = inpaint_pipe(
        prompt=prompt,
        image=original_image,
        mask_image=final_mask.resize((512, 512)),
        generator=generator,
        num_inference_steps=30
    ).images[0]
    
    return result