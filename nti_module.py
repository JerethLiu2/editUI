from __future__ import annotations
import time
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler


__all__ = [
    "load_sd15_pipeline",
    "reset_attention_processors",
    "get_execution_device",
    "process_nti",
]


def load_sd15_pipeline(model_id: str = "runwayml/stable-diffusion-v1-5", torch_dtype: torch.dtype = torch.float16) -> DiffusionPipeline:
    """
    Load a Stable Diffusion v1.5 pipeline with settings that are friendly to NTI and low VRAM.
    """
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, use_safetensors=True)
    # Inversion-friendly scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True
    )
    # Disable safety checker / feature extractor for speed
    try:
        pipe.safety_checker = None
    except Exception:
        pass
    try:
        pipe.feature_extractor = None
    except Exception:
        pass

    # Memory helpers
    pipe.enable_attention_slicing("max")
    pipe.enable_vae_slicing()
    try:
        pipe.unet.enable_gradient_checkpointing()
    except Exception:
        pass
    try:
        pipe.enable_sdpa()
    except Exception:
        pass
    try:
        pipe.enable_model_cpu_offload()
    except Exception:
        # Fallback: do nothing
        pass
    return pipe


def reset_attention_processors(pipe: DiffusionPipeline) -> None:
    """
    Reset UNet attention processors to default (AttnProcessor2_0).
    Some custom attention processors can destabilize NTI; we restore them after NTI completes.
    """
    from diffusers.models.attention_processor import AttnProcessor2_0
    default_attn_proc = AttnProcessor2_0()
    attn_procs = {}
    for name in pipe.unet.attn_processors.keys():
        attn_procs[name] = default_attn_proc
    pipe.unet.set_attn_processor(attn_procs)


def get_execution_device(pipe: DiffusionPipeline) -> torch.device:
    """
    Retrieve the device where the UNet executes (supports CPU offload).
    """
    return getattr(pipe, "_execution_device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))


@torch.no_grad()
def _vae_encode(pipe: DiffusionPipeline, image: Image.Image, device: torch.device) -> torch.Tensor:
    """
    VAE-encode a PIL image -> latent in the correct dtype & scaling.
    """
    image_tensor = pipe.image_processor.preprocess(image)
    image_tensor = image_tensor.to(device, dtype=pipe.vae.dtype)
    latent = pipe.vae.encode(image_tensor).latent_dist.sample()
    latent = latent * pipe.vae.config.scaling_factor
    return latent


def _encode_text_pair(
    pipe: DiffusionPipeline, prompt: str, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (cond_embeds, uncond_embeds) in UNet dtype on the given device.
    """
    cond, _ = pipe.encode_prompt(prompt, device=device, num_images_per_prompt=1, do_classifier_free_guidance=False)
    uncond, _ = pipe.encode_prompt("", device=device, num_images_per_prompt=1, do_classifier_free_guidance=False)
    return cond.to(device, dtype=pipe.unet.dtype), uncond.to(device, dtype=pipe.unet.dtype)


def process_nti(
    image: Image.Image,
    prompt: str,
    pipe: DiffusionPipeline,
    nti_seed: Optional[int] = None,
    max_iterations: int = 150,
    nti_timesteps: int = 10,
    backprop_cadence: int = 3,
    ema_decay: float = 0.97,
    lr: float = 5e-3,
    weight_decay: float = 1e-4,
    prior_weight: float = 1e-4,
    preview: bool = True,
    time_limit_sec: float = 300.0,
) -> Dict[str, Any]:
    """
    Null-Text Inversion (NTI) that optimizes an unconditional embedding u_star
    so that the forward+reverse diffusion reconstructs the given image faithfully.

    Returns:
        {
            "latent": FloatTensor (CPU, fp32),
            "u_star": FloatTensor (CPU, fp32),
            "metadata": {...},
            "preview": Optional[PIL.Image.Image]
        }
    """
    start_time = time.time()
    exec_device = get_execution_device(pipe)

    # Save & reset attention processors (for stability during NTI)
    original_attn_procs = pipe.unet.attn_processors.copy()
    reset_attention_processors(pipe)

    # Handle NTI seed: random default for diverse optimization basins
    if nti_seed is None:
        nti_seed = torch.randint(0, 2**31 - 1, (1,)).item()
        print(f"Generated random NTI seed: {nti_seed}")
    else:
        print(f"Using provided NTI seed: {nti_seed}")

    try:
        # Setup phase - can use inference_mode for VAE encoding and text embedding
        with torch.inference_mode():
            # Use image processor and move to execution device
            image_tensor = pipe.image_processor.preprocess(image)
            image_tensor = image_tensor.to(exec_device, dtype=pipe.vae.dtype)
            
            # VAE encode original image
            target_latent = pipe.vae.encode(image_tensor).latent_dist.sample()
            target_latent = target_latent * pipe.vae.config.scaling_factor
            
            # Prepare text embeddings
            prompt_embeds, _ = pipe.encode_prompt(
                prompt, device=exec_device, num_images_per_prompt=1, do_classifier_free_guidance=False
            )
            
            # Get default unconditional embedding
            uncond_embeds, _ = pipe.encode_prompt(
                "", device=exec_device, num_images_per_prompt=1, do_classifier_free_guidance=False
            )
        
        # Convert to normal tensors outside inference_mode for gradient computation
        text_embeddings = prompt_embeds.to(exec_device, dtype=pipe.unet.dtype).clone()
        uncond_embeddings = uncond_embeds.to(exec_device, dtype=pipe.unet.dtype).clone()
        target_latent = target_latent.clone()
        
        # Clone unconditional embedding as nn.Parameter in fp32 for numerical stability
        u_star = torch.nn.Parameter(uncond_embeddings.clone().detach().to(torch.float32))
        
        # EMA stabilization for smoother optimization
        u_star_ema = u_star.clone().detach()
        
        # Setup optimizer with refinements
        optimizer = torch.optim.AdamW([u_star], lr=lr, weight_decay=weight_decay)
    
        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iterations, eta_min=1e-5)
        
        # Setup grad-tail denoising schedule for NTI
        pipe.scheduler.set_timesteps(nti_timesteps, device=exec_device)
        timesteps = pipe.scheduler.timesteps
        
        # Grad-tail: positions -4, -2, -1 for better facial geometry capture
        grad_tail_positions = [-4, -2, -1]
        tail_indices = [len(timesteps) + pos for pos in grad_tail_positions if len(timesteps) + pos >= 0]
        
        # Early stopping
        best_loss = float('inf')
        best_u_star = u_star_ema.clone().detach()
        patience = 100
        no_improve_count = 0
        
        # Initialize cached values
        cached_loss_val = 0.0
        cached_mse_val = 0.0
        cached_prior_val = 0.0
        
        print(f"Starting NTI grad-tail optimization:")
        print(f"  - Total iterations: {max_iterations}")
        print(f"  - Timesteps: {nti_timesteps}")
        print(f"  - Backprop cadence: every {backprop_cadence} iterations")
        
        # NTI optimization loop with gradients enabled
        with torch.enable_grad():
            for iteration in range(max_iterations):
                optimizer.zero_grad()
                
                # Forward diffusion process
                working_latent = target_latent.clone().to(exec_device)
                
                # Add noise to start from a noisy state
                generator = torch.Generator(device=exec_device).manual_seed(nti_seed + iteration)
                noise = torch.randn(
                    working_latent.shape, 
                    generator=generator, 
                    device=exec_device, 
                    dtype=working_latent.dtype
                )
                init_timestep = timesteps[0]  # Start from most noisy
                working_latent = pipe.scheduler.add_noise(target_latent.to(exec_device), noise, init_timestep.to(exec_device))
                
                # Grad-tail denoising
                should_backprop = (iteration % backprop_cadence == 0 or iteration == max_iterations - 1)
                
                # Process all timesteps in order
                early_step_count = 0
                
                for i, timestep in enumerate(timesteps):
                    # Determine if this timestep should have gradients
                    is_grad_step = i in tail_indices and should_backprop
                    
                    # CFG schedule
                    if i in tail_indices:
                        tail_progress = list(tail_indices).index(i) / max(1, len(tail_indices) - 1)
                        cfg_scale = 3.0 + tail_progress * 1.0
                        guidance_rescale = 0.7 + tail_progress * 0.15
                    else:
                        total_early_steps = len(timesteps) - len(tail_indices)
                        early_progress = early_step_count / max(1, total_early_steps - 1)
                        cfg_scale = 0.0 + early_progress * 2.0
                        guidance_rescale = 0.8
                        early_step_count += 1
                    
                    # Enable gradients only at grad-tail positions during backprop iterations
                    context_manager = torch.enable_grad() if is_grad_step else torch.no_grad()
                    
                    with context_manager:
                        working_latent_input = working_latent.to(exec_device, dtype=pipe.unet.dtype)
                        timestep_input = timestep.to(exec_device)
                        working_latent_input = pipe.scheduler.scale_model_input(working_latent_input, timestep_input)
                        
                        # Choose embeddings: grad-enabled u_star for grad steps, EMA for others
                        if is_grad_step:
                            enc_uncond = u_star.to(exec_device, dtype=pipe.unet.dtype)
                            enc_cond = text_embeddings.to(exec_device, dtype=pipe.unet.dtype)
                        else:
                            enc_uncond = u_star_ema.to(exec_device, dtype=pipe.unet.dtype)
                            enc_cond = text_embeddings.detach().to(exec_device, dtype=pipe.unet.dtype)
                        
                        with torch.cuda.amp.autocast(enabled=True):
                            # Unconditional pass
                            noise_pred_uncond = pipe.unet(
                                working_latent_input, 
                                timestep_input, 
                                encoder_hidden_states=enc_uncond
                            ).sample
                            
                            # Conditional pass
                            noise_pred_text = pipe.unet(
                                working_latent_input, 
                                timestep_input, 
                                encoder_hidden_states=enc_cond
                            ).sample
                        
                        # CFG combination with guidance rescale
                        noise_pred = noise_pred_uncond + cfg_scale * guidance_rescale * (noise_pred_text - noise_pred_uncond)
                        
                        # Scheduler step
                        working_latent = pipe.scheduler.step(noise_pred, timestep, working_latent).prev_sample
                
                # Loss computation for backprop iterations
                if should_backprop:
                    latent_target = target_latent.to(exec_device, dtype=working_latent.dtype)
                    latent_reconstruction = working_latent.to(exec_device, dtype=working_latent.dtype)
                    
                    # Full-frame loss for NTI
                    mse_loss = F.mse_loss(latent_reconstruction, latent_target)
                    
                    # Prior loss to keep u_star close to original
                    prior_loss = prior_weight * F.mse_loss(
                        u_star.to(exec_device), 
                        uncond_embeddings.detach().to(exec_device, dtype=u_star.dtype)
                    )
                    
                    total_loss = mse_loss + prior_loss
                    
                    # Backward pass and optimization step
                    total_loss.backward()
                    optimizer.step()
                    scheduler.step()
                    
                    # Update EMA after optimizer step
                    with torch.no_grad():
                        u_star_ema.mul_(ema_decay).add_(u_star.data, alpha=1 - ema_decay)
                    
                    # Cache loss for monitoring
                    cached_loss_val = total_loss.item()
                    cached_mse_val = mse_loss.item()
                    cached_prior_val = prior_loss.item()
                else:
                    # Non-backprop iterations: minimal optimization
                    if cached_loss_val > 0:
                        prior_loss = 1e-5 * F.mse_loss(
                            u_star.to(exec_device), 
                            uncond_embeddings.detach().to(exec_device, dtype=u_star.dtype)
                        )
                        prior_loss.backward()
                        optimizer.step()
                        scheduler.step()
                        
                        with torch.no_grad():
                            u_star_ema.mul_(ema_decay).add_(u_star.data, alpha=1 - ema_decay)
                        
                        total_loss = torch.tensor(cached_loss_val)
                        mse_loss = torch.tensor(cached_mse_val)
                        prior_loss_val = prior_loss.item()
                    else:
                        optimizer.step()
                        scheduler.step()
                        total_loss = torch.tensor(1.0)
                        mse_loss = torch.tensor(1.0)
                        prior_loss_val = 0.0
                
                # Early stopping check
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_u_star = u_star_ema.clone().detach()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                if iteration % 20 == 0 or iteration == max_iterations - 1:
                    elapsed = time.time() - start_time
                    print(f"NTI Iter {iteration:3d}: Loss={total_loss.item():.6f} (MSE={mse_loss.item():.6f}), Time={elapsed:.1f}s")
                
                # Early stopping and timeout
                if no_improve_count >= patience:
                    print(f"Early stopping at iteration {iteration}")
                    break
                
                if time.time() - start_time > time_limit_sec:
                    print(f"Timeout reached at iteration {iteration}")
                    break
        
        print(f"NTI optimization completed in {time.time() - start_time:.1f}s")
        
        # Restore best u_star
        u_star_final = best_u_star
        latent_final = target_latent
        print(f"Restored best u_star with loss {best_loss:.6f}")

        # Optional preview decode
        preview_image = None
        if preview:
            with torch.inference_mode():
                latent_for_preview = latent_final.to(exec_device, dtype=pipe.vae.dtype)
                img = pipe.decode_latents(latent_for_preview)[0]
                img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
                img = (img * 255.0).round().clip(0, 255).astype("uint8")
                preview_image = Image.fromarray(img)

        metadata = {
            "nti_seed": nti_seed,
            "nti_timesteps": nti_timesteps,
            "max_iterations": max_iterations,
            "grad_tail_positions": grad_tail_positions,
            "scheduler": "DPMSolverMultistep++_Karras",
            "loss_best": float(best_loss),
            "elapsed_sec": float(time.time() - start_time),
            "iterations": iteration + 1,
        }

        # Return CPU fp32 copies for persistence
        return {
            "latent": latent_final.to(torch.float32).cpu(),
            "u_star": u_star_final.to(torch.float32).cpu(),
            "metadata": metadata,
            "preview": preview_image,
        }

    finally:
        # Restore original attention processors
        pipe.unet.set_attn_processor(original_attn_procs)