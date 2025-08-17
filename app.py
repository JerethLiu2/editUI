"""
Flask backend API for image generation and editing
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import base64
import io
import os
import json
import numpy as np
from PIL import Image
from datetime import datetime
import traceback
import threading
import uuid
from pathlib import Path

# Import the existing generation code
from diffusers import DiffusionPipeline, ControlNetModel, DPMSolverMultistepScheduler
from scribble_edit import scribble_edit_with_attention_mask
from models.replacement import my_attn_processor2
from utils.frame import Frame
from utils.ptp_utils import fill_tensor
from utils.vis_utils import save_images
from nti_module import process_nti

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for model and state
pipe = None
controlnet = None
current_latent = None
current_seed = 42
sessions = {}  # Store session data

# Create directories
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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

denoising_tracker = DenoisingStepTracker()

def init_model(pipe, prompt, guidance_func, masks, folder_path=None):
    """Initialize model with attention processors."""
    attn_procs = {}
    tokens = pipe.tokenizer.tokenize(prompt)
    for name in pipe.unet.attn_processors.keys():
        if "attn2" in name:
            attn_procs[name] = my_attn_processor2(
                guidance_func,
                masks,
                len(tokens),
                place_in_unet=name,
                is_vis=False,
                step_vis=10,
                guidance_step_interval=None,
                folder_path=None,
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer
            )
        else:
            attn_procs[name] = pipe.unet.attn_processors[name]
    pipe.unet.set_attn_processor(attn_procs)

def reset_attention_processors(pipe):
    """Reset UNet attention processors to default."""
    from diffusers.models.attention_processor import AttnProcessor2_0
    default_attn_proc = AttnProcessor2_0()
    attn_procs = {}
    for name in pipe.unet.attn_processors.keys():
        attn_procs[name] = default_attn_proc
    pipe.unet.set_attn_processor(attn_procs)

def get_execution_device():
    """Get the actual execution device for CPU-offloaded pipeline."""
    return getattr(pipe, "_execution_device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def scribble_edit_sdedit(
    pipe,
    controlnet,
    init_latent,
    scribble_image,
    scribble_prompt,
    negative_prompt="",
    num_inference_steps=50,
    seed=42,
    edit_strength=0.6,
    controlnet_conditioning_scale=2.5,
    guidance_scale=12.0,
    exec_device=None
):
    """
    SDEdit-based scribble editing with minimal drift and latent masking.
    """
    import cv2
    from scipy.ndimage import binary_dilation, binary_fill_holes
    import torch.nn.functional as F
    
    # Get execution device if not provided
    if exec_device is None:
        exec_device = get_execution_device()
    
    # Ensure init_latent is on execution device with correct dtype
    init_latent = init_latent.to(exec_device, dtype=torch.float16)
    
    # Prepare text embeddings
    text_input = pipe.tokenizer(
        [scribble_prompt],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).to(exec_device)
    
    with torch.no_grad():
        text_embeddings = pipe.text_encoder(text_input.input_ids)[0]
    
    # Prepare uncond embeddings
    uncond_input = pipe.tokenizer(
        [negative_prompt],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).to(exec_device)
    
    with torch.no_grad():
        uncond_embeddings = pipe.text_encoder(uncond_input.input_ids)[0]
    
    # For CFG
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
    # Generate random seed for varied edit results
    import random
    edit_seed = random.randint(0, 2**32 - 1)
    print(f"Using random scribble edit seed: {edit_seed}")
    
    # Prepare scribble control image (white background, black strokes, high contrast)
    control_image = scribble_image.resize((512, 512))
    
    # Convert to grayscale and create strong white-bg/black-lines contrast
    control_gray = np.array(control_image.convert('L'))
    
    # Invert and threshold to get white background with black strokes
    control_inverted = 255 - control_gray
    _, control_binary = cv2.threshold(control_inverted, 127, 255, cv2.THRESH_BINARY)
    
    # Optionally dilate slightly for stronger strokes
    kernel = np.ones((3, 3), np.uint8)
    control_binary = cv2.dilate(control_binary, kernel, iterations=1)
    
    # Convert to RGB for ControlNet
    control_rgb = cv2.cvtColor(control_binary, cv2.COLOR_GRAY2RGB)
    control_image = torch.from_numpy(control_rgb).float() / 255.0  # Normalize to [0,1]
    control_image = control_image.permute(2, 0, 1).unsqueeze(0)
    control_image = control_image.to(exec_device, dtype=torch.float16)
    
    # Create zero control image for uncond arm of CFG
    control_zeros = torch.zeros_like(control_image)
    
    print(f"Control image stats: min={control_image.min():.3f}, max={control_image.max():.3f}, unique_vals={torch.unique(control_image).numel()}")
    
    # Create latent-resolution mask from scribble using two-band approach
    from scipy.ndimage import binary_erosion, gaussian_filter
    
    scribble_array = np.array(scribble_image.convert('L'))
    edges = cv2.Canny(scribble_array, 10, 50)
    
    if edges.sum() == 0:
        _, edges = cv2.threshold(255 - scribble_array, 10, 255, cv2.THRESH_BINARY)
    
    # Initial mask from edges
    initial_mask = (edges > 0).astype(np.float32)
    initial_mask = binary_dilation(initial_mask, iterations=10).astype(np.float32)
    initial_mask = binary_fill_holes(initial_mask).astype(np.float32)
    
    # Convert to latent resolution first
    latent_h, latent_w = init_latent.shape[2], init_latent.shape[3]
    initial_mask_tensor = torch.from_numpy(initial_mask).to(exec_device)
    mask_latent = F.interpolate(
        initial_mask_tensor.unsqueeze(0).unsqueeze(0), 
        size=(latent_h, latent_w), 
        mode='nearest'  # Use nearest for binary mask
    ).squeeze()
    
    # Convert to numpy for morphological operations in latent space
    mask_latent_np = mask_latent.cpu().numpy()
    
    # Two-band mask creation in latent space
    # 1. Create core by eroding by 1 latent pixel
    from scipy.ndimage import binary_erosion as scipy_erosion
    core_mask = scipy_erosion(mask_latent_np > 0.5, iterations=1).astype(np.float32)
    
    # 2. Create feather ring by dilating core by +3 latent pixels
    feather_mask = binary_dilation(core_mask, iterations=3).astype(np.float32)
    
    # 3. Apply Gaussian blur to create soft transition
    # Kernel 7x7, sigma ~1.2 in latent units
    soft_mask = gaussian_filter(feather_mask, sigma=1.2)
    
    # 4. Convert back to tensor and ensure core stays at 100%
    edit_mask = torch.from_numpy(soft_mask).to(exec_device, dtype=torch.float32)
    core_mask_tensor = torch.from_numpy(core_mask).to(exec_device, dtype=torch.float32)
    
    # Ensure core region stays at 100% (take maximum of blurred mask and core)
    edit_mask = torch.maximum(edit_mask, core_mask_tensor)
    
    # Clamp to [0,1] range
    edit_mask = torch.clamp(edit_mask, 0.0, 1.0)
    
    # Log mask characteristics
    mask_coverage = edit_mask.mean().item()
    core_coverage = (edit_mask > 0.9).float().mean().item()
    transition_coverage = ((edit_mask > 0.1) & (edit_mask < 0.9)).float().mean().item()
    
    print(f"Two-band mask - Total coverage: {mask_coverage:.2%}, Core: {core_coverage:.2%}, Transition: {transition_coverage:.2%}")
    
    # SDEdit: compute starting timestep and inject noise once (INVERTED for proper mapping)
    pipe.scheduler.set_timesteps(num_inference_steps, device=exec_device)
    timesteps = pipe.scheduler.timesteps
    # Invert: strength 1.0 = start at beginning (index 0), strength 0.0 = start at end
    t_start_idx = int((1.0 - edit_strength) * (len(timesteps) - 1))
    t_start_idx = max(0, min(t_start_idx, len(timesteps) - 1))  # Clamp to [0, len-1]
    
    # Single noise injection with random seed for variety
    latents = init_latent.clone()
    if t_start_idx < len(timesteps):
        generator = torch.Generator(device=exec_device)
        generator.manual_seed(edit_seed)  # Use random edit seed for varied noise patterns
        noise = torch.randn(latents.shape, generator=generator, device=exec_device, dtype=torch.float16)
        # Use timesteps from start index forward
        timesteps = timesteps[t_start_idx:]
        # Use the first timestep from our sliced schedule
        t_start = timesteps[0]
        latents = pipe.scheduler.add_noise(latents, noise, t_start)
        print(f"SDEdit: injecting noise at timestep {t_start} (step {t_start_idx}/{len(pipe.scheduler.timesteps)}) with random noise seed {edit_seed}")
    
    # Set random seed globally for generation process (varied results)
    torch.manual_seed(edit_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(edit_seed)
    
    # Denoise with ControlNet and latent masking
    for i, t in enumerate(timesteps):
        # Optional: Linear schedule for ControlNet conditioning scale
        # Start at passed value, decrease by 25% over time
        progress = i / max(1, len(timesteps) - 1)
        current_control_scale = controlnet_conditioning_scale * (1.0 - 0.25 * progress)
        
        # Expand latents for CFG
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        
        # Get ControlNet conditioning with proper CFG batching (zeros for uncond, scribble for cond)
        control_batch = torch.cat([control_zeros, control_image])  # [uncond, cond]
        down_block_res_samples, mid_block_res_sample = controlnet(
            latent_model_input.to(controlnet.dtype),
            t,
            encoder_hidden_states=text_embeddings.to(controlnet.dtype),
            controlnet_cond=control_batch.to(controlnet.dtype),
            conditioning_scale=current_control_scale,  # Use scheduled scale
            return_dict=False,
        )
        
        # Apply to UNet with ControlNet guidance
        with torch.no_grad():
            noise_pred = pipe.unet(
                latent_model_input.to(pipe.unet.dtype),
                t,
                encoder_hidden_states=text_embeddings.to(pipe.unet.dtype),
                down_block_additional_residuals=[d.to(pipe.unet.dtype) for d in down_block_res_samples],
                mid_block_additional_residual=mid_block_res_sample.to(pipe.unet.dtype)
            ).sample
        
        # Perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Scheduler step
        step_latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Latent-space masking: preserve unchanged areas
        latents = edit_mask.unsqueeze(0).unsqueeze(0) * step_latents + (1.0 - edit_mask.unsqueeze(0).unsqueeze(0)) * init_latent.to(step_latents.device, step_latents.dtype)
    
    return latents

def load_models():
    """Load the diffusion models."""
    global pipe, controlnet
    
    if pipe is None:
        print("Loading diffusion model...")
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = DiffusionPipeline.from_pretrained(model_id, 
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        # Use NTI-friendly scheduler configuration
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True
        )
        pipe.safety_checker = None
        pipe.feature_extractor = None
        
        # Enable memory optimizations
        pipe.enable_attention_slicing("max")
        pipe.enable_vae_slicing()
        pipe.unet.enable_gradient_checkpointing()
        
        # Enable CPU offloading
        pipe.enable_model_cpu_offload()
        
        print("Model loaded successfully!")
    
    if controlnet is None:
        print("Loading ControlNet model...")
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_scribble",
            torch_dtype=torch.float16
        ).to("cuda")
        print("ControlNet loaded successfully!")

def image_to_base64(image):
    """Convert PIL image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def base64_to_image(base64_string):
    """Convert base64 string to PIL image."""
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "models_loaded": pipe is not None})

@app.route('/api/generate', methods=['POST'])
def generate_image():
    """Generate a new image from prompt."""
    global current_latent, current_seed
    
    try:
        load_models()
        
        # Reset attention processors to clean state before generation
        reset_attention_processors(pipe)
        
        data = request.json
        prompt = data.get('prompt', 'A person wearing a white cotton t-shirt and blue jeans')
        seed = data.get('seed', -1)
        num_inference_steps = data.get('num_inference_steps', 50)
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        # Generate random seed if -1 is provided
        if seed == -1:
            import random
            seed = random.randint(0, 2**32 - 1)
            print(f"Generated random seed: {seed}")
        
        current_seed = seed
        negative_prompt = "blurry, low quality, bad anatomy, bad hands, missing fingers, extra digits, cropped, worst quality, low resolution, text, watermark"
        
        print(f"Generating image with prompt: {prompt}, seed: {seed}")
        
        # Generate image
        generator = torch.manual_seed(seed)
        output = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            generator=generator,
            negative_prompt=negative_prompt,
            output_type="latent",
            return_dict=False,
        )
        
        latent = output[0].detach()
        
        # Store latent in session in FP32 on CPU to save VRAM
        if session_id not in sessions:
            sessions[session_id] = {}
        sessions[session_id]['latent'] = latent.to(torch.float32).cpu()
        sessions[session_id]['prompt'] = prompt
        sessions[session_id]['seed'] = seed
        
        # Decode latent to image - ensure no gradients
        with torch.no_grad():
            arr = pipe.decode_latents(latent.detach())[0]
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        img_np = (arr * 255.0).round().clip(0, 255).astype("uint8")
        image = Image.fromarray(img_np)
        
        # Convert to base64
        image_base64 = image_to_base64(image)
        
        return jsonify({
            "success": True,
            "image": image_base64,
            "session_id": session_id
        })
        
    except Exception as e:
        print(f"Error in generate_image: {str(e)}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

def create_masks_from_add_image(add_image):
    """Create masks for regional guidance from drawn add region image."""
    import cv2
    from scipy.ndimage import binary_fill_holes
    
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
        from PIL import Image as PILImage
        mask_pil = PILImage.fromarray((mask_2d * 255).astype(np.uint8))
        mask_resized = mask_pil.resize((s, s), PILImage.NEAREST)
        mask_tensor = torch.from_numpy(np.array(mask_resized) / 255.0).float().to("cuda")
        
        # Invert mask (0 where we want to add, 1 where we want to preserve)
        masks[s] = 1.0 - mask_tensor
    
    return masks

@app.route('/api/edit/add', methods=['POST'])
def edit_add():
    """Add an element to image using drawn region."""
    try:
        load_models()
        
        # Reset attention processors to clean state before add editing
        reset_attention_processors(pipe)
        
        data = request.json
        session_id = data.get('session_id')
        add_image_base64 = data.get('add_image')  # Changed from box to add_image
        add_prompt = data.get('add_prompt')
        num_inference_steps = data.get('num_inference_steps', 50)
        
        if session_id not in sessions or 'latent' not in sessions[session_id]:
            return jsonify({"success": False, "error": "No active session found"}), 400
        
        # Get execution device for CPU offloading compatibility
        exec_device = get_execution_device()
        
        # Move FP32 latent to execution device and convert to FP16 for inference
        init_latent = sessions[session_id]['latent'].to(exec_device, torch.float16)
        seed = sessions[session_id].get('seed', 42)
        
        # Convert base64 add image to PIL image
        add_image = base64_to_image(add_image_base64)
        
        print(f"Adding '{add_prompt}' to drawn region")
        
        # Create masks from drawn add region
        masks = create_masks_from_add_image(add_image)
        
        # Define regional guidance function
        def simple_regional_guidance(attn, region, all_tokens, current_step=0):
            current_step = denoising_tracker.get_step()
            out = attn.clone()
            
            if len(out.shape) == 3:
                B, HW, C = out.shape
            elif len(out.shape) == 4:
                B, heads, HW, C = out.shape
            else:
                return out, []
            
            region_mask = region.flatten()[:HW]
            boost_strength = 3.0  # Original boost value
            suppression_strength = 0.5  # Original suppression value
            
            outside_region = 1.0 - region_mask
            for idx in range(C):
                if len(out.shape) == 3:
                    out[:, :, idx] = (out[:, :, idx] * boost_strength * region_mask.unsqueeze(0) + 
                                    out[:, :, idx] * suppression_strength * outside_region.unsqueeze(0))
                else:
                    region_expanded = region_mask.unsqueeze(0).unsqueeze(0).expand(B, heads, HW)
                    outside_expanded = outside_region.unsqueeze(0).unsqueeze(0).expand(B, heads, HW)
                    out[:, :, :, idx] = (out[:, :, :, idx] * boost_strength * region_expanded + 
                                    out[:, :, :, idx] * suppression_strength * outside_expanded)
            
            return out, []
        
        # Generate random seed for varied edit results
        import random
        edit_seed = random.randint(0, 2**32 - 1)
        print(f"Using random edit seed: {edit_seed}")
        
        # Initialize attention processors
        init_model(pipe, add_prompt, simple_regional_guidance, masks)
        
        # Prepare embeddings
        negative_prompt = "blurry, low quality, bad anatomy, bad hands, missing fingers, extra digits, cropped, worst quality, low resolution, text, watermark"
        
        add_input = pipe.tokenizer(
            [add_prompt],
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(exec_device)
        
        with torch.no_grad():
            add_embeddings = pipe.text_encoder(add_input.input_ids)[0]
        
        uncond_input = pipe.tokenizer(
            [negative_prompt],
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(exec_device)
        
        with torch.no_grad():
            uncond_embeddings = pipe.text_encoder(uncond_input.input_ids)[0]
        
        text_embeddings = torch.cat([uncond_embeddings, add_embeddings])
        guidance_scale = 9.0  # Reduced CFG to minimize artifacts while maintaining effectiveness
        
        # SDEdit approach: increased strength for more pronounced add mode edits
        edit_strength = 0.8  # Higher strength for stronger add mode effects
        
        # Compute starting timestep for SDEdit (INVERTED for proper mapping)
        pipe.scheduler.set_timesteps(num_inference_steps, device=exec_device)
        timesteps = pipe.scheduler.timesteps
        # Invert: strength 1.0 = start at beginning (index 0), strength 0.0 = start at end
        t_start_idx = int((1.0 - edit_strength) * (len(timesteps) - 1))
        t_start_idx = max(0, min(t_start_idx, len(timesteps) - 1))  # Clamp to [0, len-1]
        timesteps = timesteps[t_start_idx:]
        
        # Create latent-resolution mask for blending
        latent_h, latent_w = init_latent.shape[2], init_latent.shape[3]
        mask_latent = torch.zeros((latent_h, latent_w), device=exec_device, dtype=torch.float32)
        
        # Convert drawn region masks to latent resolution
        for scale, region_mask in masks.items():
            if scale == 64:  # Use the highest resolution mask
                # Resize to latent resolution and smooth edges
                import torch.nn.functional as F
                mask_resized = F.interpolate(
                    region_mask.unsqueeze(0).unsqueeze(0), 
                    size=(latent_h, latent_w), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze()
                
                # Smooth the mask boundaries to avoid hard edges
                kernel_size = 5
                padding = kernel_size // 2
                smooth_kernel = torch.ones(1, 1, kernel_size, kernel_size, device=exec_device) / (kernel_size * kernel_size)
                mask_latent = F.conv2d(
                    mask_resized.unsqueeze(0).unsqueeze(0), 
                    smooth_kernel, 
                    padding=padding
                ).squeeze()
                break
        
        # Invert mask (1 = edit region, 0 = preserve region)
        edit_mask = 1.0 - mask_latent
        
        # Clamp mask to [0,1] range after smoothing
        edit_mask = torch.clamp(edit_mask, 0.0, 1.0)
        
        # Log mask coverage for debugging
        mask_coverage = edit_mask.mean().item()
        print(f"Edit mask coverage: {mask_coverage:.2%} of latent space")
        
        # Single noise injection at computed timestep (use random seed for varied noise patterns)
        latents = init_latent.clone()
        if len(timesteps) > 0:
            generator = torch.Generator(device=exec_device)
            generator.manual_seed(edit_seed)  # Use random edit seed for varied noise patterns
            noise = torch.randn(latents.shape, generator=generator, device=exec_device, dtype=torch.float16)
            # Use the first timestep from our sliced schedule
            t_start = timesteps[0]
            latents = pipe.scheduler.add_noise(latents, noise, t_start)
            print(f"SDEdit: injecting noise at timestep {t_start} (step {t_start_idx}/{len(pipe.scheduler.timesteps)}) with random noise seed {edit_seed}")
        
        denoising_tracker.reset(len(timesteps))
        
        # Set random seed globally for generation process (varied results)
        torch.manual_seed(edit_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(edit_seed)
        
        # Denoise only from the computed timestep to end
        for i, t in enumerate(timesteps):
            denoising_tracker.increment()
            
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            
            with torch.no_grad():
                noise_pred = pipe.unet(
                    latent_model_input.to(pipe.unet.dtype), 
                    t, 
                    encoder_hidden_states=text_embeddings.to(pipe.unet.dtype)
                ).sample
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            step_latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
            
            # Latent-space masking: blend edited regions with preserved regions
            latents = edit_mask.unsqueeze(0).unsqueeze(0) * step_latents + (1.0 - edit_mask.unsqueeze(0).unsqueeze(0)) * init_latent.to(step_latents.device, step_latents.dtype)
        
        # Store edited latent in FP32 on CPU for quality preservation and VRAM efficiency
        edited_latent = latents.detach().to(torch.float32).cpu()
        
        # Reset attention processors after editing to prevent conflicts
        reset_attention_processors(pipe)
        
        # Decode to image for preview (ensure proper dtype for VAE)
        with torch.no_grad():
            arr = pipe.decode_latents(edited_latent.to(exec_device).to(pipe.vae.dtype))[0]
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        img_np = (arr * 255.0).round().clip(0, 255).astype("uint8")
        edited_image = Image.fromarray(img_np)
        
        # Store edited latent in FP32 on CPU
        sessions[session_id]['edited_latent'] = edited_latent
        
        return jsonify({
            "success": True,
            "image": image_to_base64(edited_image)
        })
        
    except Exception as e:
        print(f"Error in edit_add: {str(e)}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/edit/scribble', methods=['POST'])
def edit_scribble():
    """Edit image using scribble."""
    try:
        load_models()
        
        # Reset attention processors to clean state before scribble editing
        reset_attention_processors(pipe)
        
        data = request.json
        session_id = data.get('session_id')
        scribble_image_base64 = data.get('scribble_image')
        scribble_prompt = data.get('scribble_prompt')
        num_inference_steps = data.get('num_inference_steps', 50)
        
        if session_id not in sessions or 'latent' not in sessions[session_id]:
            return jsonify({"success": False, "error": "No active session found"}), 400
        
        # Get execution device for CPU offloading compatibility
        exec_device = get_execution_device()
        
        # Move FP32 latent to execution device and convert to FP16 for inference
        init_latent = sessions[session_id]['latent'].to(exec_device, torch.float16)
        seed = sessions[session_id].get('seed', 42)
        
        # Convert base64 scribble to PIL image
        scribble_image = base64_to_image(scribble_image_base64)
        
        print(f"Applying scribble edit with prompt: {scribble_prompt}")
        
        negative_prompt = "blurry, low quality, bad anatomy, bad hands, missing fingers, extra digits, cropped, worst quality, low resolution, text, watermark"
        
        # Run scribble edit with SDEdit approach (consistent with add mode)
        edited_latent = scribble_edit_sdedit(
            pipe=pipe,
            controlnet=controlnet,
            init_latent=init_latent,
            scribble_image=scribble_image,
            scribble_prompt=scribble_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            seed=seed,
            edit_strength=0.4,  # Lower strength for better color adherence
            controlnet_conditioning_scale=1.2,  # Start value, will decrease by 25% over time
            guidance_scale=10.0,  # Higher CFG for stronger prompt adherence in scribble
            exec_device=exec_device  # Pass execution device for consistency
        )
        
        # Store edited latent in FP32 on CPU for quality preservation and VRAM efficiency
        edited_latent = edited_latent.detach().to(torch.float32).cpu()
        
        # Decode to image for preview (ensure proper dtype for VAE)
        with torch.no_grad():
            arr = pipe.decode_latents(edited_latent.to(exec_device).to(pipe.vae.dtype))[0]
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        img_np = (arr * 255.0).round().clip(0, 255).astype("uint8")
        edited_image = Image.fromarray(img_np)
        
        # Store edited latent in FP32 on CPU
        sessions[session_id]['edited_latent'] = edited_latent
        
        return jsonify({
            "success": True,
            "image": image_to_base64(edited_image)
        })
        
    except Exception as e:
        print(f"Error in edit_scribble: {str(e)}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/accept_edit', methods=['POST'])
def accept_edit():
    """Accept the current edit and make it the new base image."""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if session_id not in sessions:
            return jsonify({"success": False, "error": "No active session found"}), 400
        
        if 'edited_latent' in sessions[session_id]:
            # Move edited latent to be the new base latent (no VAE round-trip to avoid degradation)
            sessions[session_id]['latent'] = sessions[session_id]['edited_latent']
            del sessions[session_id]['edited_latent']
            print("Accepted edit - stored latent as-is without VAE round-trip to preserve quality")
            
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": "No edit to accept"}), 400
            
    except Exception as e:
        print(f"Error in accept_edit: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/reject_edit', methods=['POST'])
def reject_edit():
    """Reject the current edit."""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if session_id not in sessions:
            return jsonify({"success": False, "error": "No active session found"}), 400
        
        if 'edited_latent' in sessions[session_id]:
            del sessions[session_id]['edited_latent']
        
        return jsonify({"success": True})
        
    except Exception as e:
        print(f"Error in reject_edit: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/upload_image', methods=['POST'])
def upload_image():
    """Upload and process an image using NTI (Null-Text Inversion)."""
    try:
        load_models()
        
        data = request.json
        image_base64 = data.get('image')
        prompt = data.get('prompt', 'A person wearing clothing')
        session_id = data.get('session_id', str(uuid.uuid4()))
        nti_seed = data.get('nti_seed')  # None means random
        
        if not image_base64:
            return jsonify({"success": False, "error": "No image provided"}), 400
        
        # Convert base64 to PIL image
        image = base64_to_image(image_base64)
        
        print(f"Processing uploaded image with NTI. Prompt: '{prompt}'")
        print(f"Image size: {image.size}, mode: {image.mode}")
        
        # Resize image to 512x512 if needed
        if image.size != (512, 512):
            image = image.resize((512, 512), Image.Resampling.LANCZOS)
            print(f"Resized image to 512x512")
        
        # Convert RGBA to RGB if needed
        if image.mode == 'RGBA':
            white_bg = Image.new('RGB', image.size, (255, 255, 255))
            white_bg.paste(image, mask=image.split()[3])
            image = white_bg
            print("Converted RGBA to RGB with white background")
        elif image.mode != 'RGB':
            image = image.convert('RGB')
            print(f"Converted {image.mode} to RGB")
        
        # Process with NTI module
        print("Starting NTI processing...")
        nti_result = process_nti(
            image=image,
            prompt=prompt,
            pipe=pipe,
            nti_seed=nti_seed,
            max_iterations=150,
            nti_timesteps=10,
            preview=True,
            time_limit_sec=300.0
        )
        
        print(f"NTI completed. Best loss: {nti_result['metadata']['loss_best']:.6f}")
        print(f"NTI time: {nti_result['metadata']['elapsed_sec']:.1f}s")
        
        # Store NTI results in session
        if session_id not in sessions:
            sessions[session_id] = {}
        
        sessions[session_id]['latent'] = nti_result['latent']  # Already in FP32 on CPU
        sessions[session_id]['u_star'] = nti_result['u_star']   # NTI unconditional embedding
        sessions[session_id]['prompt'] = prompt
        sessions[session_id]['is_nti'] = True  # Flag to indicate this is from NTI
        sessions[session_id]['nti_metadata'] = nti_result['metadata']
        
        # Convert preview image to base64
        if nti_result['preview']:
            image_base64 = image_to_base64(nti_result['preview'])
        else:
            # Fallback: decode the latent
            exec_device = get_execution_device()
            with torch.no_grad():
                arr = pipe.decode_latents(nti_result['latent'].to(exec_device).to(pipe.vae.dtype))[0]
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
            img_np = (arr * 255.0).round().clip(0, 255).astype("uint8")
            preview_image = Image.fromarray(img_np)
            image_base64 = image_to_base64(preview_image)
        
        result = {
            "success": True,
            "image": image_base64,
            "session_id": session_id,
            "nti_metadata": nti_result['metadata']
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in upload_image: {str(e)}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Loading models on startup...")
    load_models()
    print("Server ready!")
    app.run(host='0.0.0.0', port=5000, debug=False)