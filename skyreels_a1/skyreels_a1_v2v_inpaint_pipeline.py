import torch
import torch.nn.functional as F

from diffusers.models import AutoencoderKLCogVideoX
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.schedulers import CogVideoXDDIMScheduler
from transformers import AutoModelForDepthEstimation, AutoProcessor, SiglipImageProcessor, SiglipVisionModel
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from einops import rearrange
from tqdm import tqdm

from typing import Tuple
import inspect
import types

from skyreels_a1.models.transformer3d import CogVideoXTransformer3DModel

# Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps_for_inference(
    scheduler,
    num_inference_steps,
    device,
    timesteps=None,
    sigmas=None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class SkyReelsA1V2VInpaintPipeline:
    def __init__(self, model_name, siglip_name, device, offload_device, dtype):
        self.dtype = dtype
        self.device = device
        self.offload_device = offload_device

        # We restore the full model only if no lora rank is specified.
        self.transformer = CogVideoXTransformer3DModel.from_pretrained(
            model_name,
        ).to(offload_device, self.dtype)

        self.vae = AutoencoderKLCogVideoX.from_pretrained(
            model_name, 
            subfolder="vae"
        ).to(offload_device, self.dtype)
        self.vae.enable_tiling()

        self.lmk_encoder = AutoencoderKLCogVideoX.from_pretrained(
            model_name, 
            subfolder="pose_guider",
        ).to(offload_device, self.dtype)

        self.scheduler = CogVideoXDDIMScheduler.from_pretrained(
            model_name,
            subfolder="scheduler"
        )
        self.inference_scheduler = CogVideoXDDIMScheduler.from_pretrained(
            model_name,
            subfolder="scheduler"
        )
        self.vae_scaling_factor_image = self.vae.config.scaling_factor
        self.lmk_scaling_factor_image = self.lmk_encoder.config.scaling_factor

        self.siglip = SiglipVisionModel.from_pretrained(siglip_name).to(self.dtype)
        self.siglip_normalize = SiglipImageProcessor.from_pretrained(siglip_name)
        self.t_config = self.transformer.config

    # Copied from diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline._prepare_rotary_positional_embeddings
    @torch.no_grad()
    def _prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.t_config.patch_size
        vae_scale_factor_spatial = 8
        grid_height = height // (vae_scale_factor_spatial * p)
        grid_width = width // (vae_scale_factor_spatial * p)
        base_size_width = self.t_config.sample_width // p
        base_size_height = self.t_config.sample_height // p

        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=self.t_config.attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
        )

        freqs_cos = freqs_cos.to(device=device)
        freqs_sin = freqs_sin.to(device=device)
        return freqs_cos, freqs_sin

    @torch.no_grad()
    def prepare_masks(self, pixel_masks):
        """
            Given a list of pixel masks, this function prepares masks in latent space.
            These have 64 channels, to account for the 8x8 downsampling of the VAE.
        """
        result_masks = []
        for mask in pixel_masks:
            mask = mask.to(torch.bfloat16)
            c, depth, height, width = mask.shape
            new_depth = int((depth + 3) // 4)
            height = 2 * (int(height) // (8 * 2))
            width = 2 * (int(width) // (8 * 2))

            # scale by 8x8 to match the latent space of the VAE
            mask = rearrange(
                mask,
                "1 t (h ph) (w pw) -> (ph pw) t h w",
                ph=8,
                pw=8,
            )

            # temporal interpolation
            mask = F.interpolate(mask.unsqueeze(0), size=(new_depth, height, width), mode='nearest-exact').squeeze(0)

            result_masks.append(mask)

        return result_masks
    
    @torch.no_grad()
    def prepare_latent(
        self, 
        ref_videos, 
        driving_videos, 
        pixel_masks, 
        latent_masks, 
        timesteps,
        noisy_latent=None,
        generator=None,
    ):
        """
        Prepares the latent representations of the reference and driving videos, as well as the pixel masks.

        Args:
            ref_videos: List of reference video tensors (C, T, H, W)
            driving_videos: List of driving video tensors (C, T, H, W)
            pixel_masks: List of pixel mask tensors (1, T, H, W)
            latent_masks: List of latent mask tensors (64, T', H', W')
            timesteps: List of timesteps

        Returns model inputs (B, (k+64)C', T', H', W') and clean latents (B, C', T', H', W').
        The model inputs are concatenated along the channel dimension in the order:
            [noisy_latent, lmk_latent, ref_latent, latent_mask] with the +64 depending on
            whether `self.explicit_mask_channels` is set to True or False.
        """

        ref_videos = torch.stack(ref_videos, dim=0).to(self.device, self.dtype) # (B, C, T, H, W)
        driving_videos = torch.stack(driving_videos, dim=0).to(self.device, self.dtype)  # (B, C, T, H, W)
        pixel_masks = torch.stack(pixel_masks, dim=0).to(self.device, self.dtype)  # (B, 1, T, H, W)
        latent_masks = torch.stack(latent_masks, dim=0).to(self.device, self.dtype)  # (B, 64, T', H', W')

        if noisy_latent is None:
            clean_latent = self.vae.encode(ref_videos).latent_dist.sample()  # (B, C, T', H', W_)
            clean_latent = clean_latent * self.vae_scaling_factor_image
            noise = torch.randn_like(
                clean_latent, 
                device=ref_videos.device, 
                dtype=self.dtype,
                generator=generator
            )

            noisy_latent = self.scheduler.add_noise(
                clean_latent, 
                noise, 
                torch.tensor(timesteps, dtype=torch.int64, device=self.device)
            )
        else:
            clean_latent = None

        # note: this is a no-op anyway with this scheduler, but would require looping for this impl
        # so we skip it
        # noisy_latent = self.scheduler.scale_model_input(noisy_latent, timesteps)

        # Cut out all pixels in the mask from the reference.
        # We leave the first frame intact.
        pixel_mask = pixel_masks[:, :, 1:, :, :]
        ref_videos[:, :, 1:, :, :] *= (1.0 - pixel_mask)

        ref_latent = self.vae.encode(ref_videos).latent_dist.sample() * self.vae_scaling_factor_image

        lmk_latent = self.lmk_encoder.encode(driving_videos).latent_dist.mode()
        lmk_latent *= self.lmk_scaling_factor_image

        # concatenate along channel dimension (B, C, T', H', W')
        model_input = torch.cat([noisy_latent, lmk_latent, ref_latent, latent_masks], dim=1)

        return model_input, clean_latent

    @torch.no_grad()
    def embed_reference_prompt(self, identity_images):
        """
        Embeds each identity image using the Siglip model.

        Args:
            identity_images: List of reference video images (C, H, W) with values in [0, 255]

        Returns:
            A tensor of shape [B, 729, 1152] where B is the number of reference videos.
        """
        
        self.siglip.to(self.device)
        imgs = self.siglip_normalize.preprocess(images=identity_images, do_resize=True, return_tensors="pt", do_convert_rgb=True)
        imgs = imgs.to(self.device, self.dtype)
        image_embeddings = self.siglip(**imgs).last_hidden_state  # torch.Size([B, 729, 1152])
        self.siglip.to(self.offload_device)

        return image_embeddings.to(self.device, self.dtype)
    
    @torch.no_grad()
    def full_inference(
        self,
        ref_videos, 
        driving_videos, 
        pixel_masks,
        identity_images,
        height,
        width,
        num_inference_steps=5,
        guidance_scale=3.0,
        generator=None,
    ):
        """
        Performs a full inference trajectory of the pipeline as it stands.
        This uses CFG with a guidance scale of 3.0 over 5 inference steps.
        Args:
            transformer: The unwrapped transformer model.
            ref_videos: List of reference video tensors (C, T, H, W) in [-1, 1]
            driving_videos: List of driving video tensors (C, T, H, W) in [-1, 1]
            pixel_masks: List of pixel mask tensors (1, T, H, W)
            identity_images: List of identity images (C, H, W) in [0, 255]
            height: Height of the input videos
            width: Width of the input videos
            guidance_scale: Guidance scale for CFG
            num_inference_steps: Number of inference steps to perform
        Returns:
            A tensor of shape (B, C, T, H, W) where B is the batch size of ref_videos.

        """

        self.vae.to(self.device)
        self.lmk_encoder.to(self.device)

        batch_size = len(ref_videos)
        noisy_latent = torch.randn(
            (batch_size, 16, 13, 60, 90), 
            device=self.device,
            generator=generator,
        )
        latent_masks = self.prepare_masks(pixel_masks)
        model_inputs, _ = self.prepare_latent(
            ref_videos, 
            driving_videos, 
            pixel_masks, 
            latent_masks, 
            [0] * batch_size, #not used
            noisy_latent=noisy_latent,
            generator=generator,
        )

        self.lmk_encoder.to(self.offload_device)

        self.siglip.to(self.device)

        model_inputs = torch.cat([model_inputs, model_inputs], dim=0)  # Duplicate for CFG
        image_embeddings = self.embed_reference_prompt(identity_images)

        self.siglip.to(self.offload_device)

        # extend to B*2 with zeros, for CFG
        image_embeddings = torch.cat([torch.zeros_like(image_embeddings), image_embeddings], dim=0)
        image_rotary_emb = self._prepare_rotary_positional_embeddings(
            height=height,
            width=width,
            num_frames=model_inputs.shape[2],  # T'
            device=self.device,
        )

        timesteps, num_inference_steps = retrieve_timesteps_for_inference(
            self.inference_scheduler,
            num_inference_steps,
            self.device, 
            None,
        )

        self.transformer.to(self.device)

        # Swap channels/frames -> (B, T, C, H', W')
        model_inputs = model_inputs.permute(0, 2, 1, 3, 4)
        for i, t in enumerate(tqdm(timesteps)):
            model_inputs = model_inputs.to(self.transformer.device, self.dtype)
            image_embeddings = image_embeddings.to(self.transformer.device, self.dtype)

            timestep = t.expand(model_inputs.shape[0])
            noise_pred = self.transformer(
                hidden_states=model_inputs,
                encoder_hidden_states=image_embeddings,
                timestep=timestep,
                image_rotary_emb=image_rotary_emb,
                return_dict=False,
            )[0]
            noise_pred = noise_pred.float()

            noise_pred_uncond, noise_pred = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            # update the noisy latents, leave the rest of the model inputs untouched.
            latents = model_inputs[batch_size:, :, 0:16, :, :].float()
            latents = self.inference_scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            latents = torch.cat([latents, latents], dim=0).to(model_inputs.dtype)
            model_inputs[:, :, 0:16, :, :] = latents

        self.transformer.to(self.offload_device)

        latents = latents[batch_size:].permute(0, 2, 1, 3, 4)  # (B, C, T', H', W')
        latents = latents * (1 / self.vae_scaling_factor_image)
        out = self.vae.decode(latents).sample
        out = torch.clamp(out * 0.5 + 0.5, min=0.0, max=1.0)  # Scale back to [0, 1]

        self.vae.to(self.offload_device)
        return out 
