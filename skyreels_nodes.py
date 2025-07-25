import torch
import numpy as np
from PIL import Image
import os
import cv2
import node_helpers
import folder_paths
import comfy.model_management as mm
import comfy

# Import classes from the original SkyReels-A1 codebase
from .skyreels_a1.pre_process_lmk3d import FaceAnimationProcessor, smooth_params
from .skyreels_a1.src.media_pipe.mp_utils import LMKExtractor
from .skyreels_a1.src.media_pipe.draw_util_2d import FaceMeshVisualizer2d
from .skyreels_a1.skyreels_a1_i2v_pipeline import SkyReelsA1ImagePoseToVideoPipeline
from .skyreels_a1.skyreels_a1_v2v_inpaint_pipeline import SkyReelsA1V2VInpaintPipeline
from .skyreels_a1.models.transformer3d import CogVideoXTransformer3DModel
from .skyreels_a1.ddim_solver import DDIMSolver

from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from diffusers.models import AutoencoderKLCogVideoX
from transformers import SiglipVisionModel, SiglipImageProcessor
from safetensors.torch import load_file

from einops import rearrange


script_directory = os.path.dirname(os.path.abspath(__file__))
models_directory = os.path.join(folder_paths.models_dir, "skyreels")
smirk_directory = os.path.join(models_directory, "smirk")

# A simple cache for loaded models to avoid reloading
LOADED_MODELS = {}
DEVICE = mm.get_torch_device()

def tensor_to_pil(tensor):
    """
    Converts a torch tensor to a PIL Image or a list of PIL Images.
    Handles shapes:
      - (C, H, W)
      - (H, W, C)
      - (1, C, H, W)
      - (B, C, H, W)
      - (B, H, W, C)
      - (H, W)
    Returns a single PIL Image if input is a single image, or a list of PIL Images if input is a batch.
    """
    tensor = tensor.detach().cpu()
    # If float, scale to 0-255
    if tensor.dtype != torch.uint8:
        tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
    # Remove batch dimension if present and batch size is 1
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    # Handle batch of images
    if tensor.ndim == 4:
        # (B, C, H, W) or (B, H, W, C)
        images = []
        for img in tensor:
            images.append(tensor_to_pil(img))
        return images
    # (C, H, W) -> (H, W, C)
    if tensor.ndim == 3 and tensor.shape[0] in [1, 3, 4]:
        tensor = tensor.permute(1, 2, 0)
    # (H, W, C) or (H, W)
    arr = tensor.numpy()
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr.squeeze(2)
    return Image.fromarray(arr)

def pil_to_tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def crop_and_resize(image, height, width):
    if isinstance(image, list):
            return [crop_and_resize(img, height, width) for img in image]
    image_width, image_height = image.size
        
    # Determine new size
    if image_width / image_height > width / height:
        new_height = height
        new_width = int(height * (image_width / image_height))
    else:
        new_width = width
        new_height = int(width * (image_height / image_width))
            
    image = image.resize((new_width, new_height), Image.LANCZOS)
        
    # Center crop
    left = (new_width - width) / 2
    top = (new_height - height) / 2
    right = (new_width + width) / 2
    bottom = (new_height + height) / 2
        
    return image.crop((left, top, right, bottom))

class SkyReelsPrepareDrivingImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "driving_image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("landmark_images",)
    FUNCTION = "prepare"
    CATEGORY = "SkyReels-A1"

    smirk_checkpoint = os.path.join(smirk_directory, "SMIRK_em1.pt")

    def prepare(self, source_image, driving_image):
        smirk_full_path = self.smirk_checkpoint
        # Initialize processors. It's okay to do this on each run for stateless processors.
        processor = FaceAnimationProcessor(checkpoint=smirk_full_path)
        lmk_extractor = LMKExtractor()
        vis = FaceMeshVisualizer2d(forehead_edge=False, draw_head=False, draw_iris=False)

        # Convert source and driving images to numpy arrays [F, H, W, C]
        source_images_np = (source_image.cpu().numpy() * 255.).astype(np.uint8)
        driving_frames_np = (driving_image.cpu().numpy() * 255.).astype(np.uint8)

        # --- Reference Face Preparation (from source_image) ---
        # Use only the first frame of the source_image as the reference, as in inference.py
        ref_frame_np = source_images_np[0]
        
        # Crop the face from the reference frame. This defines the face area and position.
        ref_face_crop_np, x1, y1 = processor.face_crop(ref_frame_np)
        if ref_face_crop_np is None:
            raise ValueError("No face detected in the source image.")
        face_h, face_w, _ = ref_face_crop_np.shape

        # --- Driving Frames Preparation ---
        # Crop the face from each driving frame
        driving_video_crop_np = []
        for d_frame in driving_frames_np:
            cropped_frame, _, _ = processor.face_crop(d_frame)
            if cropped_frame is not None:
                driving_video_crop_np.append(cropped_frame)

        if not driving_video_crop_np:
            raise ValueError("No faces detected in the driving image(s).")

        # --- Landmark Generation ---
        # 1. Generate the neutral (first motion) landmark frame from the reference face
        ref_face_resized_np = cv2.resize(ref_face_crop_np, (512, 512))
        ref_lmk = lmk_extractor(ref_face_resized_np[:, :, ::-1]) # Requires BGR
        ref_lmk_img = vis.draw_landmarks_v3((512, 512), (face_w, face_h), ref_lmk['lmks'].astype(np.float32), normed=True)
        
        # 2. Generate animated landmark frames from the driving video crops
        animated_lmk_imgs = processor.preprocess_lmk3d(
            source_image=ref_face_crop_np,
            driving_image_list=driving_video_crop_np
        )

        # --- Combine and Paste Landmarks onto Full-Size Canvas ---
        all_landmark_frames = []
        
        # Add the neutral frame first
        canvas_neutral = np.zeros_like(ref_frame_np)
        canvas_neutral[y1:y1+face_h, x1:x1+face_w] = ref_lmk_img
        all_landmark_frames.append(canvas_neutral)

        # Add the animated frames
        for lmk_img in animated_lmk_imgs:
            resized_landmark = cv2.resize(lmk_img, (face_w, face_h))
            canvas_animated = np.zeros_like(ref_frame_np)
            canvas_animated[y1:y1+face_h, x1:x1+face_w] = resized_landmark
            all_landmark_frames.append(canvas_animated)

        # Convert the list of numpy frames to a single torch tensor
        final_landmarks_tensor = torch.from_numpy(np.array(all_landmark_frames)).float() / 255.0
        
        return (final_landmarks_tensor,)

    
class SkyReelsPrepareDrivingImagesFromVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "driving_image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("landmark_images",)
    FUNCTION = "prepare"
    CATEGORY = "SkyReels-A1"

    smirk_checkpoint = os.path.join(smirk_directory, "SMIRK_em1.pt")

    def prepare(self, source_image, driving_image):
        smirk_full_path = self.smirk_checkpoint
        # Initialize processors. It's okay to do this on each run for stateless processors.
        processor = FaceAnimationProcessor(checkpoint=smirk_full_path)
        lmk_extractor = LMKExtractor()
        vis = FaceMeshVisualizer2d(forehead_edge=False, draw_head=False, draw_iris=False)

        # Convert source and driving images to numpy arrays [F, H, W, C]
        source_images_np = (source_image.cpu().numpy() * 255.).astype(np.uint8)
        driving_frames_np = (driving_image.cpu().numpy() * 255.).astype(np.uint8)

        # --- Reference Face Preparation (from source_image) ---
        # Use only the first frame of the source_image as the reference, as in inference.py
        ref_frame_np = source_images_np[0]
        ref_frame_crops = []
        ref_frame_h = []
        ref_frame_w = []
        ref_x1s = []
        ref_y1s = []
        
        # Crop the face from the reference frame. This defines the face area and position.
        for r_frame in source_images_np:
            ref_face_crop_np, x1, y1 = processor.face_crop(r_frame)
            if ref_face_crop_np is None:
                raise ValueError("No face detected in the source image.")
            face_h, face_w, _ = ref_face_crop_np.shape
            ref_frame_crops.append(ref_face_crop_np)
            ref_frame_h.append(face_h)
            ref_frame_w.append(face_w)
            ref_x1s.append(x1)
            ref_y1s.append(y1)

        # --- Driving Frames Preparation ---
        # Crop the face from each driving frame
        driving_video_crop_np = []
        for d_frame in driving_frames_np:
            cropped_frame, _, _ = processor.face_crop(d_frame)
            if cropped_frame is not None:
                driving_video_crop_np.append(cropped_frame)

        if not driving_video_crop_np:
            raise ValueError("No faces detected in the driving image(s).")

        # --- Combine and Paste Landmarks onto Full-Size Canvas ---
        all_landmark_frames = []
        num_frames = min(len(source_images_np), len(driving_frames_np))
        if len(source_images_np) != len(driving_frames_np):
            print(f"Warning: Source and driving videos have different lengths. Processing {num_frames} frames.")

        for i in range(num_frames):
            source_frame = source_images_np[i]
            ref_identity_face_crop_np = ref_frame_crops[i]
            driving_frame = driving_frames_np[i]
            driving_face_crop_np = driving_video_crop_np[i]

            animated_lmk_imgs = processor.preprocess_lmk3d(
                source_image=ref_identity_face_crop_np,
                driving_image_list=[driving_face_crop_np]
            )

            if not animated_lmk_imgs:
                print(f"Warning: Landmark generation failed for frame {i}. Reusing last successful frame.")
                if all_landmark_frames:
                    all_landmark_frames.append(all_landmark_frames[-1])
                else:
                    # If the first frame fails, we have nothing to reuse. Append a blank canvas.
                    all_landmark_frames.append(np.zeros_like(source_frame))
                continue

            lmk_img = animated_lmk_imgs[0]


        # Add the animated frames
            resized_landmark = cv2.resize(lmk_img, (ref_frame_w[i], ref_frame_h[i]))
            canvas_animated = np.zeros_like(source_frame)
            canvas_animated[ref_y1s[i]:ref_y1s[i]+ref_frame_h[i], ref_x1s[i]:ref_x1s[i]+ref_frame_w[i]] = resized_landmark
            all_landmark_frames.append(canvas_animated)

        # Convert the list of numpy frames to a single torch tensor
        final_landmarks_tensor = torch.from_numpy(np.array(all_landmark_frames)).float() / 255.0
        
        return (final_landmarks_tensor,)
    
class SkyReelsGetVideoData:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "driving_image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("DRIVING_DATA",)
    RETURN_NAMES = ("driving_data",)
    FUNCTION = "process"
    CATEGORY = "SkyReels-A1/Processing"

    smirk_checkpoint = os.path.join(smirk_directory, "SMIRK_em1.pt")

    def process(self, driving_image):
        # 1. Initialize the FaceAnimationProcessor
        processor = FaceAnimationProcessor(checkpoint=self.smirk_checkpoint)

        # 2. Convert the input tensor to a list of numpy frames
        driving_frames_np = (driving_image.cpu().numpy() * 255.).astype(np.uint8)

        # 3. Get driving_video_crop_np (copied directly from SkyReelsPrepareDrivingImages)
        # This loop iterates through the video, finds the face in each frame,
        # and collects the cropped faces into a list.
        driving_video_crop_np = []
        for d_frame in driving_frames_np:
            cropped_frame, _, _ = processor.face_crop(d_frame)
            if cropped_frame is not None:
                driving_video_crop_np.append(cropped_frame)

        # Handle case where no faces are found
        if not driving_video_crop_np:
            raise ValueError("No faces detected in the driving image(s).")

        # 4. Call processor.process_driving_img_list
        # This is the core step. We pass the list of cropped faces to get the raw parameters.
        # The function returns a 5-element tuple.
        (
            processed_frames, 
            driving_outputs, 
            driving_tforms, 
            weights_473, 
            weights_468
        ) = processor.process_driving_img_list(driving_video_crop_np)

        # 5. Package the outputs into a single driving_data object
        # As we discussed, this object will be a tuple containing the three elements
        # needed for blending and rendering.
        driving_data = (processed_frames, driving_outputs, driving_tforms, weights_473, weights_468)

        # 6. Return the packaged data
        return (driving_data,)
    
class SkyReelsBlendVideoData:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "driving_data_A": ("DRIVING_DATA",),
                "driving_data_B": ("DRIVING_DATA",),
                "expression_blend_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "jaw_blend_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "eye_blend_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("DRIVING_DATA",)
    RETURN_NAMES = ("driving_data_blended",)
    FUNCTION = "blend"
    CATEGORY = "SkyReels-A1/Processing"

    smirk_checkpoint = os.path.join(smirk_directory, "SMIRK_em1.pt")

    def blend(self, driving_data_A, driving_data_B, expression_blend_ratio, jaw_blend_ratio, eye_blend_ratio):
        # 1. Initialize the FaceAnimationProcessor
        processor = FaceAnimationProcessor(checkpoint=self.smirk_checkpoint)

        # 2. Call the blend_driving_outputs method
        # This method handles all the complex blending logic.
        driving_data_blended = processor.blend_driving_outputs(
            driving_data_A,
            driving_data_B,
            expression_blend_ratio,
            jaw_blend_ratio,
            eye_blend_ratio
        )

        # 3. Return the new blended data object
        return (driving_data_blended,)


class SkyreelsPrepareDrivingImagesFromData:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "driving_data": ("DRIVING_DATA",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("landmark_images",)
    FUNCTION = "prepare"
    CATEGORY = "SkyReels-A1/Processing"

    smirk_checkpoint = os.path.join(smirk_directory, "SMIRK_em1.pt")

    def prepare(self, source_image, driving_data):
        # 1. Initialize the FaceAnimationProcessor
        processor = FaceAnimationProcessor(checkpoint=self.smirk_checkpoint)

        # 2. Process the source image to get the reference face crop
        # This logic is copied from SkyReelsPrepareDrivingImages
        source_images_np = (source_image.cpu().numpy() * 255.).astype(np.uint8)
        ref_frame_np = source_images_np[0]
        
        ref_face_crop_np, x1, y1 = processor.face_crop(ref_frame_np)
        if ref_face_crop_np is None:
            raise ValueError("No face detected in the source image.")
        face_h, face_w, _ = ref_face_crop_np.shape

        # 3. Unpack the blended driving_data
        # The data can be a 3-element tuple from the blend node or a 5-element from the get node
        _, driving_outputs, _, weights_473, weights_468 = driving_data

        # 4. Call preprocess_lmk3d_from_outputs to generate animated landmark images
        animated_lmk_imgs = processor.preprocess_lmk3d_from_outputs(
            source_image=ref_face_crop_np,
            driving_outputs=driving_outputs,
            weights_473=weights_473,
            weights_468=weights_468
        )

        # 5. Combine and paste landmarks onto a full-size canvas
        # This logic is also adapted from SkyReelsPrepareDrivingImages
        all_landmark_frames = []
        for lmk_img in animated_lmk_imgs:
            resized_landmark = cv2.resize(lmk_img, (face_w, face_h))
            canvas_animated = np.zeros_like(ref_frame_np)
            canvas_animated[y1:y1+face_h, x1:x1+face_w] = resized_landmark
            all_landmark_frames.append(canvas_animated)

        # 6. Convert the list of numpy frames to a single torch tensor
        if not all_landmark_frames:
            return (torch.zeros_like(source_image),) # Return empty tensor if no frames
            
        final_landmarks_tensor = torch.from_numpy(np.array(all_landmark_frames)).float() / 255.0
        
        return (final_landmarks_tensor,)
    
V2V_CONFIG = {
  "activation_fn": "gelu-approximate",
  "attention_bias": True,
  "attention_head_dim": 64,
  "dropout": 0.0,
  "flip_sin_to_cos": True,
  "freq_shift": 0,
  "in_channels": 112,
  "max_text_seq_length": 226,
  "norm_elementwise_affine": True,
  "norm_eps": 1e-05,
  "num_attention_heads": 48,
  "num_layers": 42,
  "out_channels": 16,
  "patch_size": 2,
  "sample_frames": 49,
  "sample_height": 60,
  "sample_width": 90,
  "spatial_interpolation_scale": 1.875,
  "temporal_compression_ratio": 4,
  "temporal_interpolation_scale": 1.0,
  "text_embed_dim": 4096,
  "time_embed_dim": 512,
  "timestep_activation_fn": "silu",
  "use_learned_positional_embeddings": True,
  "use_rotary_positional_embeddings": True
}

VAE_CONFIG = {
  "act_fn": "silu",
  "block_out_channels": [
    128,
    256,
    256,
    512
  ],
  "down_block_types": [
    "CogVideoXDownBlock3D",
    "CogVideoXDownBlock3D",
    "CogVideoXDownBlock3D",
    "CogVideoXDownBlock3D"
  ],
  "force_upcast": True,
  "in_channels": 3,
  "latent_channels": 16,
  "latents_mean": None,
  "latents_std": None,
  "layers_per_block": 3,
  "norm_eps": 1e-06,
  "norm_num_groups": 32,
  "out_channels": 3,
  "sample_height": 480,
  "sample_width": 720,
  "scaling_factor": 0.7,
  "shift_factor": None,
  "temporal_compression_ratio": 4,
  "up_block_types": [
    "CogVideoXUpBlock3D",
    "CogVideoXUpBlock3D",
    "CogVideoXUpBlock3D",
    "CogVideoXUpBlock3D"
  ],
  "use_post_quant_conv": False,
  "use_quant_conv": False
}
    
class SkyReelsV2VModelLoader:
    @classmethod
    def INPUT_TYPES(_):
        return {
            "required": {
                "model_path": (folder_paths.get_filename_list("diffusion_models"))
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    FUNCTION = "loadmodel"
    CATEGORY = "SkyReels-A1"

    def loadmodel(self, model_path):
        model = CogVideoXTransformer3DModel(**V2V_CONFIG)
        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model_path)
        state_dict = comfy.utils.load_torch_file(model_path, safe_load=True)

        model.load_state_dict(state_dict, strict=True)
        model.eval().cuda()

        # TODO: create a class with a custom forward method and wrap in model patcher.

        return (model,)
    
class SkyReelsVAE:
    def __init__(self, sd):
        self.vae = AutoencoderKLCogVideoX(**VAE_CONFIG)
        self.vae.load_state_dict(sd, strict=True)
        self.vae.eval().cuda()

class SkyReelsVAELoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "vae_name": (folder_paths.get_filename_list("vae"), )}}
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"

    CATEGORY = "SkyReels-A1"
    DESCRIPTION = "Load a SkyReels VAE model."

    def load_vae(self, vae_name):
                # load uno lora safetensors
        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        sd = comfy.utils.load_torch_file(vae_path, safe_load=True)

        return (SkyReelsVAE(sd),)
    
class SkyReelsVAEEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "video": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "encode"
    CATEGORY = "SkyReels-A1"

    def encode(self, vae, video):
        # Normalize to tensor.
        if isinstance(video, list):
            video = torch.stack(video)
        elif video.ndim == 3:
            video = video.unsqueeze(0)

        video = video.float()

        if video.max() > 1.0:
            video = video / 255.0

        if video.min() >= 0.0:
            video = video * 2.0 - 1.0

        video = rearrange(video, "b f h w c -> b c f h w")

        device = mm.get_torch_device()
        video = video.to(device=device, dtype=torch.bfloat16)
        vae.vae.to(device=device, dtype=torch.bfloat16)

        latent = vae.vae.encode(video).latent_dist.sample() * VAE_CONFIG["scaling_factor"]
        return (latent,)
    
class SkyReelsVAEDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "SkyReels-A1"

    def decode(self, vae, latent):
        device = mm.get_torch_device()
        video = latent.to(device=device, dtype=torch.bfloat16)
        vae.vae.to(device=device, dtype=torch.bfloat16)

        latent = latent * (1 / VAE_CONFIG["scaling_factor"])
        video = vae.vae.decode(latent).latent_dist.sample()
        video = video * 0.5 + 0.5  # Scale to [0, 1]

        video = rearrange(video, "b c f h w -> b f h w c")
        # convert batch dimension to list.
        video = [b_video.float() for b_video in video]
        return (video,)
    
class SkyReelsLoadSiglipModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "siglip_model_path": (folder_paths.get_filename_list("TODO"),),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("siglip_model",)
    FUNCTION = "load_siglip"
    CATEGORY = "SkyReels-A1"

    def load_siglip(self, siglip_model_path):
        siglip_full_path = folder_paths.get_full_path_or_raise("siglip_models", siglip_model_path)
        siglip_model = SiglipVisionModel.from_pretrained(siglip_full_path)
        return (siglip_model,)
    
class SkyReelsFaceEmbedder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "siglip_model": ("MODEL",),
                "input_face": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("siglip_face",)
    FUNCTION = "embed"
    CATEGORY = "SkyReels-A1"

    def embed(self, siglip_model, input_face):
        pass

PIPELINE = None    
    
class SkyReelsSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ref_frames": ("IMAGE",),
                "landmark_frames": ("IMAGE",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "guidance_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "num_inference_steps": ("INT", {"default": 10, "min": 1, "max": 100}),
                "mask": ("MASK",)
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "sample"
    CATEGORY = "SkyReels-A1"

    smirk_checkpoint = os.path.join(smirk_directory, "SMIRK_em1.pt")

    def _load_model(self, loader_key, model_class, *args, **kwargs):
        if loader_key not in LOADED_MODELS:
            model_instance = model_class(*args, **kwargs)
            LOADED_MODELS[loader_key] = model_instance
        return LOADED_MODELS[loader_key]

    def _load_pipeline(self):
        global PIPELINE
        if PIPELINE is not None:
            return PIPELINE
        
        weight_dtype = torch.bfloat16

        device = mm.get_torch_device()
        offload_device = mm.get_offload_device()

        siglip_path = os.path.join(models_directory, "siglip-so400m-patch14-384")
        pipeline = SkyReelsA1V2VInpaintPipeline(
            models_directory, 
            siglip_path, 
            device, 
            offload_device, 
            weight_dtype,
        )

        PIPELINE = pipeline
        return pipeline

    def sample(
            self, 
            ref_frames, # [F, H, W, C]
            landmark_frames, # [F, H, W, C]
            seed, 
            guidance_scale, 
            num_inference_steps,
            mask, # [F, H, W]
        ):
        # Load models and helpers
        pipe = self._load_pipeline()
        face_helper = self._load_model("face_restore_helper", FaceRestoreHelper, upscale_factor=1, face_size=512, crop_ratio=(1, 1), det_model='retinaface_resnet50', save_ext='png', device=DEVICE)
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

        # Convert ref_frames to numpy for face_helper
        ref_frames_np = (ref_frames.cpu().numpy() * 255.).astype(np.uint8)
        if ref_frames_np.ndim == 4 and ref_frames_np.shape[0] == 1:
            ref_frames_np = ref_frames_np[0]

        # 1. Reshape ref frames and landmark frames.
        # Current shape: [F, H, W, C]
        ref_frames = rearrange(ref_frames, "f h w c -> c f h w")
        control_video = rearrange(landmark_frames, "f h w c -> c f h w")

        # 2. Prepare aligned face for pipeline (using numpy)
        face_helper.clean_all()
        face_helper.read_image(ref_frames_np[:, :, ::-1])
        face_helper.get_face_landmarks_5(only_center_face=True)
        face_helper.align_warp_face()
        aligned_face_bgr = face_helper.cropped_faces[0]
        aligned_face_np = aligned_face_bgr[:, :, ::-1]
        aligned_face = torch.from_numpy(aligned_face_np).float().permute(2, 0, 1)

        mask = mask.unsqueeze(0)

        if ref_frames.shape != control_video.shape:
            raise ValueError(f"Reference frame shape doesn't match control video shape")
        
        if ref_frames.shape != (3, 49, 480, 720):
            raise ValueError(f"49 frames of 480x720 expected, got {ref_frames.shape}")
        
        if mask.shape != (1, 49, 480, 720):
            raise ValueError(f"Mask shape should be (1, 49, 480, 720), got {mask.shape}")

        # 3. Run Sampler (returns (B, C, T, H, W))
        video = pipe.full_inference(
            ref_videos=[ref_frames],
            driving_videos=[control_video],
            pixel_masks=[mask],
            identity_images=[aligned_face],
            height=480,
            width=720,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        
        # Squeeze the batch dimension -> (F, C, H, W)
        video = video.squeeze(0)
        video = rearrange(video, "c f h w -> f h w c")

        return (video.float(),)

# --------------------------------------------------------------------------------
# --- REGISTRATION ---
# --------------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "SkyReelsPrepareDrivingImages": SkyReelsPrepareDrivingImages,
    "SkyReelsPrepareDrivingImagesFromVideo": SkyReelsPrepareDrivingImagesFromVideo,
    "SkyReelsGetVideoData": SkyReelsGetVideoData,
    "SkyReelsBlendVideoData": SkyReelsBlendVideoData,
    "SkyreelsPrepareDrivingImagesFromData": SkyreelsPrepareDrivingImagesFromData,
    "SkyReelsSampler": SkyReelsSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SkyReelsPrepareDrivingImages": "SkyReels Prepare Driving Images",
    "SkyReelsPrepareDrivingImagesFromVideo": "SkyReels Prepare Driving Images From Video",
    "SkyReelsGetVideoData": "SkyReels Get Video Data",
    "SkyReelsBlendVideoData": "SkyReels Blend Video Data",
    "SkyreelsPrepareDrivingImagesFromData": "SkyReels Prepare Driving Images From Data",
    "SkyReelsSampler": "SkyReels Sampler"
}
