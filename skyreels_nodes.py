import torch
import numpy as np
from PIL import Image
import os
import cv2
import folder_paths
import comfy.model_management as mm

# Import classes from the original SkyReels-A1 codebase
from .skyreels_a1.pre_process_lmk3d import FaceAnimationProcessor, smooth_params
from .skyreels_a1.src.media_pipe.mp_utils import LMKExtractor
from .skyreels_a1.src.media_pipe.draw_util_2d import FaceMeshVisualizer2d
from .skyreels_a1.skyreels_a1_i2v_pipeline import SkyReelsA1ImagePoseToVideoPipeline
from .skyreels_a1.skyreels_a1_i2v_inpaint_pipeline import SkyReelsA1InpaintPoseToVideoPipeline
from .skyreels_a1.models.transformer3d import CogVideoXTransformer3DModel
from .skyreels_a1.ddim_solver import DDIMSolver

from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from diffusers.models import AutoencoderKLCogVideoX
from transformers import SiglipVisionModel, SiglipImageProcessor
from safetensors.torch import load_file


script_directory = os.path.dirname(os.path.abspath(__file__))
models_directory = os.path.join(folder_paths.models_dir, "skyreels")
smirk_directory = os.path.join(models_directory, "smirk")

# A simple cache for loaded models to avoid reloading
LOADED_MODELS = {}
DEVICE = mm.get_torch_device()

def tensor_to_pil(tensor):
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim == 3:
        tensor = tensor.permute(1, 2, 0)
    
    tensor = (tensor * 255).to(torch.uint8)
    return Image.fromarray(tensor.cpu().numpy())

def pil_to_tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def crop_and_resize(image, height, width):
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
        processor = FaceAnimationProcessor(checkpoint=smirk_full_path)
        lmk_extractor = LMKExtractor()
        vis = FaceMeshVisualizer2d(forehead_edge=False, draw_head=False, draw_iris=False,)

        # Convert source_image and driving_image to numpy arrays
        source_images_np = (source_image.cpu().numpy() * 255.).astype(np.uint8)
        if source_images_np.ndim == 3:
            source_images_np = source_images_np[np.newaxis, ...]  # [F, H, W, C]
        driving_frames_np = (driving_image.cpu().numpy() * 255.).astype(np.uint8)
        if driving_frames_np.ndim == 3:
            driving_frames_np = driving_frames_np[np.newaxis, ...]  # [D, H, W, C]

        all_landmark_frames = []
        # Single-frame source: classic behavior
        if source_images_np.shape[0] == 1:
            frame = source_images_np[0]
            ref_image, x1, y1 = processor.face_crop(frame)
            face_h, face_w, _ = ref_image.shape
            # First motion: neutral pose
            ref_image_resized = cv2.resize(ref_image, (512, 512))
            ref_lmk = lmk_extractor(ref_image_resized[:, :, ::-1])
            ref_img = vis.draw_landmarks_v3((512, 512), (face_w, face_h), ref_lmk['lmks'].astype(np.float32), normed=True)
            first_motion = np.zeros_like(frame)
            first_motion[y1:y1+face_h, x1:x1+face_w] = ref_img
            all_landmark_frames.append(first_motion)

            # Prepare driving video crops
            driving_video_crop = []
            for dframe in driving_frames_np:
                cropped_frame, _, _ = processor.face_crop(dframe)
                driving_video_crop.append(cropped_frame)

            # Generate animated landmark frames
            driving_landmarks_list = processor.preprocess_lmk3d(
                source_image=ref_image,
                driving_image_list=driving_video_crop
            )
            for landmark in driving_landmarks_list:
                resized_landmark = cv2.resize(landmark, (face_w, face_h))
                canvas = np.zeros_like(frame)
                canvas[y1:y1+face_h, x1:x1+face_w] = resized_landmark
                all_landmark_frames.append(canvas)
        else:
            # Multi-frame: pair each source frame with corresponding driving frame
            num_frames = min(source_images_np.shape[0], driving_frames_np.shape[0])
            for idx in range(num_frames):
                frame = source_images_np[idx]
                dframe = driving_frames_np[idx]
                ref_image, x1, y1 = processor.face_crop(frame)
                face_h, face_w, _ = ref_image.shape
                # First motion: neutral pose for this frame
                ref_image_resized = cv2.resize(ref_image, (512, 512))
                ref_lmk = lmk_extractor(ref_image_resized[:, :, ::-1])
                ref_img = vis.draw_landmarks_v3((512, 512), (face_w, face_h), ref_lmk['lmks'].astype(np.float32), normed=True)
                first_motion = np.zeros_like(frame)
                first_motion[y1:y1+face_h, x1:x1+face_w] = ref_img
                all_landmark_frames.append(first_motion)

                # Driving frame crop
                cropped_driving, _, _ = processor.face_crop(dframe)
                # Generate landmark for this pair
                driving_landmarks_list = processor.preprocess_lmk3d(
                    source_image=ref_image,
                    driving_image_list=[cropped_driving]
                )
                for landmark in driving_landmarks_list:
                    resized_landmark = cv2.resize(landmark, (face_w, face_h))
                    canvas = np.zeros_like(frame)
                    canvas[y1:y1+face_h, x1:x1+face_w] = resized_landmark
                    all_landmark_frames.append(canvas)

        final_rendered_landmarks = torch.from_numpy(np.array(all_landmark_frames)).float() / 255.0
        return (final_rendered_landmarks,)


class SkyReelsSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder",}),
                "VAE": (folder_paths.get_filename_list("vae"), {"default": "SkyReelsVAE.safetensors"}),
                "source_image": ("IMAGE",),
                "landmark_images": ("IMAGE",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "guidance_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "num_inference_steps": ("INT", {"default": 10, "min": 1, "max": 100}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Leave on 1.0 unless in inpainting mode"}),
                "inpaint_mode": ("BOOLEAN", {"default": False, "tooltip": "Enable inpainting mode to use the mask input."}),
            },
            "optional": {
                "source_video": ("IMAGE",),
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

    def _load_pipeline(self, model, pose_guider, VAE, inpaint_mode):
        
        cache_key = (model, pose_guider, VAE, inpaint_mode)
        if cache_key in LOADED_MODELS:
            return LOADED_MODELS[cache_key]
        
        weight_dtype = torch.bfloat16

        vae_config_path = os.path.join(script_directory, 'configs', 'vae_config.json')
        pose_guider_config_path = os.path.join(script_directory, 'configs', 'pose_guider_config.json')
        transformer_config_path = os.path.join(script_directory, 'configs', 'transformer_config.json')

        vae_path = folder_paths.get_full_path_or_raise("vae", VAE)
        transformer_path = folder_paths.get_full_path_or_raise("diffusion_models", model)
        pose_guider_path = os.path.join(models_directory, 'skyreels/pose_guider', 'diffusion_pytorch_model.safetensors')
        
        transformer = CogVideoXTransformer3DModel.from_pretrained(transformer_config_path)
        transformer.load_state_dict(load_file(transformer_path))

        vae = AutoencoderKLCogVideoX.from_pretrained(vae_config_path)
        vae.load_state_dict(load_file(vae_path))

        lmk_encoder = AutoencoderKLCogVideoX.from_pretrained(pose_guider_config_path)
        lmk_encoder.load_state_dict(load_file(pose_guider_path))
        
        siglip_path = os.path.join(models_directory, "siglip-so400m-patch14-384")
        siglip = SiglipVisionModel.from_pretrained(siglip_path)
        siglip_normalize = SiglipImageProcessor.from_pretrained(siglip_path)

        if inpaint_mode == True:
            pipe = SkyReelsA1InpaintPoseToVideoPipeline(
                transformer=transformer, vae=vae, lmk_encoder=lmk_encoder,
                image_encoder=siglip, feature_extractor=siglip_normalize,
                torch_dtype=torch.bfloat16
            )
        else:
            pipe = SkyReelsA1ImagePoseToVideoPipeline(
                transformer=transformer, vae=vae, lmk_encoder=lmk_encoder,
                image_encoder=siglip, feature_extractor=siglip_normalize,
                torch_dtype=torch.bfloat16
            )

        pipe.to(DEVICE)
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
        LOADED_MODELS[cache_key] = pipe
        return pipe

    def sample(self, model, pose_guider, vae, source_image, landmark_images, seed, denoise, guidance_scale, num_inference_steps, inpaint_mode, source_video=None, mask=None):
        # Load models and helpers
        pipe = self._load_pipeline(model, pose_guider, vae, inpaint_mode)
        face_helper = self._load_model("face_restore_helper", FaceRestoreHelper, upscale_factor=1, face_size=512, crop_ratio=(1, 1), det_model='retinaface_resnet50', save_ext='png', device=DEVICE)
        
        smirk_full_path = self.smirk_checkpoint
        processor = self._load_model(f"face_animation_processor_{smirk_full_path}", FaceAnimationProcessor, checkpoint=smirk_full_path)
        
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

        # Get data from inputs
        source_image_pil = tensor_to_pil(source_image)
        source_image_pil_resized = crop_and_resize(source_image_pil, 480, 720)
        source_image_np = np.array(source_image_pil_resized)
        landmark_images_np = (landmark_images.cpu().numpy() * 255.).astype(np.uint8)
        
        height, width = source_image_np.shape[:2]
        num_frames = len(landmark_images_np)

        # 1. Prepare motion input for pipeline (paste all landmark frames onto full canvas)
        final_input_video_np = np.zeros((num_frames,) + source_image_np.shape, dtype=np.uint8)
        crop_info_list = []

        if inpaint_mode:
            if source_video is None:
                raise ValueError("Inpaint mode requires 'source_video' input.")
            
            source_video_np = (source_video.cpu().numpy() * 255.).astype(np.uint8)

            for i, frame_np in enumerate(source_video_np):
                try:
                    cropped_face, x1, y1 = processor.face_crop(frame_np)
                    h, w, _ = cropped_face.shape
                    crop_info_list.append({'x1': x1, 'y1': y1, 'w': w, 'h': h})

                    landmark_frame = landmark_images_np[i]
                    resized_landmark = cv2.resize(landmark_frame, (w, h))
                    final_input_video_np[i, y1:y1+h, x1:x1+w] = resized_landmark
                except IndexError:
                    print(f"Warning: No face detected in source_video frame {i}. Control frame will be black.")
                    crop_info_list.append(None)
        else:
            _, x1, y1 = processor.face_crop(source_image_np)
            face_h, face_w = landmark_images_np.shape[1:3]
            for i, landmark_frame in enumerate(landmark_images_np):
                # Ensure the landmark frame is resized to the source face crop size
                resized_landmark = cv2.resize(landmark_frame, (face_w, face_h))
                final_input_video_np[i, y1:y1+face_h, x1:x1+face_w] = resized_landmark
        
        # 2. Create final input video tensor
        input_video = torch.from_numpy(final_input_video_np).permute(0, 3, 1, 2)
        input_video = input_video.unsqueeze(0).permute(0, 2, 1, 3, 4)
        input_video = input_video / 255.0
        final_input_video = input_video.to(device=DEVICE, dtype=pipe.torch_dtype)

        # 3. Prepare aligned face for pipeline
        face_helper.clean_all()
        face_helper.read_image(np.array(source_image_pil)[:, :, ::-1])
        face_helper.get_face_landmarks_5(only_center_face=True)
        face_helper.align_warp_face()
        aligned_face_bgr = face_helper.cropped_faces[0]
        aligned_face_pil = Image.fromarray(aligned_face_bgr[:, :, ::-1])

        # 4. Run Sampler
        pipeline_args = {
            "image_face": aligned_face_pil,
            "image": source_image_pil_resized,
            "control_video": final_input_video,
            "prompt": "", 
            "generator": generator,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "output_type": "pil",
        }

        if inpaint_mode:
            if source_video is None or mask is None:
                raise ValueError("Inpaint mode requires 'source_video' and 'mask' inputs.")

            video_for_pipeline = source_video.permute(3, 0, 1, 2).unsqueeze(0)
            pipeline_args["video"] = video_for_pipeline.to(device=DEVICE, dtype=pipe.torch_dtype)

            pipeline_args["strength"] = denoise

            # support per-frame mask stacks [F, H, W] or a single 2D mask [H, W]
            if isinstance(mask, torch.Tensor) and mask.ndim == 3:
                mask_for_pipeline = mask.unsqueeze(0).unsqueeze(1)  # [1, 1, F, H, W]
            else:
                mask_for_pipeline = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, 1, num_frames, 1, 1)
            pipeline_args["mask"] = mask_for_pipeline.to(device=DEVICE, dtype=pipe.torch_dtype)
            
            # Run inpaint pipeline and return full frames directly
            with torch.no_grad():
                video_frames_pil = pipe(**pipeline_args).frames
            output_tensors = [torch.from_numpy(np.array(frame)).float() / 255.0 for frame in video_frames_pil]
            final_video_tensor = torch.stack(output_tensors)

        else:
            with torch.no_grad():
                video_frames_pil = pipe(**pipeline_args).frames
            output_tensors = [torch.from_numpy(np.array(frame)).float() / 255.0 for frame in video_frames_pil]
            final_video_tensor = torch.stack(output_tensors)

        return (final_video_tensor,)



# --------------------------------------------------------------------------------
# --- REGISTRATION ---
# --------------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "SkyReelsPrepareDrivingImages": SkyReelsPrepareDrivingImages,
    "SkyReelsSampler": SkyReelsSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SkyReelsPrepareDrivingImages": "SkyReels Prepare Driving Images",
    "SkyReelsSampler": "SkyReels Sampler"
}