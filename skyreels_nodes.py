import torch
import numpy as np
from PIL import Image
import os
import cv2
import folder_paths

# Import classes from the original SkyReels-A1 codebase
from skyreels_a1.pre_process_lmk3d import FaceAnimationProcessor
from skyreels_a1.src.media_pipe.mp_utils import LMKExtractor
from skyreels_a1.src.media_pipe.draw_util_2d import FaceMeshVisualizer2d
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from skyreels_a1.skyreels_a1_i2v_long_pipeline import SkyReelsA1ImagePoseToVideoPipeline
from skyreels_a1.models.transformer3d import CogVideoXTransformer3DModel
from diffusers.models import AutoencoderKLCogVideoX
from transformers import SiglipVisionModel, SiglipImageProcessor
from diffposetalk.utils.common import OneEuroFilter  # Import the filter

# A simple cache for loaded models to avoid reloading
LOADED_MODELS = {}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SkyReelsPrepareDrivingEmbeddings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "driving_image": ("IMAGE",),
            }
        }
    RETURN_TYPES = ("DRIVING_EMBEDDINGS",)
    RETURN_NAMES = ("driving_embeddings",)
    FUNCTION = "prepare"
    CATEGORY = "SkyReels-A1"

    def _load_model(self, loader_key, model_class, *args, **kwargs):
        if loader_key not in LOADED_MODELS:
            LOADED_MODELS[loader_key] = model_class(*args, **kwargs)
        return LOADED_MODELS[loader_key]

    def prepare(self, driving_image):
        lmk_extractor = self._load_model("lmk_extractor", LMKExtractor)
        
        # Convert torch tensor to numpy array
        driving_frames_np = (driving_image.cpu().numpy() * 255.).astype(np.uint8)
        
        # Extract 3D landmarks from each frame in the driving image batch
        driving_lmk3d_list = lmk_extractor.extract_lmk3d_from_frames(driving_frames_np)

        # Return raw, unsmoothed landmarks
        return ({"driving_lmk3d_list": driving_lmk3d_list, "type": "raw"},)

class SkyReelsSmoothEmbeddings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "driving_embeddings": ("DRIVING_EMBEDDINGS",),
                "min_cutoff": ("FLOAT", {"default": 0.004, "min": 0.0, "max": 1.0, "step": 0.001}),
                "beta": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }
    RETURN_TYPES = ("DRIVING_EMBEDDINGS",)
    FUNCTION = "smooth"
    CATEGORY = "SkyReels-A1"

    def smooth(self, driving_embeddings, min_cutoff, beta):
        if driving_embeddings.get("type") != "raw":
            # If embeddings are already smoothed, just pass them through
            return (driving_embeddings,)

        lmk3d_list = driving_embeddings["driving_lmk3d_list"]
        if not lmk3d_list or len(lmk3d_list) <= 1:
            return (driving_embeddings,) # Not enough frames to smooth

        # Initialize OneEuroFilter
        one_euro_filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)

        smoothed_lmk3d_list = []
        for lmk3d in lmk3d_list:
            if lmk3d is not None:
                smoothed_lmk3d = one_euro_filter.process(lmk3d)
                smoothed_lmk3d_list.append(smoothed_lmk3d)
            else:
                smoothed_lmk3d_list.append(None) # Preserve None entries

        return ({"driving_lmk3d_list": smoothed_lmk3d_list, "type": "smoothed"},)

class SkyReelsSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": (folder_paths.get_folder_list("skyreels/pretrained_models/SkyReels-A1-5B"), ),
                "smirk_checkpoint": (folder_paths.get_filename_list("skyreels/pretrained_models/smirk"), ),
                "source_image": ("IMAGE",),
                "driving_embeddings": ("DRIVING_EMBEDDINGS",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "guidance_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "num_inference_steps": ("INT", {"default": 10, "min": 1, "max": 100}),
            }
        }
    RETURN_TYPES = ("IMAGE", "IMAGE", "CROP_INFO")
    RETURN_NAMES = ("animated_face", "source_image", "crop_info")
    FUNCTION = "sample"
    CATEGORY = "SkyReels-A1"

    def _load_model(self, loader_key, model_class, *args, **kwargs):
        if loader_key not in LOADED_MODELS:
            model_instance = model_class(*args, **kwargs)
            LOADED_MODELS[loader_key] = model_instance
        return LOADED_MODELS[loader_key]

    def _load_pipeline(self, model_path):
        if model_path in LOADED_MODELS:
            return LOADED_MODELS[model_path]
        
        weight_dtype = torch.bfloat16
        
        transformer = CogVideoXTransformer3DModel.from_pretrained(model_path, subfolder="transformer").to(weight_dtype)
        vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae").to(weight_dtype)
        lmk_encoder = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="pose_guider").to(weight_dtype)
        
        siglip_path = os.path.join(model_path, "siglip-so400m-patch14-384")
        siglip = SiglipVisionModel.from_pretrained(siglip_path)
        siglip_normalize = SiglipImageProcessor.from_pretrained(siglip_path)

        pipe = SkyReelsA1ImagePoseToVideoPipeline.from_pretrained(
            model_path,
            transformer=transformer, vae=vae, lmk_encoder=lmk_encoder,
            image_encoder=siglip, feature_extractor=siglip_normalize,
            torch_dtype=weight_dtype
        )
        pipe.to(DEVICE)
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
        LOADED_MODELS[model_path] = pipe
        return pipe

    def sample(self, model_path, smirk_checkpoint, source_image, driving_embeddings, seed, guidance_scale, num_inference_steps):
        model_full_path = folder_paths.get_full_path("skyreels", model_path)
        pipe = self._load_pipeline(model_full_path)
        
        smirk_full_path = folder_paths.get_full_path("skyreels", smirk_checkpoint)
        processor = self._load_model(f"face_animation_processor_{smirk_checkpoint}", FaceAnimationProcessor, checkpoint=smirk_full_path)
        
        face_helper = self._load_model("face_restore_helper", FaceRestoreHelper, upscale_factor=1, face_size=512, crop_ratio=(1, 1), det_model='retinaface_resnet50', save_ext='png', device=DEVICE)
        lmk_extractor = self._load_model("lmk_extractor", LMKExtractor)
        vis = self._load_model("visualizer", FaceMeshVisualizer2d, forehead_edge=False, draw_head=False, draw_iris=False)
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

        # 1. Process Source Image
        source_image_pil = Image.fromarray((source_image[0].cpu().numpy() * 255.).astype(np.uint8))
        source_image_np = np.array(source_image_pil)
        source_image_cropped_np, x1, y1 = processor.face_crop(source_image_np)
        if source_image_cropped_np is None:
            raise ValueError("No face detected in the source image.")
        face_h, face_w, _ = source_image_cropped_np.shape
        source_outputs, source_tform, _ = processor.process_source_image(source_image_cropped_np)

        # 2. Process Driving Data
        driving_lmk3d_list = driving_embeddings["driving_lmk3d_list"]

        # 3. Generate motion poses from landmarks
        out_frames_np = FaceAnimationProcessor.preprocess_lmk3d_from_lmk3d(
            source_outputs, source_tform, source_image_cropped_np.shape, driving_lmk3d_list
        )

        # 4. Prepare inputs for the pipeline
        rescale_motions = np.zeros((len(out_frames_np),) + source_image_np.shape, dtype=np.uint8)
        for i, out_frame in enumerate(out_frames_np):
            rescale_motions[i, y1:y1+face_h, x1:x1+face_w] = out_frame
        
        input_video = torch.from_numpy(rescale_motions).permute(0, 3, 1, 2)
        input_video = input_video.unsqueeze(0).permute(0, 2, 1, 3, 4)
        input_video = input_video / 255.0

        ref_image_resized = cv2.resize(source_image_cropped_np, (512, 512))
        ref_lmk = lmk_extractor(ref_image_resized[:, :, ::-1])
        ref_img_drawn = vis.draw_landmarks_v3((512, 512), (face_w, face_h), ref_lmk['lmks'].astype(np.float32), normed=True)
        
        first_motion_np = np.zeros_like(source_image_np)
        first_motion_np[y1:y1+face_h, x1:x1+face_w] = cv2.resize(ref_img_drawn, (face_w, face_h))
        first_motion = torch.from_numpy(first_motion_np).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        first_motion = first_motion / 255.0

        final_input_video = torch.cat([first_motion, input_video], dim=2).to(device=DEVICE, dtype=pipe.torch_dtype)

        face_helper.clean_all()
        face_helper.read_image(source_image_np[:, :, ::-1])
        face_helper.get_face_landmarks_5(only_center_face=True)
        face_helper.align_warp_face()
        aligned_face_bgr = face_helper.cropped_faces[0]
        aligned_face_pil = Image.fromarray(aligned_face_bgr[:, :, ::-1])

        # 5. Run Sampler
        with torch.no_grad():
            video_frames_pil = pipe(
                aligned_face_pil,
                final_input_video,
                generator=generator,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                height=source_image_np.shape[0],
                width=source_image_np.shape[1],
                latents_cache=None,
                is_first_batch=True,
                is_last_batch=True,
                output_type="pil"
            ).frames

        # 6. Crop faces from output and convert to tensor
        output_tensors = []
        for frame_pil in video_frames_pil:
            frame_np = np.array(frame_pil)
            animated_face_crop_np = frame_np[y1:y1+face_h, x1:x1+face_w]
            output_tensors.append(torch.from_numpy(animated_face_crop_np).float() / 255.0)
        
        animated_faces_tensors = torch.stack(output_tensors)

        crop_info = {"x1": x1, "y1": y1, "face_w": face_w, "face_h": face_h}

        return (animated_faces_tensors, source_image, crop_info)

class SkyReelsComposite:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "animated_face": ("IMAGE",),
                "source_image": ("IMAGE",),
                "crop_info": ("CROP_INFO",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite"
    CATEGORY = "SkyReels-A1"

    def composite(self, animated_face, source_image, crop_info):
        x1 = crop_info["x1"]
        y1 = crop_info["y1"]
        face_w = crop_info["face_w"]
        face_h = crop_info["face_h"]

        source_np = (source_image[0].cpu().numpy() * 255.).astype(np.uint8)
        animation_np = (animated_face.cpu().numpy() * 255.).astype(np.uint8)
        
        output_frames = []
        for i in range(animation_np.shape[0]):
            composited_frame = source_np.copy()
            composited_frame[y1:y1+face_h, x1:x1+face_w] = animation_np[i]
            output_frames.append(composited_frame)
            
        output_tensors = [torch.from_numpy(frame).float() / 255.0 for frame in output_frames]
        return (torch.stack(output_tensors),)

# --------------------------------------------------------------------------------
# --- REGISTRATION ---
# --------------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "SkyReelsPrepareDrivingEmbeddings": SkyReelsPrepareDrivingEmbeddings,
    "SkyReelsSmoothEmbeddings": SkyReelsSmoothEmbeddings,
    "SkyReelsSampler": SkyReelsSampler,
    "SkyReelsComposite": SkyReelsComposite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SkyReelsPrepareDrivingEmbeddings": "SkyReels Prepare Driving Embeddings",
    "SkyReelsSmoothEmbeddingsForVideo": "SkyReels Smooth Driving Embeddings",
    "SkyReelsSampler": "SkyReels Sampler",
    "SkyReelsComposite": "SkyReels Composite",
}