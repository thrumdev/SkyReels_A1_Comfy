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
        if len(driving_data) == 5:
            _, driving_outputs, _, weights_473, weights_468 = driving_data
        elif len(driving_data) == 3:
            driving_outputs, weights_473, weights_468 = driving_data
        else:
            raise ValueError(f"Unexpected driving_data tuple length: {len(driving_data)}")

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


class SkyReelsSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
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

    def _load_pipeline(self, inpaint_mode):
        
        weight_dtype = torch.bfloat16

        transformer = CogVideoXTransformer3DModel.from_pretrained(
            models_directory, 
            subfolder="transformer"
        ).to(weight_dtype)

        vae = AutoencoderKLCogVideoX.from_pretrained(
            models_directory, 
            subfolder="vae"
        ).to(weight_dtype)

        lmk_encoder = AutoencoderKLCogVideoX.from_pretrained(
            models_directory, 
            subfolder="pose_guider"
        ).to(weight_dtype)
        
        siglip_path = os.path.join(models_directory, "siglip-so400m-patch14-384")
        siglip = SiglipVisionModel.from_pretrained(siglip_path)
        siglip_normalize = SiglipImageProcessor.from_pretrained(siglip_path)

        cache_key = (transformer, lmk_encoder, vae, inpaint_mode)
        if cache_key in LOADED_MODELS:
            return LOADED_MODELS[cache_key]

        pipeline_class = SkyReelsA1InpaintPoseToVideoPipeline if inpaint_mode else SkyReelsA1ImagePoseToVideoPipeline
        pipe = pipeline_class.from_pretrained(
            models_directory,
            transformer=transformer,
            vae=vae,
            lmk_encoder=lmk_encoder,
            image_encoder=siglip,
            feature_extractor=siglip_normalize,
            torch_dtype=weight_dtype
        )

        pipe.to(DEVICE)
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
        LOADED_MODELS[cache_key] = pipe
        return pipe

    def sample(self, source_image, landmark_images, seed, denoise, guidance_scale, num_inference_steps, inpaint_mode, source_video=None, mask=None):
        # Load models and helpers
        pipe = self._load_pipeline(inpaint_mode)
        face_helper = self._load_model("face_restore_helper", FaceRestoreHelper, upscale_factor=1, face_size=512, crop_ratio=(1, 1), det_model='retinaface_resnet50', save_ext='png', device=DEVICE)
        smirk_full_path = self.smirk_checkpoint
        processor = self._load_model(f"face_animation_processor_{smirk_full_path}", FaceAnimationProcessor, checkpoint=smirk_full_path)
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

        # Convert source_image to numpy for face_helper
        source_image_np = (source_image.cpu().numpy() * 255.).astype(np.uint8)
        if source_image_np.ndim == 4 and source_image_np.shape[0] == 1:
            source_image_np = source_image_np[0]

        # The landmark_images tensor is the complete, final control video.
        # We just need to get its shape and format it for the pipeline.
        num_frames, height, width, _ = landmark_images.shape

        # 1. Prepare the control_video tensor for the pipeline
        # Current shape: [F, H, W, C]
        # Target shape: [B, C, F, H, W] (where B=1)
        control_video = landmark_images.permute(3, 0, 1, 2) # -> [C, F, H, W]
        control_video = control_video.unsqueeze(0)         # -> [1, C, F, H, W]
        final_input_video = control_video.to(device=DEVICE, dtype=torch.bfloat16)

        # 2. Prepare aligned face for pipeline (using numpy)
        face_helper.clean_all()
        face_helper.read_image(source_image_np[:, :, ::-1])
        face_helper.get_face_landmarks_5(only_center_face=True)
        face_helper.align_warp_face()
        aligned_face_bgr = face_helper.cropped_faces[0]
        aligned_face_np = aligned_face_bgr[:, :, ::-1]

        # 3. Run Sampler
        pipeline_args = {
            "image_face": Image.fromarray(aligned_face_np),
            "image": Image.fromarray(source_image_np),
            "control_video": final_input_video,
            "prompt": "", 
            "generator": generator,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "output_type": "pt",
        }

        if inpaint_mode:
            if source_video is None or mask is None:
                raise ValueError("Inpaint mode requires 'source_video' and 'mask' inputs.")
            video_for_pipeline = source_video.permute(3, 0, 1, 2).unsqueeze(0)
            pipeline_args["video"] = video_for_pipeline.to(device=DEVICE, dtype=torch.bfloat16)
            pipeline_args["strength"] = denoise

            # support per-frame mask stacks [F, H, W] or a single 2D mask [H, W]
            if isinstance(mask, torch.Tensor) and mask.ndim == 3:
                mask_for_pipeline = mask.unsqueeze(0).unsqueeze(1)  # [1, 1, F, H, W]
            else:
                mask_for_pipeline = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, 1, num_frames, 1, 1)
            pipeline_args["mask"] = mask_for_pipeline.to(device=DEVICE, dtype=torch.bfloat16)
            
            # Run inpaint pipeline and return full frames directly
            with torch.no_grad():
                video_tensor = pipe(**pipeline_args).frames
            #output_tensors = [torch.from_numpy(np.array(frame)).float() / 255.0 for frame in video_frames_pil]
            #video_tensor = torch.stack(output_tensors)
        else:
            with torch.no_grad():
            # The pipeline now returns a tensor with shape (B, F, C, H, W)
                video_tensor = pipe(**pipeline_args).frames

                # ComfyUI's video nodes expect a tensor of shape (F, H, W, C).
                # We need to convert from (1, F, C, H, W) to (F, H, W, C).
        
        # Squeeze the batch dimension (from 1 to none) -> (F, C, H, W)
        video_tensor_squeezed = video_tensor.squeeze(0)
        
        # Permute the dimensions to match the expected format -> (F, H, W, C)
        final_video_tensor = video_tensor_squeezed.permute(0, 2, 3, 1)

        return (final_video_tensor.float(),)



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