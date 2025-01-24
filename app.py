import os
import torch
from PIL import Image
import numpy as np
from diffusers import *
from moviepy.editor import ImageSequenceClip
import cv2
import requests
import psutil
import GPUtil

class ImageToVideo:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", low_memory_mode=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.low_memory_mode = low_memory_mode
        
        # Check available memory
        self.check_memory_requirements()
        
        # Load model with memory optimizations if needed
        if self.low_memory_mode:
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                revision="fp16",
                use_safetensors=True
            ).to(self.device)
            self.pipe.enable_attention_slicing()
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
        
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

    def check_memory_requirements(self):
        """Check if system meets minimum memory requirements"""
        ram = psutil.virtual_memory()
        ram_gb = ram.total / (1024**3)
        
        print(f"\nSystem Memory Status:")
        print(f"Total RAM: {ram_gb:.1f}GB")
        print(f"Available RAM: {ram.available / (1024**3):.1f}GB")
        
        if torch.cuda.is_available():
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                print(f"\nGPU Memory Status ({gpu.name}):")
                print(f"Total VRAM: {gpu.memoryTotal/1024:.1f}GB")
                print(f"Available VRAM: {gpu.memoryFree/1024:.1f}GB")
                
                if gpu.memoryTotal < 6144:  # Less than 6GB
                    print("\nWarning: GPU memory is less than recommended minimum (6GB)")
                    print("Enabling low memory mode...")
                    self.low_memory_mode = True
        
        if ram_gb < 8:
            raise MemoryError("Minimum 8GB RAM required")

    def generate_frames(self, init_image, prompt, num_frames=24, strength_range=(0.3, 0.7)):
        """Generate frames with memory-efficient processing"""
        frames = []
        frames.append(np.array(init_image))
        
        # Clear CUDA cache between frames if in low memory mode
        if self.low_memory_mode and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        for i in range(num_frames - 1):
            strength = strength_range[0] + (strength_range[1] - strength_range[0]) * (i / (num_frames - 1))
            
            with torch.autocast(self.device):
                output = self.pipe(
                    prompt=prompt,
                    image=init_image,
                    strength=strength,
                    guidance_scale=7.5,
                    num_inference_steps=30 if self.low_memory_mode else 50
                ).images[0]
            
            frames.append(np.array(output))
            init_image = output
            
            # Clear memory after each frame in low memory mode
            if self.low_memory_mode and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return frames
    def interpolate_frames(self, frames, target_fps=30):
        """Interpolate between generated frames for smoother motion"""
        interpolated_frames = []
        
        for i in range(len(frames) - 1):
            frame1 = frames[i]
            frame2 = frames[i + 1]
            
            # Create intermediate frames
            for t in range(target_fps // len(frames)):
                alpha = t / (target_fps // len(frames))
                interpolated = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
                interpolated_frames.append(interpolated)
                
        # Add last frame
        interpolated_frames.append(frames[-1])
        return interpolated_frames
    def create_video(self, image_path, prompt, output_path, duration=6):
        """Create video with memory monitoring"""
        print(f"\nStarting video generation:")
        print(f"Duration: {duration} seconds")
        print(f"Expected frames: {duration * 30}")
        
        init_image = self.preprocess_image(image_path)
        
        # Calculate frames based on available memory
        if self.low_memory_mode:
            num_frames = max(12, int(duration * 5))  # Minimum 12 frames
            print(f"\nLow memory mode enabled:")
            print(f"Generating {num_frames} base frames")
        else:
            num_frames = int(duration * 8)
            print(f"\nGenerating {num_frames} base frames")
        
        frames = self.generate_frames(init_image, prompt, num_frames=num_frames)
        interpolated_frames = self.interpolate_frames(frames)
        
        print(f"\nCreating final video:")
        print(f"Total frames: {len(interpolated_frames)}")
        
        clip = ImageSequenceClip(interpolated_frames, fps=30)
        clip.write_videofile(output_path, verbose=False, logger=None)
        
        # Clear memory after video creation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    def create_image(self, prompts ):
        width=1080
        height=1920
        model='flux' 
        seed=None
        url = f"https://image.pollinations.ai/prompt/{prompts}?width={width}&height={height}&model={model}&seed={seed}"
        response = requests.get(url)
        filepath= "input/image.jpg"
        with open(filepath, 'wb') as file:
            file.write(response.content)
            print(f"created using POllinationAI: {filepath}")
    def preprocess_image(self, image_path):
        """Load and preprocess the image"""
        image = Image.open(image_path).convert("RGB")
        # You may need to resize or apply other transformations here
        return image
if __name__ == "__main__":
    # Example usage
    converter = ImageToVideo()
    converter.create_image(
        prompts= "Krishna smiling and glowing "
    )
    converter.create_video(
        image_path="input/image.jpg",
        prompt="krishna turned into vishnu",
        output_path="output/video_out.mp4"
    )    