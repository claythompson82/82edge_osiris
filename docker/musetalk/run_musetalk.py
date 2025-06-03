import argparse
import os
import sys
import yaml
from pathlib import Path

# Add MuseTalk_code to sys.path to allow importing its modules
# Assumes this script is in /app/ and MuseTalk code is in /app/musetalk_code/
sys.path.append('/app/musetalk_code')

from scripts.inference import main as musetalk_inference_main

def create_temp_config(wav_path, img_path, result_dir, output_video_name, fps):
    """
    Creates a temporary YAML config file for MuseTalk's inference.py script.
    """
    config_data = {
        'inference_config': {
            'audio_path': wav_path,
            'video_path': img_path, # MuseTalk's video_path can be an image
            'result_dir': result_dir,
            'output_video_name': output_video_name,
            # Using default values from a typical MuseTalk config for v1.5
            'batch_size': 1,
            'fps': fps, # Default in many MuseTalk examples, can be adjusted if needed
            'L': 5, # Default from inference.sh v1.5
            'full_mask_use_org_real_face': True, # Default from inference.sh v1.5
            'mouth_mask_dilation': 0, # Default from inference.sh v1.5
            'face_smooth_filter_size': 0, # Default from inference.sh v1.5
            'save_frames': False, # Usually not needed for final output
            'skip_save_images': True # Avoid saving intermediate frames
        },
        'paths_config': {
            # Paths to models for MuseTalk v1.5
            'unet_model_path': '/app/musetalk_code/models/musetalkV15/unet.pth',
            'unet_config_path': '/app/musetalk_code/models/musetalkV15/musetalk.json',
            'sd_vae_model_path': '/app/musetalk_code/models/sd-vae',
            'whisper_model_path': '/app/musetalk_code/models/whisper',
            'face_parse_model_path': '/app/musetalk_code/models/face-parse-bisent/79999_iter.pth',
            'dwpose_model_path': '/app/musetalk_code/models/dwpose/dw-ll_ucoco_384.pth',
            'resnet_model_path': '/app/musetalk_code/models/face-parse-bisent/resnet18-5c106cde.pth', # Used by face_parse
            'syncnet_model_path': '/app/musetalk_code/models/syncnet/latentsync_syncnet.pt',
            # ffmpeg_path is usually expected to be in PATH in the Docker container
        },
        'version': 'v15' # Specify MuseTalk 1.5
    }

    temp_config_path = os.path.join(result_dir, 'temp_musetalk_config.yaml')
    os.makedirs(result_dir, exist_ok=True)
    with open(temp_config_path, 'w') as f:
        yaml.dump(config_data, f)
    
    print(f"Temporary config created at: {temp_config_path}")
    return temp_config_path

def main():
    parser = argparse.ArgumentParser(description="MuseTalk Docker Runner Script")
    parser.add_argument('--wav', required=True, help="Path to the input WAV audio file")
    parser.add_argument('--img', required=True, help="Path to the input avatar PNG image file")
    parser.add_argument('--out', required=True, help="Path for the output MP4 video file")
    parser.add_argument('--fps', type=int, default=25, help="Frames per second for the output video")
    
    args = parser.parse_args()

    # Ensure input files exist (within the Docker /pipe context)
    if not os.path.exists(args.wav):
        print(f"Error: Input audio file not found: {args.wav}")
        sys.exit(1)
    if not os.path.exists(args.img):
        print(f"Error: Input image file not found: {args.img}")
        sys.exit(1)

    output_path = Path(args.out)
    result_dir = str(output_path.parent)
    output_video_name = output_path.name

    print(f"Output directory: {result_dir}")
    print(f"Output video name: {output_video_name}")
    
    # Create a temporary directory for intermediate files if MuseTalk needs it,
    # or use the output directory directly if MuseTalk handles it well.
    # For simplicity, using a sub-directory of the output_dir for any intermediate results.
    # MuseTalk's inference script uses `result_dir` for its outputs.
    # We'll use the parent of the --out file as the result_dir and specify the output_video_name.
    
    temp_config_file = create_temp_config(args.wav, args.img, result_dir, output_video_name, args.fps)

    # Prepare arguments for MuseTalk's inference.py
    # The inference.py script takes a single argument: --config
    musetalk_args = argparse.Namespace(config=temp_config_file)
    
    print("Starting MuseTalk inference...")
    try:
        musetalk_inference_main(musetalk_args)
        print(f"MuseTalk inference completed. Output should be at {args.out}")
    except Exception as e:
        print(f"Error during MuseTalk inference: {e}")
        sys.exit(1)
    finally:
        # Clean up the temporary config file
        if os.path.exists(temp_config_file):
            os.remove(temp_config_file)
            print(f"Temporary config file {temp_config_file} removed.")

if __name__ == "__main__":
    main()
