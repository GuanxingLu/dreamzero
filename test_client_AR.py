#!/usr/bin/env python3
"""Simple test client to verify socket_test_optimized.py server works correctly.

Expected observation format for agibot embodiment:
- Video observations: video.exterior_image_1_left, video.exterior_image_2_left, video.wrist_image_left (180x320x3 uint8)
- State observations: 
    - state.joint_position (7D)
    - state.gripper_position (1D)
- Language/Annotation: annotation.language.action_text (string)
"""

import asyncio
import websockets
import numpy as np
import sys
from openpi_client import msgpack_numpy
import time
from pathlib import Path
from PIL import Image


# The websockets library by default sends a ping every 20 seconds and
# expects a pong response within 20 seconds. However, the sever may not
# send a pong response immediately if it is busy processing a request.
# Increase the ping interval and timeout so that the client can wait
# for a longer time before closing the connection.
PING_INTERVAL_SECS = 60
PING_TIMEOUT_SECS = 600


def load_images_for_step(base_path, step_idx, view_name):
    """Load images for a specific step and view.
    
    Args:
        base_path: Base directory containing step folders (e.g., 000362_11_24_22_36_27)
        step_idx: Step index (0, 1, 2, ...)
        view_name: View name (e.g., "video.exterior_image_1_left", "video.exterior_image_2_left", "video.wrist_image_left")
    
    Returns:
        numpy array of shape (num_images, H, W, C) in uint8 format
    """
    # Find folder that starts with the step index (handles timestamped folders)
    base = Path(base_path)
    matching_folders = list(base.glob(f"{step_idx:06d}_*"))
    
    if not matching_folders:
        raise FileNotFoundError(f"No folder found starting with {step_idx:06d}_* in {base_path}")
    
    # Use the first matching folder (there should only be one per step)
    step_folder = matching_folders[0] / view_name
    
    if not step_folder.exists():
        raise FileNotFoundError(f"View folder not found: {step_folder}")
    
    # Load all images in the folder (f00.png, f01.png, etc.)
    image_files = sorted(step_folder.glob("f*.png"))
    
    if not image_files:
        raise FileNotFoundError(f"No images found in {step_folder}")
    
    images = []
    for img_file in image_files:
        img = Image.open(img_file)
        img_array = np.array(img)
        
        # Ensure image is in correct format (H, W, C)
        if img_array.ndim == 2:
            # Grayscale image, convert to RGB
            img_array = np.stack([img_array] * 3, axis=-1)
        
        images.append(img_array)
    
    # Stack images: (num_images, H, W, C)
    images_array = np.stack(images, axis=0)
    
    return images_array


async def test_inference(host="localhost", port=8000, num_requests=10, delay_between_requests=5, image_base_path=None, start_step=None):
    """Send inference requests to test the server.
    
    Args:
        host: Server hostname
        port: Server port
        num_requests: Number of test requests to send
        delay_between_requests: Seconds to wait between requests (to test keep-alive)
        image_base_path: Path to directory containing step folders with real images. 
                        If None, uses dummy zero images. If provided, cycles through available steps.
        start_step: Starting step index (e.g., 29 for step 000029). If None, starts from first available step.
    """
    uri = f"ws://{host}:{port}"
    
    print(f"Connecting to {uri}...")
    
    async with websockets.connect(
        uri,
        max_size=None,
        ping_interval=PING_INTERVAL_SECS,
        ping_timeout=PING_TIMEOUT_SECS,
    ) as websocket:
        def _dummy_video(num_frames: int) -> np.ndarray:
            return np.zeros((num_frames, 180, 320, 3), dtype=np.uint8)

        def _dummy_frames_for_request(request_idx: int) -> int:
            return 1 if request_idx == 0 else 4

        # Receive metadata
        metadata_raw = await websocket.recv()
        metadata = msgpack_numpy.unpackb(metadata_raw)
        print(f"Connected! Server metadata: {metadata}")
        
        # Determine available steps if using real images
        available_steps = []
        if image_base_path:
            base = Path(image_base_path)
            if not base.exists():
                raise FileNotFoundError(f"Image base path not found: {image_base_path}")
            
            # Find all step folders (e.g., 000362_11_24_22_36_27)
            # Parse step index from folder names that start with 6 digits followed by underscore
            step_folders = []
            for d in base.iterdir():
                if d.is_dir():
                    name = d.name
                    # Check if name starts with 6 digits followed by underscore (timestamped format)
                    if len(name) >= 7 and name[:6].isdigit() and name[6] == '_':
                        step_idx = int(name[:6])
                        step_folders.append((step_idx, d))
                    # Also support old format without timestamps (just 6 digits)
                    elif name.isdigit() and len(name) == 6:
                        step_idx = int(name)
                        step_folders.append((step_idx, d))
            
            step_folders.sort(key=lambda x: x[0])
            available_steps = [step_idx for step_idx, _ in step_folders]
            
            # Filter to start from specified step if provided
            if start_step is not None:
                available_steps = [s for s in available_steps if s >= start_step]
                if not available_steps:
                    raise ValueError(f"No steps found >= {start_step:06d}")
                print(f"Starting from step {start_step:06d}. Found {len(available_steps)} step folders >= {start_step:06d}: {available_steps}")
            else:
                print(f"Found {len(available_steps)} step folders: {available_steps}")
        
        packer = msgpack_numpy.Packer()
        
        for i in range(num_requests):
            print(f"\n{'='*60}")
            print(f"Request {i+1}/{num_requests}")
            print(f"{'='*60}")
            
            # Determine which step to use (cycle through available steps)
            if image_base_path and available_steps:
                step_idx = available_steps[i % len(available_steps)]
                print(f"Loading images from step {step_idx:06d}")
                
                # Load real images for each view
                try:
                    video_exterior_image_1_left = load_images_for_step(image_base_path, step_idx, "video.exterior_image_1_left")
                    video_exterior_image_2_left = load_images_for_step(image_base_path, step_idx, "video.exterior_image_2_left")
                    video_wrist_image_left = load_images_for_step(image_base_path, step_idx, "video.wrist_image_left")
                    
                    print(f"  video.exterior_image_1_left: {video_exterior_image_1_left.shape}, dtype={video_exterior_image_1_left.dtype}")
                    print(f"  video.exterior_image_2_left: {video_exterior_image_2_left.shape}, dtype={video_exterior_image_2_left.dtype}")
                    print(f"  video.wrist_image_left: {video_wrist_image_left.shape}, dtype={video_wrist_image_left.dtype}")
                except Exception as e:
                    print(f"⚠️  Error loading images: {e}")
                    print(f"Falling back to dummy data for this request")
                    dummy_frames = _dummy_frames_for_request(i)
                    video_exterior_image_1_left = _dummy_video(dummy_frames)
                    video_exterior_image_2_left = _dummy_video(dummy_frames)
                    video_wrist_image_left = _dummy_video(dummy_frames)
            else:
                # Use dummy data
                print(f"Using dummy zero images")
                dummy_frames = _dummy_frames_for_request(i)
                video_exterior_image_1_left = _dummy_video(dummy_frames)
                video_exterior_image_2_left = _dummy_video(dummy_frames)
                video_wrist_image_left = _dummy_video(dummy_frames)
            
            # Create observation data
            # Format matches your server's expected observation space for agibot
            obs = {
                # Video observations (batched: num_images,H,W,C)
                "video.exterior_image_1_left": video_exterior_image_1_left,
                "video.exterior_image_2_left": video_exterior_image_2_left,
                "video.wrist_image_left": video_wrist_image_left,
                # State observations (batched: B,D)
                "state.joint_position": np.zeros((1, 7), dtype=np.float64),
                "state.gripper_position": np.zeros((1, 1), dtype=np.float64),
                # Language/Annotation (string)
                "annotation.language.action_text": "remove eraser from the whiteboard with right arm",
            }

            
            print(f"Sending observation data...")
            start_time = time.time()
            
            # Send observation
            await websocket.send(packer.pack(obs))
            
            print(f"Waiting for action response...")
            # Receive action
            action_raw = await websocket.recv()
            
            # Check if response is an error message (string) or action data (bytes)
            if isinstance(action_raw, str):
                print(f"❌ Server returned error:\n{action_raw}")
                raise RuntimeError("Server encountered an error")
            
            action = msgpack_numpy.unpackb(action_raw)
            
            elapsed = time.time() - start_time
            print(f"✅ Received action response in {elapsed:.2f}s")
            print(f"Action keys: {list(action.keys())}")
            
            # Print action shapes/values
            for key, value in action.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"  {key}: {value}")
            
            # Wait before next request (to test keep-alive mechanism)
            if i < num_requests - 1:
                print(f"\nWaiting {delay_between_requests}s before next request...")
                await asyncio.sleep(delay_between_requests)
        
        print(f"\n{'='*60}")
        print(f"✅ All {num_requests} requests completed successfully!")
        print(f"{'='*60}")


async def test_long_idle(host="localhost", port=8000, idle_minutes=4, image_base_path=None):
    """Test that server handles long idle periods without timeout.
    
    Args:
        host: Server hostname
        port: Server port
        idle_minutes: How long to stay idle (to test keep-alive)
        image_base_path: Path to directory containing step folders with real images.
                        If None, uses dummy zero images.
    """
    uri = f"ws://{host}:{port}"
    
    print(f"Connecting to {uri}...")
    
    async with websockets.connect(
        uri,
        max_size=None,
        ping_interval=PING_INTERVAL_SECS,
        ping_timeout=PING_TIMEOUT_SECS,
    ) as websocket:
        # Receive metadata
        metadata_raw = await websocket.recv()
        metadata = msgpack_numpy.unpackb(metadata_raw)
        print(f"Connected! Server metadata: {metadata}")
        
        print(f"\n{'='*60}")
        print(f"Testing {idle_minutes} minute idle period (to verify keep-alive works)")
        print(f"Server should send keep-alive signals every 3 minutes")
        print(f"{'='*60}\n")
        
        # Stay idle for specified time
        for minute in range(idle_minutes):
            print(f"Idle: {minute+1}/{idle_minutes} minutes...")
            await asyncio.sleep(60)
        
        # Send one request after idle period
        print(f"\n✅ Idle period complete. Sending test request...")
        
        # Load images if path is provided
        if image_base_path:
            base = Path(image_base_path)
            if base.exists():
                # Find step folders with timestamped format (e.g., 000362_11_24_22_36_27)
                step_folders = []
                for d in base.iterdir():
                    if d.is_dir():
                        name = d.name
                        # Check if name starts with 6 digits followed by underscore
                        if len(name) >= 7 and name[:6].isdigit() and name[6] == '_':
                            step_idx = int(name[:6])
                            step_folders.append((step_idx, d))
                        # Also support old format without timestamps
                        elif name.isdigit() and len(name) == 6:
                            step_idx = int(name)
                            step_folders.append((step_idx, d))
                
                step_folders.sort(key=lambda x: x[0])
                if step_folders:
                    step_idx = step_folders[0][0]
                    print(f"Loading images from step {step_idx:06d}")
                    try:
                        video_exterior_image_1_left = load_images_for_step(image_base_path, step_idx, "video.exterior_image_1_left")
                        video_exterior_image_2_left = load_images_for_step(image_base_path, step_idx, "video.exterior_image_2_left")
                        video_wrist_image_left = load_images_for_step(image_base_path, step_idx, "video.wrist_image_left")
                    except Exception as e:
                        print(f"⚠️  Error loading images: {e}")
                        print(f"Falling back to dummy data")
                        video_exterior_image_1_left = np.zeros((1, 180, 320, 3), dtype=np.uint8)
                        video_exterior_image_2_left = np.zeros((1, 180, 320, 3), dtype=np.uint8)
                        video_wrist_image_left = np.zeros((1, 180, 320, 3), dtype=np.uint8)
                else:
                    video_exterior_image_1_left = np.zeros((1, 180, 320, 3), dtype=np.uint8)
                    video_exterior_image_2_left = np.zeros((1, 180, 320, 3), dtype=np.uint8)
                    video_wrist_image_left = np.zeros((1, 180, 320, 3), dtype=np.uint8)
            else:
                video_exterior_image_1_left = np.zeros((1, 180, 320, 3), dtype=np.uint8)
                video_exterior_image_2_left = np.zeros((1, 180, 320, 3), dtype=np.uint8)
                video_wrist_image_left = np.zeros((1, 180, 320, 3), dtype=np.uint8)
        else:
            video_exterior_image_1_left = np.zeros((1, 180, 320, 3), dtype=np.uint8)
            video_exterior_image_2_left = np.zeros((1, 180, 320, 3), dtype=np.uint8)
            video_wrist_image_left = np.zeros((1, 180, 320, 3), dtype=np.uint8)
        
        packer = msgpack_numpy.Packer()
        obs = {
            # Video observations (batched: num_images,H,W,C)
            "video.exterior_image_1_left": video_exterior_image_1_left,
            "video.exterior_image_2_left": video_exterior_image_2_left,
            "video.wrist_image_left": video_wrist_image_left,
            # State observations (batched: B,D)
            "state.joint_position": np.zeros((1, 7), dtype=np.float64),
            "state.gripper_position": np.zeros((1, 1), dtype=np.float64),
            # Language/Annotation (plain string)
            "annotation.language.action_text": "test after idle",
        }
        
        start_time = time.time()
        await websocket.send(packer.pack(obs))
        action_raw = await websocket.recv()
        
        # Check if response is an error message (string) or action data (bytes)
        if isinstance(action_raw, str):
            print(f"❌ Server returned error:\n{action_raw}")
            raise RuntimeError("Server encountered an error")
        
        elapsed = time.time() - start_time
        
        print(f"✅ Request after {idle_minutes}min idle succeeded in {elapsed:.2f}s")
        print(f"✅ Keep-alive mechanism working correctly!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test client for socket_test_optimized.py server")
    parser.add_argument("--host", default="localhost", help="Server hostname (default: localhost)")
    parser.add_argument("--port", type=int, default=6000, help="Server port (default: 6000)")
    parser.add_argument("--mode", choices=["inference", "idle"], default="inference",
                        help="Test mode: 'inference' for multiple requests, 'idle' for keep-alive test")
    parser.add_argument("--num-requests", type=int, default=5,
                        help="Number of requests to send (inference mode, default: 3)")
    parser.add_argument("--delay", type=int, default=5,
                        help="Seconds between requests (inference mode, default: 5)")
    parser.add_argument("--idle-minutes", type=int, default=4,
                        help="Minutes to stay idle (idle mode, default: 4)")
    parser.add_argument("--image-path", type=str, 
                        default=None,
                        help="Path to directory containing step folders with real images (default: dreamzero checkpoint inputs)")
    parser.add_argument("--start-step", type=int, default=None,
                        help="Starting step index (e.g., 29 for step 000029). If not specified, starts from first available step.")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "inference":
            asyncio.run(test_inference(
                host=args.host,
                port=args.port,
                num_requests=args.num_requests,
                delay_between_requests=args.delay,
                image_base_path=args.image_path,
                start_step=args.start_step
            ))
        else:  # idle mode
            asyncio.run(test_long_idle(
                host=args.host,
                port=args.port,
                idle_minutes=args.idle_minutes,
                image_base_path=args.image_path
            ))
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

