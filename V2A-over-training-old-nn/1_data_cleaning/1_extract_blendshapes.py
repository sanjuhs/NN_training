#!/usr/bin/env python3
"""
MediaPipe Blendshapes and Head Pose Extraction Script
Extracts 52 blendshapes + 7 head pose values (x,y,z,qw,qx,qy,qz) = 59 values per frame
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm
import sys
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceBlendshapeExtractor:
    def __init__(self):
        """Initialize MediaPipe Face Landmarker with blendshapes"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # MediaPipe blendshape names (52 categories)
        self.blendshape_names = [
            '_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 
            'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 
            'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 
            'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 
            'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen', 
            'jawRight', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 
            'mouthFrownRight', 'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight', 
            'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight', 'mouthRollLower', 
            'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight', 
            'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 
            'noseSneerLeft', 'noseSneerRight'
        ]
        
        # Download the face landmarker model if it doesn't exist
        model_path = self._download_face_landmarker_model()
        
        # Create Face Landmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
    def _download_face_landmarker_model(self):
        """Download the face landmarker model if it doesn't exist"""
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "face_landmarker_v2_with_blendshapes.task"
        
        if not model_path.exists():
            print("Downloading Face Landmarker model...")
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            try:
                urllib.request.urlretrieve(url, model_path)
                print(f"Model downloaded to: {model_path}")
            except Exception as e:
                print(f"Error downloading model: {e}")
                print("Please download the model manually from:")
                print(url)
                sys.exit(1)
        
        return str(model_path)
    
    def extract_from_video(self, video_path, output_dir="extracted_features", max_frames=None, fps_limit=30):
        """
        Extract blendshapes and head pose from video. Default fps limit is 30. 
        If this function is called with no mention of fps_limit, then it will consider the fps_limit as 30.
        If fps_limit is explicitly mentioned as None, then it will process the entire video at the fps of the video.
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save extracted features
            max_frames: Maximum number of frames to process (for testing)
            fps_limit: Target FPS for extraction (default 30)
        
        Returns:
            dict: Extracted features data
        """
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Determine effective FPS limit
        if fps_limit is None:
            fps_limit = fps
            print(f"FPS limit not provided, will process at original {fps} FPS")
        else:
            print(f"FPS limit: {fps_limit}, original video FPS: {fps}")
        
        # Calculate frame interval based on FPS limit
        if fps_limit >= fps:
            # Process every frame if fps_limit is higher than or equal to video fps
            frame_interval = 1
            effective_fps = fps
        else:
            # Skip frames to achieve target fps_limit
            frame_interval = int(fps / fps_limit)
            effective_fps = fps_limit
        
        print(f"Frame interval: {frame_interval} (processing every {frame_interval} frame(s))")
        print(f"Effective extraction FPS: {effective_fps}")
        
        # Calculate frames to process
        frames_to_process = list(range(0, total_frames, frame_interval))
        if max_frames:
            frames_to_process = frames_to_process[:max_frames]
        
        print(f"Processing {len(frames_to_process)} frames out of {total_frames} total frames...")
        
        # Storage for extracted features
        frame_data = []
        failed_frames = []
        
        # Process frames
        pbar = tqdm(total=len(frames_to_process), desc="Extracting features")
        
        for i, frame_idx in enumerate(frames_to_process):
            # Seek to specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                print(f"Failed to read frame {frame_idx}")
                failed_frames.append(frame_idx)
                # Add placeholder for failed frame
                blendshapes = {name: 0.0 for name in self.blendshape_names}
                placeholder = {
                    'frame_index': frame_idx,
                    'timestamp': frame_idx / fps,
                    'blendshapes': blendshapes,
                    'headPosition': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                    'headRotation': {'w': 1.0, 'x': 0.0, 'y': 0.0, 'z': 0.0},
                    'has_face': False
                }
                frame_data.append(placeholder)
                pbar.update(1)
                continue
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Extract features
            features = self._extract_frame_features(mp_image, frame_idx)
            
            if features:
                # Update timestamp to use original frame index for accurate timing
                features['timestamp'] = frame_idx / fps
                frame_data.append(features)
            else:
                failed_frames.append(frame_idx)
                # Add placeholder data for failed frames
                blendshapes = {name: 0.0 for name in self.blendshape_names}
                placeholder = {
                    'frame_index': frame_idx,
                    'timestamp': frame_idx / fps,
                    'blendshapes': blendshapes,
                    'headPosition': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                    'headRotation': {'w': 1.0, 'x': 0.0, 'y': 0.0, 'z': 0.0},
                    'has_face': False
                }
                frame_data.append(placeholder)
            
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        # Generate session ID based on timestamp
        import time
        session_start_time = int(time.time() * 1000)  # Current time in milliseconds
        session_id = f"session_{session_start_time}_extract"
        
        # Convert timestamps to milliseconds and add session ID
        for frame in frame_data:
            frame['timestamp'] = int(frame['timestamp'] * 1000)  # Convert to milliseconds
            frame['sessionId'] = session_id
        
        # Create final output structure similar to human head data
        output_data = {
            'sessionInfo': {
                'sessionId': session_id,
                'startTime': session_start_time,
                'targetFPS': fps_limit,
                'originalFPS': fps,
                'frameInterval': frame_interval,
                'videoPath': video_path
            },
            'frameCount': len(frame_data),
            'failedFrames': len(failed_frames),
            'failureRate': len(failed_frames) / len(frame_data) if frame_data else 1.0,
            'frames': frame_data
        }
        
        # Save single clean JSON file
        output_file = Path(output_dir) / "blendshapes_and_pose.json"
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\\nExtraction complete!")
        print(f"Original video: {total_frames} frames at {fps} FPS")
        print(f"Processed frames: {len(frame_data)} at {effective_fps} FPS (every {frame_interval} frame(s))")
        print(f"Failed frames: {len(failed_frames)} ({output_data['failureRate']:.2%})")
        print(f"Features saved to: {output_file}")
        print(f"Session ID: {session_id}")
        
        return output_data
    
    def _extract_frame_features(self, mp_image, frame_idx):
        """
        Extract features from a single frame
        
        Args:
            mp_image: MediaPipe Image object
            frame_idx: Frame index
        
        Returns:
            dict: Frame features or None if extraction failed
        """
        try:
            # Detect face landmarks and blendshapes
            detection_result = self.detector.detect(mp_image)
            
            if not detection_result.face_landmarks:
                return None  # No face detected
            
            # Get first face (we only process one face)
            face_landmarks = detection_result.face_landmarks[0]
            
            # Extract blendshapes as named dictionary
            blendshapes = {}
            
            if detection_result.face_blendshapes:
                face_blendshapes = detection_result.face_blendshapes[0]
                for i, bs in enumerate(face_blendshapes):
                    if i < len(self.blendshape_names):
                        blendshapes[self.blendshape_names[i]] = bs.score
            else:
                # Fallback if no blendshapes detected
                for name in self.blendshape_names:
                    blendshapes[name] = 0.0
            
            # Extract head pose from transformation matrix
            head_position = {'x': 0.0, 'y': 0.0, 'z': 0.0}
            head_rotation = {'w': 1.0, 'x': 0.0, 'y': 0.0, 'z': 0.0}
            
            if detection_result.facial_transformation_matrixes:
                transform_matrix = detection_result.facial_transformation_matrixes[0]
                pose_array = self._matrix_to_pose(transform_matrix)
                head_position = {'x': pose_array[0], 'y': pose_array[1], 'z': pose_array[2]}
                head_rotation = {'w': pose_array[3], 'x': pose_array[4], 'y': pose_array[5], 'z': pose_array[6]}
            
            # Calculate timestamp
            timestamp = frame_idx / 30.0  # Will be updated with actual FPS later
            
            frame_features = {
                'frame_index': frame_idx,
                'timestamp': timestamp,
                'blendshapes': blendshapes,
                'headPosition': head_position,
                'headRotation': head_rotation,
                'has_face': True
            }
            
            return frame_features
            
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            return None
    
    def _matrix_to_pose(self, transform_matrix):
        """
        Convert 4x4 transformation matrix to translation + quaternion
        
        Args:
            transform_matrix: 4x4 transformation matrix
        
        Returns:
            list: [x, y, z, qw, qx, qy, qz]
        """
        try:
            # Convert to numpy array
            matrix = np.array(transform_matrix.data).reshape(4, 4)
            
            # Extract translation (x, y, z)
            translation = matrix[:3, 3]
            
            # Extract rotation matrix
            rotation_matrix = matrix[:3, :3]
            
            # Convert rotation matrix to quaternion
            quaternion = self._rotation_matrix_to_quaternion(rotation_matrix)
            
            # Normalize quaternion
            quaternion = quaternion / np.linalg.norm(quaternion)
            
            # Clamp translation to reasonable range (±0.2 m as suggested)
            translation = np.clip(translation, -0.2, 0.2)
            
            # Return as [x, y, z, qw, qx, qy, qz]
            pose = [translation[0], translation[1], translation[2], 
                   quaternion[0], quaternion[1], quaternion[2], quaternion[3]]
            
            return pose
            
        except Exception as e:
            print(f"Error converting matrix to pose: {e}")
            return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]  # Default identity pose
    
    def _rotation_matrix_to_quaternion(self, R):
        """
        Convert 3x3 rotation matrix to quaternion [w, x, y, z]
        """
        trace = np.trace(R)
        
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2  # S = 4 * qw
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S = 4 * qx
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S = 4 * qy
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S = 4 * qz
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
        
        return np.array([qw, qx, qy, qz])

def main():
    """
    Main function to extract features from video
    """

    # root path
    root_path = Path(__file__).parent.parent
    print(f"Root path: {root_path}")

    video_path = f"{root_path}/data/test.mp4"
    
    print("Initializing Face Blendshape Extractor...")
    extractor = FaceBlendshapeExtractor()
    
    print("Starting feature extraction...")
    # For initial testing, limit to first 1000 frames (~33 seconds at 30fps)
    # Remove max_frames=1000 to process the entire video
    extraction_data = extractor.extract_from_video(
        video_path, 
        output_dir=f"{root_path}/data/extracted_features",
        # max_frames=1000  # Remove this line to process full video
    )
    
    # Print summary
    print("\\n" + "="*60)
    print("FEATURE EXTRACTION SUMMARY")
    print("="*60)
    print(f"Session ID: {extraction_data['sessionInfo']['sessionId']}")
    print(f"Video: {extraction_data['sessionInfo']['videoPath']}")
    print(f"Total frames processed: {extraction_data['frameCount']}")
    print(f"Target FPS: {extraction_data['sessionInfo']['targetFPS']}")
    print(f"Original FPS: {extraction_data['sessionInfo']['originalFPS']:.2f}")
    print(f"Duration: {extraction_data['frameCount']/extraction_data['sessionInfo']['targetFPS']:.2f} seconds")
    print(f"Failed frames: {extraction_data['failedFrames']}")
    print(f"Success rate: {(1-extraction_data['failureRate'])*100:.1f}%")
    
    # Sample feature verification
    if extraction_data['frames']:
        sample_frame = extraction_data['frames'][0]
        print(f"\\nSample frame features:")
        print(f"  Blendshapes count: {len(sample_frame['blendshapes'])}")
        print(f"  Head position: {sample_frame['headPosition']}")
        print(f"  Head rotation: {sample_frame['headRotation']}")
        print(f"  Has face: {sample_frame['has_face']}")
        print(f"  Session ID: {sample_frame['sessionId']}")
    
    print("="*60)

if __name__ == "__main__":
    main()