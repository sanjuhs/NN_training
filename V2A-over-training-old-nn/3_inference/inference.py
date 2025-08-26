#!/usr/bin/env python3
"""
Inference script for Audio-to-Blendshapes TCN Model
Processes audio file and generates blendshapes + head pose predictions
"""

import torch
import numpy as np
import json
import librosa
import os
from pathlib import Path
import sys
from tqdm import tqdm
import warnings
import subprocess
import tempfile
import shutil
warnings.filterwarnings("ignore")

# Add current directory to path to import models
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))  # Add parent directory to import from V2A-over-training-old-nn

# root path
root_path = Path(__file__).parent.parent
print(f"Root path: {root_path}")

# Import the model - use absolute import from the architecture training directory
models_path = root_path / "2_architecture_training" / "models"
sys.path.append(str(models_path))

from tcn_10s_model import create_model

class AudioToBlendshapesInference:
    """
    Inference pipeline for audio-to-blendshapes conversion
    """
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu', target_fps=30.0):
        """
        Initialize inference pipeline
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
            target_fps: Target output framerate (default: 30fps to match video)
        """
        self.device = device
        self.model_path = model_path
        self.target_fps = target_fps
        
        # Audio processing parameters (must match training)
        self.sample_rate = 16000
        self.n_mels = 80
        self.hop_length = 160  # 10ms at 16kHz
        self.win_length = 400   # 25ms at 16kHz
        self.n_fft = 512
        self.mel_frame_rate = self.sample_rate / self.hop_length  # 100 Hz (internal processing)
        
        # Supported audio/video formats
        self.supported_formats = {
            # Audio formats supported by librosa
            'audio': ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'],
            # Video formats that librosa can extract audio from
            'video': ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.3gp'],
            # Formats that may need ffmpeg conversion
            'requires_conversion': ['.wma', '.opus', '.amr', '.ra', '.au']
        }
        
        print(f"Inference pipeline initialized on device: {device}")
        print(f"Audio processing parameters:")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Mel features: {self.n_mels}")
        print(f"  Internal frame rate: {self.mel_frame_rate} Hz")
        print(f"  Target output FPS: {self.target_fps} fps")
        print(f"Supported formats:")
        print(f"  Audio: {', '.join(self.supported_formats['audio'])}")
        print(f"  Video: {', '.join(self.supported_formats['video'])}")
        print(f"  With conversion: {', '.join(self.supported_formats['requires_conversion'])}")
        
        # Load model
        self.model = self._load_model()
        
    def _load_model(self):
        """Load trained model from checkpoint"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        
        print(f"Loading model from: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Create model
        model = create_model()
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"Model loaded successfully!")
        
        # Print model info if available
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            print(f"Model configuration:")
            print(f"  Architecture: {config.get('architecture', 'Unknown')}")
            print(f"  Parameters: {config.get('num_parameters', 'Unknown'):,}")
            print(f"  Receptive field: {config.get('receptive_field_seconds', 'Unknown'):.1f}s")
        
        if 'validation_metrics' in checkpoint:
            metrics = checkpoint['validation_metrics']
            print(f"Model validation metrics:")
            print(f"  Overall MAE: {metrics.get('overall_mae', 'Unknown'):.4f}")
            print(f"  Jaw correlation: {metrics.get('jaw_corr', 'Unknown'):.3f}")
        
        return model
    
    def _validate_audio_format(self, audio_path):
        """
        Validate if the audio file format is supported
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            tuple: (is_supported, format_type, file_extension)
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        file_ext = audio_path.suffix.lower()
        
        # Check if format is directly supported
        all_supported = (self.supported_formats['audio'] + 
                        self.supported_formats['video'])
        
        if file_ext in all_supported:
            format_type = 'audio' if file_ext in self.supported_formats['audio'] else 'video'
            return True, format_type, file_ext
        elif file_ext in self.supported_formats['requires_conversion']:
            return True, 'requires_conversion', file_ext
        else:
            return False, 'unsupported', file_ext
    
    def _check_ffmpeg_available(self):
        """Check if ffmpeg is available for format conversion"""
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _convert_with_ffmpeg(self, input_path, max_duration=None):
        """
        Convert audio/video file to wav using ffmpeg
        
        Args:
            input_path: Path to input file
            max_duration: Optional maximum duration to process
        
        Returns:
            str: Path to temporary wav file
        """
        if not self._check_ffmpeg_available():
            raise RuntimeError("ffmpeg is not available. Please install ffmpeg to handle this format.")
        
        # Create temporary wav file
        temp_dir = tempfile.mkdtemp()
        temp_wav = os.path.join(temp_dir, 'converted_audio.wav')
        
        try:
            # Build ffmpeg command
            cmd = ['ffmpeg', '-i', str(input_path), '-ac', '1', '-ar', str(self.sample_rate)]
            
            # Add duration limit if specified
            if max_duration:
                cmd.extend(['-t', str(max_duration)])
            
            # Output format
            cmd.extend(['-y', temp_wav])  # -y to overwrite output file
            
            print(f"Converting {input_path} to wav using ffmpeg...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")
            
            print(f"Conversion successful: {temp_wav}")
            return temp_wav
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
            raise e
    
    def _load_audio_with_fallback(self, audio_path, max_duration=None):
        """
        Load audio with format validation and conversion fallback
        
        Args:
            audio_path: Path to audio file
            max_duration: Optional maximum duration to process
        
        Returns:
            tuple: (audio_array, sample_rate, temp_file_path_if_any)
        """
        # Validate format
        is_supported, format_type, file_ext = self._validate_audio_format(audio_path)
        
        if not is_supported:
            raise ValueError(f"Unsupported audio format: {file_ext}. "
                           f"Supported formats: {', '.join(self.supported_formats['audio'] + self.supported_formats['video'] + self.supported_formats['requires_conversion'])}")
        
        temp_file = None
        
        try:
            if format_type == 'requires_conversion':
                # Convert using ffmpeg
                temp_file = self._convert_with_ffmpeg(audio_path, max_duration)
                load_path = temp_file
                print(f"Using converted file: {load_path}")
            else:
                load_path = audio_path
                print(f"Loading {format_type} file directly: {load_path}")
            
            # Load with librosa
            if max_duration and format_type != 'requires_conversion':
                # Apply duration limit for direct loading
                audio, sr = librosa.load(load_path, sr=self.sample_rate, mono=True, duration=max_duration)
            else:
                audio, sr = librosa.load(load_path, sr=self.sample_rate, mono=True)
            
            return audio, sr, temp_file
            
        except Exception as e:
            # Clean up temp file on error
            if temp_file and os.path.exists(temp_file):
                temp_dir = os.path.dirname(temp_file)
                os.remove(temp_file)
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)
            raise e
    
    def extract_audio_features(self, audio_path, max_duration=None):
        """
        Extract mel spectrogram features from audio file with enhanced format support
        
        Args:
            audio_path: Path to audio file (supports various audio/video formats)
            max_duration: Optional maximum duration to process (seconds)
        
        Returns:
            tuple: (mel_features, voice_activity, metadata)
        """
        print(f"Processing audio from: {audio_path}")
        
        # Load audio with format validation and conversion
        audio, sr, temp_file = self._load_audio_with_fallback(audio_path, max_duration)
        
        try:
            duration = len(audio) / self.sample_rate
            print(f"Audio loaded: {duration:.2f} seconds, {len(audio)} samples")
            
            # Extract mel spectrogram
            print("Computing mel spectrogram...")
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_fft=self.n_fft,
                fmin=0,
                fmax=self.sample_rate // 2
            )
            
            # Convert to log mel spectrogram (dB)
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Transpose to (time, features) format
            mel_features = log_mel_spec.T  # Shape: (time_frames, n_mels)
            
            # Normalize mel features (same as training)
            # Apply z-score normalization
            mel_mean = mel_features.mean(axis=0, keepdims=True)
            mel_std = mel_features.std(axis=0, keepdims=True) + 1e-6
            mel_features = (mel_features - mel_mean) / mel_std
            
            # Voice Activity Detection (VAD) using RMS energy
            rms_energy = librosa.feature.rms(
                y=audio,
                hop_length=self.hop_length,
                frame_length=self.win_length
            )[0]
            
            # Simple VAD threshold
            vad_threshold = np.percentile(rms_energy, 30)
            voice_activity = (rms_energy > vad_threshold).astype(float)
            
            # Ensure same length
            min_length = min(len(mel_features), len(voice_activity))
            mel_features = mel_features[:min_length]
            voice_activity = voice_activity[:min_length]
            
            # Create timestamps
            timestamps = librosa.frames_to_time(
                range(min_length),
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            print(f"Extracted features: {mel_features.shape} mel features, {len(voice_activity)} VAD frames")
            
            metadata = {
                'audio_path': audio_path,
                'duration_seconds': duration,
                'sample_rate': self.sample_rate,
                'n_frames': min_length,
                'mel_frame_rate': self.mel_frame_rate,
                'timestamps': timestamps.tolist()
            }
            
            return mel_features, voice_activity, metadata
            
        finally:
            # Clean up temporary file if it was created
            if temp_file and os.path.exists(temp_file):
                temp_dir = os.path.dirname(temp_file)
                try:
                    os.remove(temp_file)
                    if os.path.exists(temp_dir):
                        os.rmdir(temp_dir)
                    print(f"Cleaned up temporary file: {temp_file}")
                except OSError as e:
                    print(f"Warning: Could not clean up temporary file {temp_file}: {e}")
    
    def predict_blendshapes(self, mel_features, batch_size=32):
        """
        Predict blendshapes and head pose from mel features
        
        Args:
            mel_features: Mel spectrogram features (time, n_mels)
            batch_size: Batch size for processing
        
        Returns:
            numpy.ndarray: Predictions (time, 59) - 52 blendshapes + 7 pose
        """
        print(f"Running inference on {mel_features.shape[0]} frames...")
        
        # Convert to tensor
        mel_tensor = torch.FloatTensor(mel_features).to(self.device)
        
        # Process in batches to avoid memory issues
        all_predictions = []
        num_frames = len(mel_tensor)
        
        with torch.no_grad():
            for i in tqdm(range(0, num_frames, batch_size), desc="Processing batches"):
                end_idx = min(i + batch_size, num_frames)
                batch_mel = mel_tensor[i:end_idx].unsqueeze(0)  # Add batch dimension
                
                # Model expects (batch, time, features)
                batch_pred = self.model(batch_mel)  # Output: (batch, time, 59)
                
                # Remove batch dimension and append
                all_predictions.append(batch_pred.squeeze(0).cpu().numpy())
        
        # Concatenate all predictions
        predictions = np.concatenate(all_predictions, axis=0)
        
        print(f"Inference complete! Output shape: {predictions.shape}")
        
        return predictions
    
    def downsample_to_target_fps(self, predictions, timestamps, voice_activity):
        """
        Downsample predictions from 100Hz to target FPS (default 30fps)
        
        Args:
            predictions: Model predictions at 100Hz (time, 59)
            timestamps: Timestamps at 100Hz (list or array)
            voice_activity: Voice activity at 100Hz
        
        Returns:
            tuple: (downsampled_predictions, downsampled_timestamps, downsampled_vad)
        """
        if self.target_fps >= self.mel_frame_rate:
            # No downsampling needed
            return predictions, timestamps, voice_activity
        
        # Convert to numpy arrays if needed
        timestamps = np.array(timestamps)
        
        # Calculate downsampling ratio
        downsample_ratio = self.mel_frame_rate / self.target_fps
        
        # Generate target frame indices
        num_output_frames = int(len(predictions) / downsample_ratio)
        target_indices = np.linspace(0, len(predictions) - 1, num_output_frames, dtype=int)
        
        # Downsample by selecting frames at target indices
        downsampled_predictions = predictions[target_indices]
        downsampled_timestamps = timestamps[target_indices]
        downsampled_vad = voice_activity[target_indices]
        
        print(f"Downsampled from {len(predictions)} frames ({self.mel_frame_rate}Hz) to {len(downsampled_predictions)} frames ({self.target_fps}fps)")
        
        return downsampled_predictions, downsampled_timestamps, downsampled_vad
    
    def process_audio_file(self, audio_path, output_path, max_duration=None):
        """
        Complete inference pipeline: audio file -> blendshapes + pose
        
        Args:
            audio_path: Path to input audio file
            output_path: Path to save inference results
            max_duration: Optional maximum duration to process
        
        Returns:
            dict: Complete inference results in blendshapes_and_pose.json format
        """
        print(f"\n=== Audio-to-Blendshapes Inference ===")
        print(f"Input: {audio_path}")
        print(f"Output: {output_path}")
        
        # Step 1: Extract audio features
        mel_features, voice_activity, metadata = self.extract_audio_features(
            audio_path, max_duration
        )
        
        # Step 2: Run model inference
        predictions = self.predict_blendshapes(mel_features)
        
        # Step 3: Downsample to target FPS (default 30fps)
        downsampled_predictions, downsampled_timestamps, downsampled_vad = self.downsample_to_target_fps(
            predictions, metadata['timestamps'], voice_activity
        )
        
        # Step 4: Process and organize results
        num_frames, num_features = downsampled_predictions.shape
        
        # Split predictions into blendshapes and pose
        blendshapes = downsampled_predictions[:, :52]  # First 52 features
        head_pose = downsampled_predictions[:, 52:]    # Last 7 features (x,y,z,qw,qx,qy,qz)
        
        # Define the blendshape names in the correct order (from create_dataset.py)
        blendshape_names = [
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
        
        # Create frame-by-frame results in the original format
        frame_results = []
        failed_frames = 0
        
        for i in range(num_frames):
            # Convert blendshapes array to named dictionary
            blendshapes_dict = {}
            for j, name in enumerate(blendshape_names):
                blendshapes_dict[name] = float(blendshapes[i][j])
            
            # Convert timestamp to milliseconds (like original format)
            timestamp_ms = int(downsampled_timestamps[i] * 1000)
            
            frame_data = {
                'frame_index': i,
                'timestamp': timestamp_ms,
                'blendshapes': blendshapes_dict,
                'headPosition': {
                    'x': float(head_pose[i][0]),
                    'y': float(head_pose[i][1]),
                    'z': float(head_pose[i][2])
                },
                'headRotation': {
                    'w': float(head_pose[i][3]),
                    'x': float(head_pose[i][4]),
                    'y': float(head_pose[i][5]),
                    'z': float(head_pose[i][6])
                },
                'has_face': True,  # Assume face is always present in inference
                'sessionId': f"inference_{int(torch.cuda.current_device() if torch.cuda.is_available() else 0)}_{int(metadata['timestamps'][0] * 1000)}"
            }
            frame_results.append(frame_data)
        
        # Create session info (matching original format)
        import time
        session_id = f"inference_{int(time.time() * 1000)}"
        
        # Use the configured target FPS
        target_fps = self.target_fps  # Default 30 fps
        original_fps = metadata['mel_frame_rate']  # Original processing rate (100 Hz)
        
        # Compile final results in blendshapes_and_pose.json format
        results = {
            'sessionInfo': {
                'sessionId': session_id,
                'startTime': int(time.time() * 1000),
                'targetFPS': target_fps,
                'originalFPS': original_fps,
                'frameInterval': int(self.mel_frame_rate / self.target_fps),  # Downsampling ratio
                'audioPath': audio_path,
                'modelPath': self.model_path,
                'device': str(self.device),
                'inferenceMode': True
            },
            'frameCount': num_frames,
            'failedFrames': failed_frames,
            'failureRate': failed_frames / num_frames if num_frames > 0 else 0.0,
            'frames': frame_results
        }
        
        # Save results
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Inference completed successfully!")
        print(f"Results saved to: {output_path}")
        print(f"\nSummary:")
        print(f"  Session ID: {session_id}")
        print(f"  Total frames: {num_frames}")
        print(f"  Duration: {metadata['duration_seconds']:.2f}s")
        print(f"  Frame rate: {target_fps:.0f} Hz")
        print(f"  Voice activity: {np.mean(downsampled_vad):.1%}")
        print(f"  Failed frames: {failed_frames}")
        print(f"  Success rate: {(1 - failed_frames/num_frames)*100:.1f}%")
        
        # Print some sample blendshape values
        sample_frame = frame_results[len(frame_results)//2]  # Middle frame
        print(f"\nSample frame {sample_frame['frame_index']} blendshapes:")
        print(f"  jawOpen: {sample_frame['blendshapes']['jawOpen']:.3f}")
        print(f"  mouthSmileLeft: {sample_frame['blendshapes']['mouthSmileLeft']:.3f}")
        print(f"  mouthSmileRight: {sample_frame['blendshapes']['mouthSmileRight']:.3f}")
        print(f"  eyeBlinkLeft: {sample_frame['blendshapes']['eyeBlinkLeft']:.3f}")
        print(f"  eyeBlinkRight: {sample_frame['blendshapes']['eyeBlinkRight']:.3f}")
        
        return results

def list_supported_formats():
    """
    Print information about supported audio/video formats
    """
    print("\n=== Supported Audio/Video Formats ===")
    print("\n📁 Audio formats (direct support via librosa):")
    audio_formats = ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg']
    for fmt in audio_formats:
        print(f"  ✅ {fmt}")
    
    print("\n🎬 Video formats (audio extraction via librosa):")
    video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.3gp']
    for fmt in video_formats:
        print(f"  ✅ {fmt}")
    
    print("\n🔄 Formats requiring ffmpeg conversion:")
    conversion_formats = ['.wma', '.opus', '.amr', '.ra', '.au']
    for fmt in conversion_formats:
        print(f"  🔧 {fmt} (requires ffmpeg)")
    
    print("\nNote: For formats requiring conversion, please ensure ffmpeg is installed:")
    print("  macOS: brew install ffmpeg")
    print("  Ubuntu: sudo apt install ffmpeg")
    print("  Windows: Download from https://ffmpeg.org/download.html")

def main():
    """
    Main function for standalone inference with enhanced format support
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Audio-to-Blendshapes Inference with Enhanced Format Support')
    parser.add_argument('--audio', '-a', type=str, help='Path to audio/video file')
    parser.add_argument('--output', '-o', type=str, help='Output JSON file path')
    parser.add_argument('--max-duration', '-d', type=float, help='Maximum duration to process (seconds)')
    parser.add_argument('--fps', '-f', type=float, default=30.0, help='Target output framerate (default: 30)')
    parser.add_argument('--list-formats', action='store_true', help='List supported audio/video formats')
    
    args = parser.parse_args()
    
    if args.list_formats:
        list_supported_formats()
        return
    
    print("=== TCN Audio-to-Blendshapes Inference ===")
    print("Enhanced with multi-format support!")
    
    # Get root path (should point to V2A-over-training-old-nn directory)
    root_path = Path(__file__).parent.parent
    print(f"Root path: {root_path}")
    
    # Paths
    model_path = root_path / "models" / "best_tcn_model.pth"
    
    # Use command line argument or default
    if args.audio:
        audio_path = Path(args.audio)
    else:
        # Default test files
        audio_path = root_path / "data" / "some_test_data" / "15s_hmm.webm"
        print(f"No audio file specified, using default: {audio_path}")
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = root_path / "data" / "inference" / "inferred_output.json"
    
    # Check if model exists
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first using train.py")
        return
    
    # Check if audio exists
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        print("Use --list-formats to see supported formats")
        return
    
    # Create inference pipeline
    try:
        inference = AudioToBlendshapesInference(
            model_path=str(model_path),
            device='cuda' if torch.cuda.is_available() else 'cpu',
            target_fps=args.fps
        )
        
        # Run inference
        results = inference.process_audio_file(
            audio_path=str(audio_path),
            output_path=str(output_path),
            max_duration=args.max_duration
        )
        
        print(f"\n🎉 Inference pipeline completed successfully!")
        print(f"Output saved to: {output_path}")
        
        if args.max_duration:
            print(f"Note: Processed only the first {args.max_duration} seconds of the audio")
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nTip: Use --list-formats to see supported audio/video formats")

if __name__ == "__main__":
    main()
