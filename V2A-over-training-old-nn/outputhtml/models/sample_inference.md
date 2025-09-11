```py

import torch
import torch.nn as nn
import numpy as np
import json
import librosa
import os
from pathlib import Path
import warnings
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# Model architecture (inline)
class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.norm2 = nn.BatchNorm1d(out_ch)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.act1(y)
        y = self.dropout1(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.act2(y)
        y = self.dropout2(y)
        return y + self.res(x)

class TCNModel(nn.Module):
    def __init__(self, in_features: int, out_features: int = 59,
                 hidden: int = 128, levels: int = 4, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        chans = [in_features] + [hidden] * levels
        blocks = []
        for i in range(levels):
            dilation = 2 ** i
            blocks.append(TemporalBlock(chans[i], chans[i + 1], kernel_size=kernel_size, dilation=dilation, dropout=dropout))
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Conv1d(hidden, out_features, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        y = self.tcn(x)
        y = self.head(y)
        y = y.permute(0, 2, 1)
        return y

# Simple inference class
class AudioInference:
    def __init__(self, model_path, device='auto'):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = device
        self.model_path = model_path
        self.sample_rate = 16000
        self.n_mels = 80
        self.hop_length = 160
        self.win_length = 400
        self.n_fft = 512

        self.model = self._load_model()

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

    def _load_model(self):
        checkpoint = torch.load(self.model_path, map_location=self.device)

        if 'model_config' in checkpoint:
            in_features = checkpoint['model_config'].get('in_features', 80)
        else:
            in_features = 80

        model = TCNModel(in_features=in_features, out_features=59)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model

    def infer(self, audio_path, max_duration=None, target_fps=30, save_json=None):
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True, duration=max_duration)

        # Extract mel features
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate, n_mels=self.n_mels,
            hop_length=self.hop_length, win_length=self.win_length, n_fft=self.n_fft
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        mel_features = log_mel.T

        # Normalize
        mel_mean = mel_features.mean(axis=0, keepdims=True)
        mel_std = mel_features.std(axis=0, keepdims=True) + 1e-6
        mel_features = (mel_features - mel_mean) / mel_std

        # Run inference
        mel_tensor = torch.FloatTensor(mel_features).to(self.device)
        batch_size = 32
        all_predictions = []

        with torch.no_grad():
            for i in range(0, len(mel_tensor), batch_size):
                end_idx = min(i + batch_size, len(mel_tensor))
                batch_mel = mel_tensor[i:end_idx].unsqueeze(0)
                batch_pred = self.model(batch_mel)
                all_predictions.append(batch_pred.squeeze(0).cpu().numpy())

        predictions = np.concatenate(all_predictions, axis=0)

        # Downsample to target fps
        original_fps = self.sample_rate / self.hop_length
        if target_fps < original_fps:
            downsample_ratio = original_fps / target_fps
            num_output_frames = int(len(predictions) / downsample_ratio)
            indices = np.linspace(0, len(predictions) - 1, num_output_frames, dtype=int)
            predictions = predictions[indices]

        # Process results
        blendshapes = predictions[:, :52]
        head_pose = predictions[:, 52:]
        frame_duration = 1.0 / target_fps
        timestamps = np.arange(len(predictions)) * frame_duration

        results = {
            'num_frames': len(predictions),
            'fps': target_fps,
            'duration': timestamps[-1] if len(timestamps) > 0 else 0,
            'timestamps': timestamps,
            'blendshapes': blendshapes,
            'head_pose': head_pose,
            'audio_path': audio_path
        }

        # Save JSON if requested
        if save_json:
            frames = []
            for i in range(results['num_frames']):
                blendshapes_dict = {}
                for j, name in enumerate(self.blendshape_names):
                    blendshapes_dict[name] = float(results['blendshapes'][i][j])

                frame_data = {
                    'frame_index': i,
                    'timestamp': int(results['timestamps'][i] * 1000),
                    'blendshapes': blendshapes_dict,
                    'headPosition': {
                        'x': float(results['head_pose'][i][0]),
                        'y': float(results['head_pose'][i][1]),
                        'z': float(results['head_pose'][i][2])
                    },
                    'headRotation': {
                        'w': float(results['head_pose'][i][3]),
                        'x': float(results['head_pose'][i][4]),
                        'y': float(results['head_pose'][i][5]),
                        'z': float(results['head_pose'][i][6])
                    },
                    'has_face': True
                }
                frames.append(frame_data)

            json_output = {
                'sessionInfo': {
                    'sessionId': f"inference_{hash(results['audio_path']) % 10000}",
                    'targetFPS': results['fps'],
                    'audioPath': results['audio_path']
                },
                'frameCount': results['num_frames'],
                'failedFrames': 0,
                'failureRate': 0.0,
                'frames': frames
            }

            Path(save_json).parent.mkdir(parents=True, exist_ok=True)
            with open(save_json, 'w') as f:
                json.dump(json_output, f, indent=2)

        return results

# Example usage
audio_file = "sample_audio/sample.wav"
model_file = "models/best_tcn_model_train_50.pth"

inference = AudioInference(model_file)
results = inference.infer(
    audio_path=audio_file,
    # max_duration=10,
    target_fps=30,
    save_json="output/inference_results.json"
)
```
