#!/usr/bin/env python3
"""
Canonical MediaPipe blendshape layout used across extraction, training, and evaluation.
"""

from __future__ import annotations

BLENDSHAPE_NAMES = [
    "_neutral",
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
]

BLENDSHAPE_INDEX = {name: idx for idx, name in enumerate(BLENDSHAPE_NAMES)}

MOUTH_AND_JAW_NAMES = [
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
]

MOUTH_AND_JAW_INDICES = [BLENDSHAPE_INDEX[name] for name in MOUTH_AND_JAW_NAMES]

JAW_OPEN_INDEX = BLENDSHAPE_INDEX["jawOpen"]
MOUTH_CLOSE_INDEX = BLENDSHAPE_INDEX["mouthClose"]
MOUTH_FUNNEL_INDEX = BLENDSHAPE_INDEX["mouthFunnel"]
MOUTH_PUCKER_INDEX = BLENDSHAPE_INDEX["mouthPucker"]
SMILE_INDICES = [
    BLENDSHAPE_INDEX["mouthSmileLeft"],
    BLENDSHAPE_INDEX["mouthSmileRight"],
]
MOUTH_SMILE_LEFT_INDEX = BLENDSHAPE_INDEX["mouthSmileLeft"]
MOUTH_SMILE_RIGHT_INDEX = BLENDSHAPE_INDEX["mouthSmileRight"]

POSE_INDICES = list(range(52, 59))

CURVE_DEBUG_CHANNEL_NAMES = [
    "jawOpen",
    "mouthClose",
    "mouthFunnel",
    "mouthPucker",
    "mouthSmileLeft",
    "mouthSmileRight",
]
CURVE_DEBUG_CHANNEL_INDICES = [BLENDSHAPE_INDEX[name] for name in CURVE_DEBUG_CHANNEL_NAMES]
