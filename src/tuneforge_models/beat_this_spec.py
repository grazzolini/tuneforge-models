"""Pinned Beat This small0 source, runtime, and release specification."""

from __future__ import annotations

CHECKPOINT_SIZE = 8_451_101
CHECKPOINT_SHA256 = "6074be2c4d490c5f6101fcc374a1ec72ae93456e23bb6019783b849f5dc7d47b"
CHECKPOINT_URL = (
    "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/small0.ckpt"
)
UPSTREAM_REPOSITORY = "https://github.com/CPJKU/beat_this"
UPSTREAM_TAG = "v1.1.0"
UPSTREAM_TAG_COMMIT = "ad7974846029835307ba19a3d5cefbf40b243041"
BEAT_THIS_VERSION = "1.1.0"
TORCH_VERSION = "2.13.0"
EXECUTORCH_VERSION = "1.4.0"
ANDROID_AAR_COORDINATE = "org.pytorch:executorch-android:1.4.0"
ANDROID_AAR_SIZE = 7_318_312
ANDROID_AAR_SHA256 = "a4a836b9fadd5b9afdf07b8533b2c3326695d04a58dd37e2cdfe709e804854fb"
ANDROID_ABI = "arm64-v8a"
ANDROID_API = 36
BACKEND = "XNNPACK"

MODEL_DIRECTORY = "beat-this-small0"
MODEL_FILENAME = "beat-this-small0.pte"
MIN_FRAMES = 13
MAX_FRAMES = 1500
TRACE_FRAMES = 813
FEATURES = 128
RTOL = 1e-4
ATOL = 1e-4
VALIDATION_FRAMES = (
    13,
    14,
    33,
    34,
    812,
    813,
    814,
    1488,
    1489,
    1499,
    1500,
)
STRUCTURED_FRAMES = 813
STRUCTURED_BEAT_COUNT = 33
STRUCTURED_DOWNBEAT_COUNT = 9

BEAT_THIS_LICENSE_FILENAME = "beat-this-1.1.0-MIT.txt"
EXECUTORCH_LICENSE_FILENAME = "executorch-1.4.0-BSD-3-Clause.txt"
