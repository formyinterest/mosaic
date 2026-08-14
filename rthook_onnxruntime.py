"""Initialize ONNX Runtime before PyInstaller's Qt runtime hook loads Qt DLLs."""

import onnxruntime  # noqa: F401
