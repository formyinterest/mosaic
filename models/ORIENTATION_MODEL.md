# Orientation classifier contract

Place the trained model at `models/orientation.onnx`.

- Input: `float32` NCHW tensor, shape `N x 3 x 224 x 224`
- Color: RGB
- Normalization: ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`
- Output: `N x 3` logits
- Class order: `front`, `side`, `back`
- Recommended export: dynamic batch axis `N`

When the file is absent, the application keeps using the pose-keypoint
orientation heuristic. A classifier decision below confidence `0.58` becomes
`unknown`; probabilities are smoothed with EMA alpha `0.35` before selection.
