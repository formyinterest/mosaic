# Agent instructions

## Test after every censor-logic change

Whenever you modify detection/ROI logic (`censor_policy.py`, `yolo_censor_pipeline.py`,
`mosaic_renderer.py`, or region/eye/face sizing in `mosaic.py`), render the result
against the photos in `test/prev/*.png` and show the images to the user before
saying the change is done. Don't just describe it in words.

Pattern (adjust regions/params to whatever the change touches):

```python
import glob, cv2
from yolo_censor_pipeline import YoloCensorPipeline

for path in glob.glob("test/prev/*.png"):
    img = cv2.imread(path)
    pipeline = YoloCensorPipeline("models/yolo11n-pose.onnx", mosaic_ratio=15,
                                   synchronized_frame=True)  # pass enabled_regions=... to isolate one part
    out = pipeline.process(img)
    cv2.imwrite(f"<scratchpad>/{path.split('/')[-1]}", out.overlay_bgra[:, :, :3])
```

Save outputs to the scratchpad dir, then use the Read tool to show them.

Test against **every file in `test/prev/`**, not a subset, unless the user names specific
files. If the change is about one specific region (eye/face/chest/buttocks), pass
`enabled_regions=frozenset((...))` so only that region renders — makes the diff obvious.
