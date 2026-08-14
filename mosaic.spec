# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

root = Path(SPECPATH)
model = root / 'models' / 'yolo11n-pose.onnx'
rocblas_library = Path('/opt/rocm/lib/rocblas/library')
datas = [(str(model), 'models')]
orientation_model = root / 'models' / 'orientation.onnx'
if orientation_model.is_file():
    datas.append((str(orientation_model), 'models'))
if rocblas_library.is_dir():
    datas.append((str(rocblas_library), 'rocblas/library'))


a = Analysis(
    ['mosaic.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['onnxruntime'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(root / 'rthook_onnxruntime.py')],
    # This application performs inference directly with ONNX Runtime. Keep
    # unrelated ML/media stacks installed in the build environment out of the
    # dependency graph so local developer environments still produce a small,
    # fast bundle.
    excludes=[
        'torch', 'torchvision', 'torchaudio', 'ultralytics',
        'tensorflow', 'transformers', 'diffusers', 'gradio',
        'pandas', 'matplotlib', 'scipy', 'sklearn', 'sympy',
        'jupyter', 'IPython', 'yt_dlp', '__main__',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    name='mosaic',
    exclude_binaries=True,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='mosaic',
)
