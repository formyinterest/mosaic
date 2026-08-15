# 🎭 Live Local Censor

실시간으로 화면의 지정 영역 또는 YOLO Pose가 찾은 얼굴·전면 흉부·둔부를 **모자이크** 또는 **검정**으로 덮는 오버레이 프로그램입니다.  
투명 오버레이 창으로 동작하므로 화면 캡처·방송 중에도 특정 영역을 실시간으로 가릴 수 있습니다.

---

## ✨ 주요 기능

| 기능 | 설명 |
|------|------|
| **모자이크 / 검정 모드** | 지정 영역을 픽셀 모자이크 또는 완전 검정으로 실시간 처리 |
| **다중 창** | 최대 9개의 독립 오버레이 창 동시 운용 |
| **YOLO 인체 검열** | GPU(DirectML, 벤더 무관)에서 모든 사람을 한 번에 추론하고 얼굴·흉부·둔부 ROI만 투명 오버레이로 검열 -- GPU가 없으면 CPU로 자동 폴백 |
| **랜덤 잠깐 해제 (Peek)** | 설정한 랜덤 시간마다 모자이크를 잠시 해제 후 자동 복원 |
| **단축키 제어** | 키보드 단축키로 전체 ON/OFF 및 창별 설정 접근 |

---

## ⌨️ 단축키

왼손만으로 누를 수 있도록 전부 `Ctrl+Alt+` + QWERTY 왼쪽 키(1-5, Q~T, A~G, Z~B)로 통일되어 있다.

| 단축키 | 동작 |
|--------|------|
| `Ctrl+Alt+S` | 전체 모자이크 ON / OFF |
| `Ctrl+Alt+W` | 새 오버레이 영역 추가 |
| `Ctrl+Alt+F` | 랜덤 잠깐 해제 ON / OFF |
| `Ctrl+Alt+D` | 주 모니터 전체 모자이크 |
| `Ctrl+Alt+G` | YOLO 자동 검열 ON / OFF (커서가 있는 모니터) |
| `Ctrl+Alt+Q` | 트레이 메뉴 열기 (세션 프리셋 / YOLO 모드·강도·모니터 선택 / 확대 창 제어 / 종료 등) |
| `Ctrl+Alt+1~9` | 해당 번호 창 개별 설정 다이얼로그 열기 |
| `Ctrl+Alt+C` | 다음 창 설정 다이얼로그 열기 (1번부터 순환 -- 6~9번 창을 숫자키 없이 열 때) |

실행 자체도 왼손만으로: `mosaic_hotkey.ahk`(AutoHotkey)를 등록해두면 `Ctrl+Alt+R`로 실행된다.

---

## 🖱️ 오버레이 창 조작 (모자이크 OFF 상태)

- **드래그** — 창 이동
- **테두리 드래그** — 창 크기 조절 (8방향 리사이즈)

---

## 🛠️ 설치 및 실행

### 요구 사항

- Windows 10 / 11 (64-bit)
- Python 3.9 이상

### 패키지 설치

```powershell
pip install -r requirements.txt
```

### 모델 준비

`models/yolo11n-pose.onnx`는 저장소에 포함되어 있지 않습니다. Ultralytics 공식 YOLO11n-pose 모델을 그대로 ONNX로 export해서 씁니다.

```powershell
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt').export(format='onnx', imgsz=480)"
pip install --force-reinstall onnxruntime-directml
```

현재 폴더에 생성된 `yolo11n-pose.onnx`를 `models/` 아래로 옮기면 됩니다. (`orientation.onnx`는 선택 사항 -- 없으면 포즈 키포인트 기반 방향 추정으로 자동 대체되며, 직접 학습해야 하는 모델이라 별도 배포하지 않습니다. 계약은 `models/ORIENTATION_MODEL.md` 참고.)

> ⚠️ `imgsz=480`을 빼면 `yolo_pose_estimator.py`의 letterbox 입력 크기(480)와 맞지 않아 첫 추론에서 바로 shape mismatch로 죽습니다.
> ⚠️ export 과정에서 ultralytics가 순정 `onnxruntime`을 자동 설치하며 `onnxruntime-directml`의 파일을 덮어씁니다 (둘 다 같은 `onnxruntime` 모듈 경로를 씁니다). 마지막 줄로 재설치해 복구하지 않으면 `DmlExecutionProvider`가 사라지고 GPU 대신 CPU로 조용히 폴백합니다.

### 테스트 데이터 (선택)

`censor_policy.py`/`mosaic_renderer.py` 등 검출·렌더링 로직을 바꿀 때 회귀를 확인하려면 `test/prev/`에 테스트용 사진을 직접 넣으세요 (검열 대상 이미지라 저장소에 커밋하지 않습니다). `agent.md`에 안내된 대로 파이프라인을 돌리면 결과가 `test/post/`에 저장됩니다.

### 실행

```powershell
python mosaic.py
```

> ⚠️ `keyboard` 라이브러리는 전역 키 후킹을 위해 **관리자 권한** 실행을 권장합니다.

트레이 메뉴의 `YOLO 인체 영역 검열 (GPU)`에서 모니터를 선택합니다. DirectML을 지원하는 GPU(NVIDIA/AMD/Intel 무관)가 있으면 그걸 쓰고, 없으면 자동으로 CPU로 폴백합니다 (에러 없이 조용히 전환되며, 실제로 어떤 provider가 잡혔는지는 실행 시 콘솔에 `provider=...`로 출력됩니다).

---

## 📦 빌드 (PyInstaller + 설치 프로그램)

```powershell
build.bat
```

`.venv-build`에 독립된 가상환경을 새로 만들어 `requirements.txt`를 설치하고 PyInstaller로 `dist\mosaic\mosaic.exe`를 빌드합니다. PyQt5·OpenCV·onnxruntime-directml 등 번들 용량이 커서 one-file 형식은 지원하지 않습니다.

Inno Setup(`ISCC.exe`)이 설치돼 있으면 이어서 설치 프로그램(`installer_output\MosaicSetup.exe`)도 함께 빌드합니다. 없으면 `winget install JRSoftware.InnoSetup`으로 설치 후 다시 실행하세요.

> `dist/`, `build/`, `.venv-build/`, `installer_output/` 폴더는 `.gitignore`에 포함되어 있습니다.

### 시작 프로그램 등록 (선택)

설치 프로그램 자체에는 "Windows 시작 시 자동 실행" 옵션이 없습니다. 필요하면 시작 프로그램 폴더에 바로가기를 직접 등록하세요 (`installer.iss`가 단축키 바로가기를 만들 때 쓰는 것과 동일한 COM 방식).

```powershell
$ws = New-Object -ComObject WScript.Shell
$lnk = $ws.CreateShortcut("$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup\화면 모자이크.lnk")
$lnk.TargetPath = "$env:LOCALAPPDATA\Programs\Mosaic\mosaic.exe"  # 설치 프로그램으로 설치한 경우
$lnk.WorkingDirectory = Split-Path $lnk.TargetPath
$lnk.Save()
```

설치 프로그램 없이 `dist\mosaic` 폴더만 옮겨서 쓰는 포터블 방식이면 `TargetPath`를 그 폴더 안 `mosaic.exe`의 전체 경로로 바꾸면 됩니다. 해제하려면 `Win+R` → `shell:startup` 입력 후 생성된 바로가기를 지우면 됩니다.

---

## ⚙️ 창별 설정 다이얼로그 (`Ctrl+Alt+1~9`)

각 창마다 아래 항목을 개별 설정할 수 있습니다.

- **처리 모드** — 모자이크 / 검정 덮기
- **모자이크 강도** — 2(약) ~ 64(강), 슬라이더로 실시간 조절
- **랜덤 잠깐 해제**
  - 활성화 ON/OFF
  - 대기 시간 최소·최대 (초)
  - 해제 유지 시간 (초)
- **영역 자동 추적**
  - 활성화 ON/OFF
  - 검색 반경 (px) — 클수록 빠른 이동 추적, CPU 부하 증가
  - 자동 크기 조정 (줌) — 대상 확대·축소도 추적

---

## 🏗️ 코드 구조

```
mosaic.py
├── CaptureWorker   (QThread) — 60fps 모자이크 렌더링
├── TrackWorker     (QThread) — 60fps 템플릿 매칭 위치·크기 추적
├── MosaicApp       (QWidget) — 투명 오버레이 창 UI
└── WindowManager   (QObject) — 트레이 아이콘, 단축키, 다중 창 관리
```

---

## 📝 라이선스

MIT License
