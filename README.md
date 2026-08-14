# 🎭 Live Local Censor

실시간으로 화면의 지정 영역 또는 YOLO Pose가 찾은 얼굴·전면 흉부·둔부를 **모자이크** 또는 **검정**으로 덮는 오버레이 프로그램입니다.  
투명 오버레이 창으로 동작하므로 화면 캡처·방송 중에도 특정 영역을 실시간으로 가릴 수 있습니다.

---

## ✨ 주요 기능

| 기능 | 설명 |
|------|------|
| **모자이크 / 검정 모드** | 지정 영역을 픽셀 모자이크 또는 완전 검정으로 실시간 처리 |
| **다중 창** | 최대 9개의 독립 오버레이 창 동시 운용 |
| **YOLO 인체 검열** | ROCm GPU에서 모든 사람을 한 번에 추론하고 얼굴·흉부·둔부 ROI만 투명 오버레이로 검열 |
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
python -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt').export(format='onnx')"
```

현재 폴더에 생성된 `yolo11n-pose.onnx`를 `models/` 아래로 옮기면 됩니다. (`orientation.onnx`는 선택 사항 -- 없으면 포즈 키포인트 기반 방향 추정으로 자동 대체되며, 직접 학습해야 하는 모델이라 별도 배포하지 않습니다. 계약은 `models/ORIENTATION_MODEL.md` 참고.)

### 테스트 데이터 (선택)

`censor_policy.py`/`mosaic_renderer.py` 등 검출·렌더링 로직을 바꿀 때 회귀를 확인하려면 `test/prev/`에 테스트용 사진을 직접 넣으세요 (검열 대상 이미지라 저장소에 커밋하지 않습니다). `agent.md`에 안내된 대로 파이프라인을 돌리면 결과가 `test/post/`에 저장됩니다.

### 실행

```powershell
python mosaic.py
```

> ⚠️ `keyboard` 라이브러리는 전역 키 후킹을 위해 **관리자 권한** 실행을 권장합니다.

트레이 메뉴의 `YOLO 인체 영역 검열 (GPU)`에서 모니터를 선택합니다. YOLO 모드는 GPU가 없을 때 CPU로 폴백하지 않고 오류를 표시합니다.

---

## 📦 단독 실행 파일 빌드 (PyInstaller)

```powershell
pyinstaller mosaic.spec --noconfirm
```

ROCm/Torch 데이터가 4GB를 넘으므로 one-file 형식은 지원하지 않습니다. 빌드 완료 후 Linux/WSL에서는 `dist/mosaic/mosaic`, Windows 전용 환경에서는 `dist\mosaic\mosaic.exe`를 실행합니다.

> `dist/`, `build/` 폴더와 `*.spec` 파일은 `.gitignore`에 포함되어 있습니다.

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
