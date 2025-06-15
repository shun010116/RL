# PPO with BipedalWalker-v3 (Docker)

## 📦 프로젝트 개요
이 프로젝트는 OpenAI Gym의 `BipedalWalker-v3` 환경에서 Proximal Policy Optimization (PPO) 알고리즘을 적용하여 에이전트를 학습시키고, 학습된 정책을 테스트하는 강화학습 프로젝트입니다.

본 제출물은 **Docker image**로 제공되며, 실행 환경에 관계없이 동일한 결과를 재현할 수 있습니다.

---

## 🐳 Docker 이미지 정보

- 이미지 이름: `rl-bipedalwalker`
- 기반 이미지: `python:3.10-slim`
- 학습 완료된 모델 경로: `model/ppo_bipedalwalker.pth`
- 결과 비디오 출력: `/app/videos/` 경로에 `.mp4` 파일 저장

---

## ⚙️ 사용 환경

- Python 3.10
- gym==0.26.2
- torch==2.3.0
- numpy==1.24.4
- box2d-py==2.3.5
- pygame==2.1.0
- shimmy==2.0.0
- moviepy==1.0.3

---

## 🛠 실행 방법

### 1. 이미지 빌드

```bash
docker build -t rl-bipedalwalker .
```

### 2. 이미지 로드

```bash
docker load -i rl-bipedalwalker.tar
```

### 3. 테스트 실행 (비디오 생성됨)

```bash
docker run --rm -v $PWD/videos:/app/videos rl-bipedalwalker
```

### 3. 결과 확인

- 로컬 `videos/` 디렉토리에 `.mp4` 파일이 생성됩니다.
- 해당 영상은 테스트 에피소드에서 에이전트가 작동하는 모습을 시각적으로 보여줍니다.

---

## 📝 비고

- `test.py`는 `render_mode="rgb_array"`와 `RecordVideo`를 사용하여 `.mp4` 영상 출력만 수행하며, 실제 GUI 환경이 필요하지 않기 때문에 `xvfb` 없이도 작동합니다.
- 학습 코드(`train.py`)는 별도로 실행하지 않아도 되며, 미리 학습된 모델만 테스트합니다.

---

## 🎓 학습 실행

학습을 진행하고 싶을 경우, 아래 명령어로 train.py를 실행하세요:
```bash
docker run --rm -v $PWD/model:/app/model rl-bipedalwalker python3 train.py
```

---

## 📂 디렉토리 구조

```
.
├── Dockerfile
├── train.py
├── test.py
├── requirements.txt
├── model/
│   └── ppo_bipedalwalker.pth
└── videos/
```

---

## 📁 파일 설명

| 파일명 | 설명 |
|--------|------|
| `Dockerfile` | Docker 이미지 구성 |
| `train.py` | PPO 알고리즘으로 BipedalWalker 에이전트 학습 |
| `test.py` | 학습된 모델을 불러와 테스트 및 영상 저장 |
| `model/ppo_bipedalwalker.pth` | 학습된 모델 파라미터 |
| `requirements.txt` | 의존성 패키지 목록 |
| `videos/` | (호스트와 공유) 테스트 실행 결과 비디오 저장 디렉토리 |

---

## 🙋 사용 시 주의사항

- `videos/` 폴더가 없는 경우 자동 생성되지 않으므로, 반드시 `-v $PWD/videos:/app/videos` 옵션으로 마운트해야 결과가 호스트에 저장됩니다.
- 이미지 내부의 기본 실행 명령은 `test.py`입니다.

---
