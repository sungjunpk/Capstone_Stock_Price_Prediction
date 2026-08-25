# Kaggle GPU로 학습 돌리기

맥북(MPS)은 epoch당 약 19분이라 50 epoch에 16시간이 걸린다.
Kaggle 무료 GPU(T4/P100, 주당 30시간)로 **학습만** 옮긴다.

> **수집은 계속 로컬에서 한다.** 키움 API는 등록된 IP에서만 호출되는데
> Kaggle은 IP가 매번 바뀐다. 그리고 학습에는 API 호출이 전혀 없으므로
> **`.env`(API 키)를 Kaggle에 올릴 일이 없다.**

---

## 준비 (처음 한 번만)

### 0단계 — 휴대폰 인증

Kaggle은 **휴대폰 인증을 해야 GPU와 인터넷을 쓸 수 있다.** 이게 안 되어 있으면
아래 설정들이 회색으로 비활성화되어 있다.

1. https://www.kaggle.com 가입 / 로그인
2. 우측 상단 프로필 → **Settings**
3. **Phone Verification** → 번호 입력 → 문자로 온 코드 입력

### 1단계 — 로컬에서 데이터 묶음 만들기

```bash
cd ~/Desktop/Capstone_Stock_Price_Prediction
python scripts/package_data.py
```

`outputs/train_bundle.zip` (약 35MB)이 생긴다. 이 파일 하나만 올리면 된다.

### 2단계 — Kaggle에 데이터셋으로 올리기

1. https://www.kaggle.com/datasets → 우측 상단 **New Dataset**
2. `outputs/train_bundle.zip` 을 끌어다 놓는다
3. 제목: `capstone-stock-data` (아무거나 좋다)
4. 우측 하단 **Create**

업로드가 끝나면 데이터셋 페이지가 생긴다.

### 3단계 — 노트북 만들기

1. https://www.kaggle.com/code → **New Notebook**
2. 좌측 상단 **File → Import Notebook**
3. 저장소의 `notebooks/train_kaggle.ipynb` 를 올린다

### 4단계 — 노트북 설정 3가지 (제일 중요)

우측 패널에서:

| 설정 | 값 |
|---|---|
| **Accelerator** | `GPU T4 x2` (또는 P100) |
| **Internet** | `On` |
| **Input** → Add Input | 2단계에서 만든 데이터셋 검색해서 추가 |

> Accelerator를 바꾸면 세션이 재시작된다. 정상이다.

### 5단계 — 실행

노트북 셀을 **위에서부터 순서대로** 실행한다 (`Shift + Enter`).

1. GPU 확인 → `cuda: True` 가 나와야 한다
2. 코드 클론
3. 데이터 붙이기 → parquet 3개가 보이면 성공
4. **배관 점검** (`--smoke`, 1~2분) ← 전체 학습 전에 반드시 먼저
5. 전체 학습
6. 결과 확인
7. 결과 zip 다운로드

### 6단계 — 결과 가져오기

마지막 셀을 돌리면 우측 **Output** 패널에 `phase1_result.zip` 이 생긴다.
받아서 로컬 저장소의 같은 경로에 푼다:

```
outputs/checkpoints/phase1_best.pt      ← 학습된 모델
outputs/reports/*.json                  ← 실험 리포트 (VSN 피처 중요도 포함)
```

---

## 두 번째부터

데이터가 그대로면 노트북만 열어서 다시 돌리면 된다.

데이터를 새로 만들었으면 (종목 추가, 수급 피처 결합 등):

```bash
python scripts/package_data.py
```

→ Kaggle 데이터셋 페이지 → **New Version** → 새 zip 업로드

---

## 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| `cuda: False` | Accelerator가 None. GPU로 바꾸고 세션 재시작 |
| Accelerator·Internet이 회색 | 휴대폰 인증 미완료 (0단계) |
| `git clone` 이 멈춤 | Internet이 Off |
| `입력 데이터를 못 찾았다` | Add Input으로 데이터셋을 안 붙였다 |
| `CUDA out of memory` | `--batch-size` 를 512 → 256 → 128 로 낮춘다 |
| 세션이 갑자기 끊김 | 무료 GPU는 세션 12시간 / 주당 30시간. 잔여량은 우측 상단에 표시 |
| 다운로드한 체크포인트가 로컬에서 안 열림 | `torch.load(..., map_location='cpu')` — GPU에서 저장돼서 그렇다 |

## Colab을 쓰고 싶다면

`notebooks/train_colab.ipynb` 가 따로 있다. 절차는 비슷하지만
데이터를 매 세션 업로드해야 하고 세션이 더 자주 끊긴다.
**긴 학습에는 Kaggle이 유리하다.**
