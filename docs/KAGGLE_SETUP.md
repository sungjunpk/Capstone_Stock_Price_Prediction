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

**어떤 트랙을 학습할지 먼저 정한다.** 프로파일마다 피처가 다르므로 묶음도 다르다.

```bash
cd ~/Desktop/Capstone_Stock_Price_Prediction

# (A) 기본 일봉 트랙 — 지금 운영 중인 모델
python scripts/build_features.py
python scripts/package_data.py                    # → outputs/train_bundle.zip

# (B) 횡단면 피처 트랙 (2026-08-27 신설, 피처 28개)
python scripts/build_features.py --profile xs
python scripts/package_data.py --profile xs       # → outputs/train_bundle_xs.zip (45MB)

# (C) 횡단면 + 시장대비 타깃
python scripts/build_features.py --profile xs_mr
python scripts/package_data.py --profile xs_mr    # → outputs/train_bundle_xsmr.zip
```

⚠️ **묶음 안의 파일명에 접미사가 그대로 남는다**(`panel_xs.parquet`). 캐글에서
같은 `--profile` 로 돌려야 찾을 수 있고, 체크포인트도 같은 태그(`_xs`)로 저장돼
일봉 운영 체크포인트와 섞이지 않는다.

### 2단계 — Kaggle에 데이터셋으로 올리기

1. https://www.kaggle.com/datasets → 우측 상단 **New Dataset**
2. `outputs/train_bundle.zip` 을 끌어다 놓는다
3. 제목: `capstone-stock-data` (아무거나 좋다)
4. 우측 하단 **Create**

업로드가 끝나면 데이터셋 페이지가 생긴다.

### 3단계 — 노트북 만들기

1. https://www.kaggle.com/code → **New Notebook**
2. 좌측 상단 **File → Import Notebook**
3. 저장소의 `notebooks/kaggle_all_in_one.ipynb` 를 올린다

### 4단계 — 노트북 설정 3가지 (제일 중요)

우측 패널에서:

| 설정 | 값 |
|---|---|
| **Accelerator** | `GPU T4 x2` (또는 P100) |
| **Internet** | `On` |
| **Input** → Add Input | 2단계에서 만든 데이터셋 검색해서 추가 |

> Accelerator를 바꾸면 세션이 재시작된다. 정상이다.

### 5단계 — 실행

⚠️ **먼저 `PROFILE` 셀을 1단계에서 고른 것과 맞춘다.** 노트북 위쪽에 있다.

```python
PROFILE = 'xs'      # 기본 트랙이면 ''  /  조합이면 'xs_mr'
```

⚠️ **노트북은 GitHub 에서 코드를 클론한다.** 로컬에서 코드를 고쳤으면
**push 부터 해야 반영된다** — 안 하면 예전 코드로 학습된다.

```bash
git add -A && git commit -m "..." && git push
```

그다음 노트북 셀을 **위에서부터 순서대로** 실행한다 (`Shift + Enter`).

1. GPU 확인 → `cuda: True` 가 나와야 한다
2. `PROFILE` 설정
3. 코드 클론
3. 데이터 붙이기 → parquet 3개가 보이면 성공
4. **배관 점검** (`--smoke`, 1~2분) ← 전체 학습 전에 반드시 먼저
5. 전체 학습
6. 결과 확인
7. 결과 zip 다운로드

### 6단계 — 결과 가져오기

마지막 셀을 돌리면 우측 **Output** 패널에 `phase1_result.zip` 이 생긴다.
받아서 로컬 저장소의 같은 경로에 푼다:

```
outputs/checkpoints/phase1_<해시>_xs.pt   ← 학습된 모델 (프로파일 태그가 붙는다)
outputs/reports/*.json                    ← 실험 리포트 (VSN 피처 중요도 포함)
```

가져온 뒤 로컬에서 **같은 프로파일로** 백테스트한다:

```bash
python scripts/backtest.py --profile xs
```

⚠️ `scripts/backtest.py` 는 프로파일 태그가 맞는 체크포인트만 고른다
(`tests/test_checkpoint_selection.py`). 태그가 없는 `phase1_<해시>.pt` 는
**일봉 운영용**이고 `scripts/paper_trade.py` 가 그것만 쓴다 — 실험 모델이
실주문 경로로 새는 것을 막기 위해서다.

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

## 노트북은 하나만 쓴다

예전에 있던 `train_kaggle.ipynb` / `train_colab.ipynb` 는 삭제했다.
`/kaggle/working/` 이 세션 간에 남아 있어서 옛 노트북이 낡은 클론을 그대로 재사용했고,
이미 폐기한 1.88M 설정으로 학습이 도는 사고가 있었다 (2026-08-25).

`kaggle_all_in_one.ipynb` 는 이걸 두 겹으로 막는다:
- 클론 전에 `shutil.rmtree` 로 기존 폴더를 지운다
- 클론 직후 `configs/config.yaml` 의 `d_model`/`n_layers` 를 검사해
  확정 설정(minimal, 0.33M)이 아니면 그 자리에서 멈춘다

**학습 로그에 `파라미터 1.88M` 이 뜨면 낡은 코드다.** 2번 셀부터 다시 실행할 것.
