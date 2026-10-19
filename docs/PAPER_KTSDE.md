<!--
KIPS Transactions on Software and Data Engineering (KTSDE) 투고 원고
양식 출처: docs/kips/Submission_Format_KTSDE.hwp (조판값은 그 파일에서 실측)
언어: 국문 본문. 규정 2.1 에 따라 제목·저자·소속·초록·키워드·표·그림·감사의 글·
참고문헌은 영문으로 작성한다.
1쪽 구성은 실제 게재논문(정보처리학회논문지 제13권 제1호)을 따랐다.
저자 정보는 AUTHORS 표시 구간에 모아 두었고, 심사용 원고에서는 자동으로 빠진다.
⚠️ hwp 최종본 작업 시: 식 (1)~(6)이 여기서는 일반 텍스트다. 게재본은 모두 수식 개체를
쓰므로 hwp 수식 편집기로 다시 입력할 것.
-->

# A Lightweight Quantile Model for Stock Returns Combining Patch Tokenization and Variable Selection

<!-- 제목 98자. 양식 영문 지침에는 "A manuscript title should be no more than 40
characters in length." 가 있으나, docs/kips/samples/ 의 게재본 5편 영문 제목이 62~103자다
(전력수요 80 · 비디오 보간 80 · e-러닝 103 · 로봇팔 62). 문서에는 있고 실무에서는 지켜지지
않는 규정으로 보여 게재본 관행과 국문 제목과의 대응을 택했다. 사무국 문의 시 함께 확인할 것. -->

<!-- AUTHORS:BEGIN -->
Sungjun Park†⋅Jimin Kim†⋅Ohjae Kwon†⋅Haeol Song†⋅Jeongil Seo††⋅Juhan Yoo†††
<!-- AUTHORS:END -->

## ABSTRACT

Most stock forecasting models give a single answer to whether a price will rise or fall. In
practice, an investor also needs to know how far that answer can be trusted and when to hold
off. The Temporal Fusion Transformer (TFT) forecasts quantiles rather than a single value, but it
attends to every time step and is therefore expensive. Patch-based transformers are inexpensive
but produce only a single value. This paper combines the two. We equip a patch-based transformer
with the quantile forecasting of TFT and with its variable selection network, placed before the
encoder so that the channel axis collapses into one and the encoder runs a single time. A rule
withholds a decision when the prediction interval is unusually wide. On daily data for 201 KOSPI 200
stocks since 2015, the model had a lower loss than TFT in all five
repeated runs. On the validation split, which early stopping had already used, the two were
comparable; on the held-out test split its improvement over the baseline was 2.2 times that of
TFT. Training ran 3.8 times and inference 4.8 times faster. Its prediction intervals widened in
volatile markets, so withheld decisions concentrated there. The withheld forecasts carried about
1.5 times the error of those acted on, a sharper separation than a simple volatility rule
achieved. Its intervals tracked volatility less closely than those of TFT, and its directional
accuracy was lower than that of a tree-based baseline. The model is therefore better suited to
deciding when not to act than to picking stocks. Because the evaluation period informed parts of
the design, the figures are not fully out of sample. Section 6 states this and the remaining
limitations.

<!-- 281 단어 (공백 분리 기준; 제한 300). 한 문단, 표·참고문헌 번호 없음 -->

**Keywords :** Time-Series Transformer, Quantile Regression, Variable Selection Network,
Forecast Uncertainty, Abstention

## 패치 토큰화와 변수선택망을 결합한 주가 수익률 경량 분위 모형

<!-- AUTHORS:BEGIN -->
박 성 준†⋅김 지 민†⋅권 오 재†⋅송 하 얼†⋅서 정 일††⋅유 주 한†††
<!-- AUTHORS:END -->

### 요     약

주가 예측 모형은 대부분 "오를 것인가, 내릴 것인가"에 대해 하나의 답만 내놓는다. 그러나 실제 투자 판단에서는
그 예측을 얼마나 믿을 수 있는지, 언제 판단을 미뤄야 하는지도 함께 알아야 한다. 예측값과 함께 불확실성의 폭을 내는 분위 예측 모형으로 Temporal Fusion Transformer(TFT)가 있으나, 모든 시점에서 어텐션을 수행하여 계산 비용이
크다. 계산이 가벼운 패치 기반 트랜스포머는 하나의 예측값만 낸다. 이에 본 논문에서는 패치 기반 트랜스포머에
TFT의 분위 예측 기능과 변수선택망을 결합한 경량 모형을 제안한다. 변수선택망은 입력 채널을 인코더 앞에서 하나로
축약하여 인코더를 한 번만 실행시키는 장치로 쓴다. 여기에 예측 구간이 이례적으로 넓을 때 판단을 보류하는
기권 규칙을 함께 적용하였다.

2015년 이후 코스피200 구성종목 201개의 일별 자료로 평가하였다. 다섯 번의 반복 실험 모두에서 제안 모형이
TFT보다 오차가 작았다. 조기종료에 쓰인 검증구간에서는 두 모형이 대등했으나, 그 선택에 쓰이지 않은 평가
구간에서는 기준선 대비 개선폭이 2.2배 벌어졌다. 학습 시간과 추론 시간은 TFT보다 각각 3.8배, 4.8배 짧았다. 시장
변동성이 클수록 예측 구간이 넓어져, 판단 보류는 변동성 최상위 25% 후보의 20.6%, 최하위 25% 후보의 0.2%에서
일어났다. 보류된 예측의 오차는 실행된 예측의 약 1.5배로, 최근 변동성만 보고 보류하는 단순 규칙보다 큰 오차를
잘 걸러냈다. 다만 예측 구간이 변동성을 따라가는 정도는 TFT보다 낮았고, 오르내림 방향을 맞히는 능력은 트리
기반 기준모형보다 낮았다. 따라서 제안 모형은 종목을 고르는 용도보다 판단을 미룰 시점을 정하는 용도에
적합하다. 평가 기간 자료가 입력 설계와 기권 기준 선택에 참고되었으므로 보고 수치는 완전한 사후 검증 값이
아니며, 이를 포함한 한계는 6장에서 다룬다.

**키워드 :** 시계열 트랜스포머, 분위회귀, 변수선택망, 예측 불확실성, 기권

<!-- AUTHORS:BEGIN -->
† 비 회 원 : 동아대학교 컴퓨터공학과 학부생

†† 정 회 원 : 동아대학교 컴퓨터공학과 부교수

††† 정 회 원 : 동아대학교 컴퓨터공학과 조교수

Manuscript Received :

First Revision :

Accepted :

* Corresponding Authors : Jeongil Seo(jeongilseo@dau.ac.kr), Juhan Yoo(unchinto@dau.ac.kr)
<!-- AUTHORS:END -->

## 1. 서  론

주가 예측은 투자, 위험관리, 자산배분 등 여러 분야에서 꾸준히 연구되는 주제이다. 그러나 주가는 잡음이 크고
시장 상황에 따라 성격이 바뀌기 때문에, 어떤 모형도 모든 시점에서 정확할 수는 없다. 그렇다면 예측 모형에는
정확한 답을 내는 능력만큼 "지금은 확신할 수 없다"고 말하는 능력도 필요하다. 본 연구는 이 두 번째 능력에
주목한다.

금융 시계열에 딥러닝을 적용한 연구는 장단기기억 신경망과 합성곱 구조를 거쳐 트랜스포머 계열로 확장되며
폭넓게 축적되어 왔다[1]. 국내에서도 해외지수와 투자자 수급을 입력으로 삼은 등락 예측이 보고되었다[2].
그러나 이들 다수는 최신 구조를 새 자료에 이식하거나 손실함수와 초매개변수를 조정하여 지표를 개선하는
방식이며, 구조를 과업의 요구사항으로부터 도출하지는 않는다. 그 경우 모형 선택의 근거가 성능 지표 외에
남지 않아 결과를 사후에 설명하기 어렵다.

본 연구는 먼저 두 가지 운용 요구사항을 정의하고, 이를 충족하는 구조를 설계하였다. 첫째는 기권이다. 시스템은
"오를 것인가 내릴 것인가"뿐 아니라 "지금 판단해도 되는가"를 말할 수 있어야 하며, 이는 점 추정으로 정의할 수
없고 예측 분포를 요구한다. 둘째는 계산 비용이다. 201종목을 매일 평가하려면 종목당 추론이 싸야 하고, 실질적
독립 사건이 적은 자료라 모형을 키울 수도 없다. 분위 예측을 제공하는 Temporal Fusion Transformer(TFT)는 모든
시점에서 어텐션을 수행하므로 이 제약과 충돌한다.

두 요구사항은 각각 다른 구성요소를 필요로 한다. 기권에는 분위 헤드와 핀볼 손실이 필요하다. 비용에는 패치
토큰화로 어텐션 대상 시퀀스를 줄이는 것과, 채널 축을 인코더 앞에서 하나로 모으는 장치가 필요하다. 후자에는
TFT의 변수선택망을 쓴다. PatchTST를 그대로 쓰면 첫째 요구를 충족하지 못하고, TFT를 그대로 쓰면 둘째 요구를
충족하지 못한다. 제안 구조는 두 요구사항을 충족하도록 기존 구성요소를 조합한 것이며, 본 연구는 이를
정확도·비용·기권 행태의 세 축에서 검증한다.

구체적 질문은 두 가지이다. (1) 재조합 구조의 분위 정확도와 계산 비용은 두 원본과 어떻게 비교되는가.
(2) 예측 구간이 변동성에 반응하여 기권이 실제로 더 나쁜 판단을 걸러내는가.

본 논문의 구성은 다음과 같다. 2장에서는 패치 기반 트랜스포머와 분위 예측·변수선택 구조, 분위 회귀에 관한
관련 연구를 정리한다. 3장에서는 요구사항으로부터 제안 구조를 도출하고 결합 과정의 설계 결정을 기술한다. 4장에서는
자료, 분할, 기권 규칙을 포함한 실험 설계를, 5장에서는 정확도·비용·기권 행태와 예측구간의 교정
상태에 대한 실험 결과를 제시하고, 6장에서 결론과 한계를 정리한다.

## 2. 관련 연구

### 2.1 패치 기반 시계열 트랜스포머

원래의 트랜스포머[3]는 모든 시점에서 어텐션을 수행하므로 비용이 입력 길이의 제곱에 비례한다. 희소 어텐션[4]과
계열 분해[5]로 이를 줄이려는 구조들이 제안되었으나, 단순한 선형 모형이 이들을 상회한다는 보고[6]도 있다.
PatchTST[7]는 시계열을 시점이 아니라 구간(패치) 단위로 토큰화하고 다변량 입력을 채널 독립으로 처리하여,
길이 *L*의 입력을 ⌈*L*/*S*⌉개 토큰으로 축약하고 채널 수에 선형으로 확장된다. 본 연구는 120일 입력을 패치
길이 5(1거래주)·스트라이드 5로 24개 토큰으로 축약하였다. 그러나 원 구조의 출력은 점 예측이므로 1장의 첫째 요구사항을
충족하지 못한다. 또한 채널 독립을 끝까지 유지하면 인코더가 채널 수만큼 반복 실행되어 둘째 요구사항과
충돌한다.

### 2.2 분위 예측과 변수선택 구조

TFT[8]는 변수선택망으로 입력 채널별 소프트맥스 게이트를 학습하고, 게이트 잔차 신경망(gated residual
network, GRN)으로 채널별 변환을 수행하며, 정적 공변량을 문맥 벡터로 부호화하고 복수의 분위를 동시에
산출한다. 본 연구는 이 가운데 분위 헤드와 변수선택망을 부품으로 채택하고
시점 단위 연산 구조는 버렸다. 변수선택망을 쓰는 이유는 중요도 해석이 아니라, 채널 축을 소프트맥스 가중합으로
하나로 줄여 인코더 실행 횟수를 1회로 만드는 데 있다. 국내에서도 TFT를 다지평
예측에 적용한 연구[9]와 트랜스포머를 시계열 특성에 맞추어 변형한 연구[10]가 보고되었다.

### 2.3 분포 이동 정규화와 분위 회귀

RevIN(reversible instance normalization)[11]은 입력 윈도우별 인스턴스 정규화를 적용하고 출력에서 역변환한다. 종목과 시대에 따라 가격 수준의
의미가 달라지는 자료에서는 선택이 아닌 필수 요소이다. 본 연구는 이를 선별적으로 적용하였다. 횡단면 순위 채널은 정보가 전적으로
수준에 담겨 있어 윈도우 내 표준화가 그 정보를 소거하기 때문이다.

분위회귀[12]를 신경망 출력에 적용한 것이 핀볼 손실이며, 분위 *τ*에 대하여 식 (1)과 같이 정의된다.

&nbsp;&nbsp;&nbsp;&nbsp;L(*τ*; *y*, *ŷ*) = max[ *τ*(*y* − *ŷ*), (*τ* − 1)(*y* − *ŷ*) ]&nbsp;&nbsp;&nbsp;&nbsp;(1)

본 연구는 *τ* ∈ {0.1, 0.5, 0.9}를 동시에 학습한다. 분위 예측은 불확실성을 예측 결과 자체로 산출하므로,
앙상블 분산 없이 기권 규칙을 정의할 수 있다. 분위 예측의 구간이 명목 수준을 지키는지는 별도 문제이므로,
본 연구는 등각예측을 분위회귀에 결합한 방법[13]으로 교정 상태를 5.4절에서 점검한다.

### 2.4 기준 모형과 본 연구의 위치

매크로 지표를 주입하는 크로스어텐션 모듈은 본 연구에서 직접 설계하였다. 구체적으로 종목 시퀀스를
질의로, 매크로 시퀀스를 키·값으로 두었다. 외생변수를 어텐션으로 주입하는 발상 자체는
새롭지 않다. TimeXer[14]가 같은 문제를 다루었으며, 외생 변수를 변수 단위 토큰으로 요약해 내생 패치 토큰과
교차시킨다. 본 연구의 모듈은 매크로 시퀀스를 패치 단위로 유지한 채 키와 값으로 둔다는 점이 다르다.

1장에서 언급한 금융 시계열 딥러닝 연구[1] 가운데, 장단기기억 신경망으로 미국 주식의 방향을 예측한
연구[15]와 다수의 머신러닝 기법을 자산가격 결정에 적용한 연구[16]가 대표적이다. 국내에서는 해외지수와
투자자별 매매 동향을 입력으로 삼은 등락 예측[2], 가중 손실 함수를 적용한 순환 합성곱 구조[17]가 보고되었다.
본 연구는 [2]와 마찬가지로 해외지수와 투자자 수급을 입력에 포함하되, 이를 별도 경로의 크로스어텐션으로
주입하고 예측을 점이 아니라 분포로 산출한다는 점이 다르다.

공개 벤치마크 Qlib[18]에서는 그래디언트 부스팅 계열이 트랜스포머와 LSTM을 모두 상회한다고 보고된다(랭크
정보계수 0.047~0.051 대 0.041~0.044). 잘 조정된 단순 기준모형과의 공정한 비교가 부재하다는 점이 이 분야
문헌의 공통 약점이므로 본 연구는 동일 조건의 그래디언트 부스팅 기준모형을 함께 측정하였고, 그 결과는 6장의
한계에 밝혔다. 실질적 독립 사건 규모에서 미세조정이 성립하지 않는 시계열 기반모형은 비교에서 제외하였다.

## 3. 연구 방법

### 3.1 요구사항으로부터의 구조 도출

본 연구의 구성요소는 모두 <Table 1>의 좌측 요구사항으로부터 역산되었다.

Table 1. Requirements and corresponding components

| Requirement | Required property | Part chosen |
|---|---|---|
| Abstention | A distribution, not a point: q90 - q10 must exist | Quantile head, pinball loss |
| Cost: 120-day lookback x 201 stocks, daily | Sequence-length reduction | Patch tokenization (120 to 24) |
| Cost: channel-independent encoding repeats the encoder | Channel-axis reduction before the encoder | TFT variable selection network |
| Scale differs by stock and era | Non-stationary normalization | RevIN, with channel exceptions |
| Local equities track global macro | Exogenous sequence injection | Cross-attention (this work) |

### 3.2 제안 구조

제안 모형의 전체 구조는 (Fig. 1)과 같다. 입력은 세 경로로 구성된다. 종목 동적 변수(30채널 × 120일), 매크로
시퀀스(13채널), 정적 범주형 변수 3종이다. 정적 임베딩은 문맥 벡터를 형성하여 변수선택을 조건화하고,
크로스어텐션 이후 다시 더해진다. 헤드에는 마지막 패치 토큰, 즉 가장 최근 1주의 표현만이 전달된다.

![Fig. 1](images/fig1_architecture.png)

Fig. 1. Architecture of the proposed model (a) with component detail (b-e). Colors mark
the PatchTST lineage (yellow), the TFT components (blue), the macro cross-attention added
in this work (teal) and the static context path (orange).

### 3.3 결합 과정의 설계 결정

#### 1) 채널 독립의 유지 근거

패치 임베딩은 각 채널을 독립적으로 24개 토큰으로 변환한다. 채널 축을 줄이는 가장 싼 방법은 선형 합치기이며,
매크로 경로에는 실제로 그 방식을 썼다. 종목 채널에는 대신 변수선택망을 두었다. 소프트맥스 게이트가 정적
문맥으로 조건화되어 섹터와 시가총액 구간에 따라 다른 가중을 주기 때문이다. 다만 종목 채널까지 선형
합치기로 바꾼 변형과 정확도를 비교하지는 않았으므로, 게이트가 이 비용만큼의 값을 했는지는 가리지
못하였다(6장).

#### 2) 변수선택망의 배치

변수선택망은 정적 문맥 *c_s*로 조건화된 소프트맥스 게이트를 채널 축에 적용한다(식 2, 3).

&nbsp;&nbsp;&nbsp;&nbsp;*α* = softmax( GRN( [ flat(*E*) ; *c*_{s} ] ) ) ∈ ℝ^{C}&nbsp;&nbsp;&nbsp;&nbsp;(2)

&nbsp;&nbsp;&nbsp;&nbsp;*Z* = Σ *α*_{i} · GRN_{i}(*E*_{i})&nbsp;&nbsp;&nbsp;&nbsp;(3)

변수선택망을 인코더 뒤에 두면 인코더가 채널 축을 그대로 받아 채널 수만큼 반복 실행된다. 앞에 두면 채널이
먼저 하나로 축약되어 인코더가 1회만 실행된다. 이전 설정(17채널)에서 순전파를 실측한 결과 259ms에서 52ms로
5배 단축되었다.
변수선택망이 패치 임베딩을 입력받아 채널을 가중하므로 *α*는 각 채널의 사용 비중으로 남는다. 다만 본 연구는
이 값을 매매 근거로 제시하지 않는다(6장).

#### 3) 매크로 크로스어텐션의 방향

종목 표현을 질의로, 매크로 부호화를 키·값으로 설정하였다(식 4).

&nbsp;&nbsp;&nbsp;&nbsp;CrossAttn(*Z*, *H*_{m}) = softmax[ (*Z W*_{Q})(*H*_{m} *W*_{K})^{T} ∕ √*d*_{k} ] (*H*_{m} *W*_{V})&nbsp;&nbsp;&nbsp;&nbsp;(4)

매크로를 질의로 두면 출력이 매크로 시퀀스의 형상을 따르게 되어 종목별 예측이 산출되지 않으므로, 방향을
반대로 고정하였다. 매크로 경로에는 변수선택망을 두지 않고 13채널을 하나로 합치는 선형 변환을 썼다. 채널별로 부호화하면
비용이 13배 증가하는 반면, 매크로는 크로스어텐션의 키와 값으로만 쓰이므로 채널별 가중이 필요하지 않다.

#### 4) 분위 단조성의 구조적 보장과 헤드 초기화

분위 교차를 손실 벌점이 아니라 누적 소프트플러스 구조로 원천 차단하였다(식 5).

&nbsp;&nbsp;&nbsp;&nbsp;*q*_{10} = *o*_{1},&nbsp;&nbsp; *q*_{50} = *q*_{10} + softplus(*o*_{2}),&nbsp;&nbsp; *q*_{90} = *q*_{50} + softplus(*o*_{3})&nbsp;&nbsp;&nbsp;&nbsp;(5)

교차가 발생하면 기권 폭이 음수가 되어 하위 규칙 전체가 무효화되므로 구조적 보장이 필요하다. 5.1절의 세 모형
모두 실측 교차율은 0.000이었다.

헤드의 초기화도 성능에 영향을 주었다. 기본 초기화에서는 softplus(0) = 0.693이라 분위 간격이 윈도우 변동성
*σ* 단위로 0.693씩 고정된다. 분위 출력에는 RevIN이 구한 *σ*, 즉 일간 수익률 채널의 윈도우 표준편차가 다시
곱해지므로 초기 q10~q90 폭은 1.386*σ*이고,
학습구간 평균 *σ*로 환산하면 0.035이다. 5일 수익률의 전체 분위 폭 0.124의 0.28배로, 목표 분포와 어긋난
지점에서 학습이 시작된다. 이 경우 검증 손실은 0.013993으로 기준선 0.013285보다 5.3% 열등하였다(이전 설정의
측정값이다). 최종층 편향을 학습자료의 무조건부 분위수로 설정하여
해소하였으며, 최종층 가중치는 작게 두되 0이면 기울기가 인코더까지 전달되지 않으므로 0이 아닌 작은 값으로 두었다.

### 3.4 최종 설정

<Table 2>는 이후 모든 실험에 사용한 설정이다. 정규화는 2.3절에서 기술한 방식을 따른다.

Table 2. Final model configuration

| Item | Value |
|---|---|
| Lookback / patch / stride | 120 / 5 / 5, giving 24 tokens |
| d_model / heads / layers / d_ff / dropout | 32 / 2 / 1 / 64 / 0.5 |
| Quantiles | 0.1, 0.5, 0.9 |
| Dynamic / macro / static inputs | 30 channels / 13 channels / 3 categoricals |
| Normalization | RevIN, applied selectively by channel |
| Parameters | 0.50M |

## 4. 자료와 실험 설계

### 4.1 자료

분석 자료는 코스피200 구성종목 201개의 2015년 이후 일별 자료이며, 종목당 최대 2,857거래일이다. 지수와
투자자 수급 계열을 포함해 모두 키움증권 Open API로 수집하였다. 거래정지일은 거래량 0과 시가=고가=저가=종가로
입력되는데, 제거하지 않으면 허위의 0 수익률이 생겨 변동성 추정을 왜곡하므로 지표 산출 이전에 제거하였다.
종목별 거래일이 상이하여(공통 1,887일, 전체 2,857일) 공통 달력을 강제하면 34%의 자료가 소실되므로, 윈도우는
각 종목의 자기 행 위에서 구성하였다. 모형과 평가 코드는 교신저자에게 요청하면 제공하며, 원자료는 증권사 API 이용
약관상 재배포할 수 없다.

### 4.2 변수와 목표변수

<Table 3>은 입력 변수의 구성이다. 동적 30채널 가운데 기술적 지표 8채널은 수렴·발산 지표(3), 상대강도지수,
상품채널지수, 볼린저 밴드(2), 실질변동폭(ATR)이다.

Table 3. Input channels

| Group | Channels | n |
|---|---|---|
| Dynamic (stock) | returns and momentum (5), trading activity (4), moving-average gaps (3), technical indicators (8), index-relative (6), investor flows (4) | 30 |
| Macro sequences | KOSPI and KOSDAQ derivatives, overseas index exchange-traded funds | 13 |
| Static categoricals | sector, market-cap bucket, weekday | 3 |

정적 범주형 세 개는 섹터, 시가총액 구간, 요일이다. 시가총액 구간은 학습구간 종료일의 시가총액으로 고정하였다(6장). 요일은 개체의 고정 속성이 아니라 시점마다 달라지는 사전 알려진 공변량인데, 구현에서는
윈도우 마지막 날의 요일을 정적 입력으로 넣었다. TFT 용어의 known future input 에 해당하므로 분류가 정확하지
않다.

가격 수준은 직접 사용하지 않는다. 이동평균은 종가 대비 괴리율로, 수렴·발산 지표와 ATR은 종가로 나누어,
유계 지표는 재척도하여 입력한다. 목표변수는 코스피200 대비 초과분으로 정의한 5일 누적 로그수익률이다(식 6). 1·2·3일 보조 지평은 패널에
생성해 두었으나 본 실험의 세 모형은 사용하지 않았으며, 출력은 5일 지평의 세 분위뿐이다.

목표를 초과수익으로 정렬하는 과정에서 설계 하나가 바뀌었다. 모형에 지수 대비 상대 성과를 나타내는 채널이
없으면 검증구간과 검정구간의 방향 예측력 부호가 엇갈렸다. 상대강도와 베타, 그 횡단면 순위를 추가하여 동적
채널을 24개에서 30개로 늘린 뒤 두 구간의 부호가 일치하였다.

&nbsp;&nbsp;&nbsp;&nbsp;*y*_{t} = Σ_{k=1..5} *r*_{t+k} − Σ_{k=1..5} *r*^{index}_{t+k}&nbsp;&nbsp;&nbsp;&nbsp;(6)

### 4.3 분할과 기준선

분할은 종목이 아니라 날짜 기준 전역 분할이다. 2022년까지 학습, 2023년 검증, 2024년 이후 검정이며, 목표가
5일 앞을 보므로 각 경계에 5일 격리구간을 둔다. 정규화 통계는 학습 구간 행에서만 추정한다.

윈도우는 120일 룩백이 해당 구간 안에 온전히 들어갈 때만 생성된다. 입력이 앞 구간에 걸치는 것 자체는 5일
격리구간이 목표값 중첩을 막으므로 누수가 아니나, 그 경우 검증 표본의 입력 대부분이 학습에서 이미 관측된 행이
되어 암기가 검증 성능을 부풀릴 여지가 남는다. 본 연구는 보수적으로 룩백 전체를 구간 안에 넣었고, 그 대가로
각 구간의 처음 120거래일이 소모되어 검증 표본이 122거래일로 줄었다. 검증 표본은 2023년 7월 3일부터
12월 28일까지 122거래일·190종목·23,086건, 검정 표본은 2024년 7월 3일부터 2026년 10월 8일까지
550거래일·198종목·106,490건이다. 자료는 2026년 10월 16일 수집분으로 동결하였고, 파일 해시와 구간별 윈도우
수를 기록하여 이후 모든 수치를 같은 표본에서 산출하였다.

인접 표본은 120일 중 119일의 입력과 5일 중 4일의 목표를 공유한다. 명목 학습 행 264,653건 가운데 비중첩
표본은 약 52,900건이고, 같은 날짜의 201개 종목이 시장 공통요인으로 묶이므로 날짜 단위로 세면 약 1,900거래일이다. 목표가 5일씩
겹치는 것까지 반영해 비중첩으로 세면 약 380개로 줄어든다. 이전 설정(146종목·17채널)의 탐색에서 더 큰 설정은 1~2회차에 과적합되었고, 같은 구조를 30채널로 확장한
결과가 0.50M이다.

모든 실험은 학습 자료의 무조건부 분위수, 즉 입력을 무시하고 세 상수를 출력하는 예측기를 기준선으로 평가한다.
이를 상회하지 못하면 초매개변수 조정 문제가 아니라 변수·목표 설계의 실패로 간주한다.

### 4.4 기권 규칙

비교하는 모든 모형에 동일한 규칙을 적용한다. 예측 구간의 폭이 최근 분포의 일정 분위를 넘으면 기권한다.
'최근 분포'는 판단일부터 거슬러 달력 90일, 거래일로는 약 60일 안에 들어온 폭을 모은 것이며, 종목별로 나누지
않고 전 종목을 합친 분포다. 구간 전체의 폭 분포를 쓰면 뒤쪽 날짜의 폭이 앞쪽 날짜의 기권 판단에 흘러들어
look-ahead가 생기기 때문이고, 모의투자에서 실제로 쓰는 방식과도 같다.

임계값을 절대값으로 두었던 초기 설정은 학습이 2022년 하락장에서 끝난 탓에 구조적으로 도달 불가능해져
2.1년간 기권율이 95.8%에 달하였다. 최근 분포의 상대 순위로 전환한 것은 그 결함을 교정한 결과이다.

## 5. 실험 결과

### 5.1 동일조건 정확도 비교

세 모형은 자료·분할·학습절차·조기종료 규칙을 공유한다. 두 가지 제약을 먼저 밝힌다. PatchTST 계열
대조군은 원 구조에 해당 개념이 없어 매크로와 정적 공변량을 입력받지 못하므로, 본 비교는 원본을 그대로 쓸 때
무엇을 포기하게 되는가의 비교이다. 초매개변수는 제안 모형에 맞추어 조정된 값을 대조군에 동일하게
적용하였으므로 대조군의 최적값이 아니다. 기준선은 세 모형 모두 0.012175로 동일하게 산출되어 통제가
작동하였음을 수치로 확인할 수 있다. TFT 원본과 제안 모형은 시드 42~46의 다섯 번씩을 같은 GPU·같은
혼합정밀도 설정으로 학습하여 검증·검정 두 구간의 평균과 표준편차를 보고한다. 분위 헤드만 부착한 PatchTST는 기준선 대비 격차가
+0.26%로 시드 변동의 범위를 크게 벗어나므로 단일 실행값을 보고한다. <Table 4>가 예측 정확도이다.

Table 4. Forecast accuracy (three-quantile pinball loss, mean over five seeds 42-46 with the standard deviation). TFT and the proposed model were trained on one GPU under identical mixed-precision settings. The unconditional-quantile baseline is 0.012175 on validation and 0.018753 on test. The PatchTST row is a single run on a Mac MPS backend and its test figure was not computed. The original point-forecast PatchTST is left out: scored as three identical quantiles it falls 47.79% below the baseline, which reflects the loss function rather than a comparison

| Model | Params | Validation | vs baseline | Test | vs baseline | Best epoch |
|---|---|---|---|---|---|---|
| PatchTST + quantile head | 0.035M | 0.012143 | +0.26% | - | - | 5 |
| TFT, original | 0.687M | 0.011935 ± 0.000020 | +1.97% | 0.018325 ± 0.000091 | +2.28% | 1-2 |
| **Ours (PatchTST x TFT)** | 0.500M | 0.011889 ± 0.000012 | +2.35% | **0.017826 ± 0.000059** | **+4.95%** | 2-10 |

결과는 세 가지로 해석된다. 첫째, 분위 헤드만 부착해서는 충분하지 않다. 해당 설정은 +0.26%로 상수 예측을
간신히 상회하는 데 그쳤으며 변수선택·정적 문맥·매크로 입력이 결여되어 있다.

둘째, 검증구간에서 TFT 원본과 제안 모형은 대등하다. 격차는 핀볼 손실 0.000046으로 기준선 대비
개선폭(0.000286)의 16%에 그친다. 다섯 시드에서 두 분포의 범위는 겹치지 않고, 시드를 짝지어 보아도 다섯 쌍
모두 제안 모형이 낮았다(부호검정 *p* = 0.031 단측). 평균 차이에 대한 Welch 검정은 *t* = 4.47(자유도 6.5,
*p* = 0.004 양측)이다. 초기화 변동을 통제하면 제안 모형이 체계적으로 낮은 쪽에 있다는 뜻이나 격차의 절대
크기가 작으므로, 본 연구는 검증구간의 결과를 우위가 아니라 대등하거나 근소하게 낫다는 수준으로 해석한다.

셋째, 검정구간에서는 격차가 분명해진다. 검증구간은 조기종료와 최적 회차 선택에 이미 쓰였으므로 세 모형
모두 낙관적으로 나오는데, 그 선택에 쓰이지 않은 검정구간에서 기준선 대비 개선폭이 제안 모형 +4.95%,
TFT 원본 +2.28%로 2.2배 벌어진다. 두 모형의 격차는 핀볼 손실 0.000499로 제안 모형 개선폭(0.000927)의
54%이며, 다섯 시드의 범위도 겹치지 않는다(0.017739~0.017895 대 0.018205~0.018407). 시드를 짝지은 부호검정은
다섯 쌍 모두 제안 모형이 낮고(*p* = 0.031 단측), Welch 검정은 *t* = 10.33(자유도 6.9, *p* < 0.001 양측)이다.

같은 표본 위의 날짜별 손실 계열에 Diebold-Mariano 검정을 적용한 결과도 같은 방향이다. 검정구간에서는
다섯 시드 모두 유의하였다(Harvey-Leybourne-Newbold 보정 후 *p* = 0.022, 0.001, 0.009, 0.003, 0.000.
550일, Newey-West 4시차). 반면 검증구간에서는 다섯 시드 모두 유의하지 않았다(*p* = 0.16~0.56). DM 검정은
날짜를 표본으로 보므로 122거래일인 검증구간에서는 검정력이 약하고, 550거래일인 검정구간에서는 격차를
가려낸다. 시드 단위 검정과 날짜 단위 검정이 검정구간에서 함께 유의하다는 점이 이 비교의 근거다.

격차가 검정구간에서 벌어지는 이유는 학습 곡선에 있다. TFT는 다섯 시드 모두 1~2회차에서 최적값에 도달한 뒤
과적합으로 돌아섰고, 제안 모형의 최적 회차는 2~10회차에 분포하였다. 다만 초매개변수가 제안 모형에 맞추어
조정된 값이므로 이 격차의 일부는 대조군의 설정에서 올 수 있다(6장). 비교의 실질은 <Table 5>의 비용으로
넘어간다.

### 5.2 계산 비용

Table 5. Computational cost measured on one device (Mac MPS, batch 64, 120-day input, 201 stocks). Seconds per epoch is the median training-step time over 40 batches times the 4,068 batches that one epoch takes over the 260,339 training windows of Section 4.3. The universe pass is the median of 21 forward passes over 201 stocks, and latency per stock is that value divided by 201 rather than an independent measurement. Activation is the device allocation added by a forward pass that retains the backward graph. Two independent runs agreed within 0.8%

| Model | s / epoch | Latency / stock | Universe pass | Activation | Param memory |
|---|---|---|---|---|---|
| PatchTST (+ quantile head) | 108.9 | 0.092 ms | 18.6 ms | 172.8 MB | 0.14 MB |
| TFT, original | 542.0 | 0.693 ms | 139.4 ms | 679.1 MB | 2.75 MB |
| Ours | 141.3 | 0.143 ms | 28.7 ms | 132.9 MB | 2.00 MB |

대등한 정확도에서 제안 모형은 TFT 원본 대비 학습 3.8배, 종목당 추론 4.8배, 활성값 메모리 5.1배의 절감을
보였다. 패치 토큰화가 어텐션 대상 시퀀스를 120에서 24로 5배 줄이고 변수선택망의 선행 배치가 채널 반복을
제거한 결과이며, 모수 1.4배 차이는 부차적이다. 실질적 절감은 활성값과 실제 수행 시간에서 발생한다. 표의 세
행은 같은 장비·같은 배치에서 다시 측정한 값이다. 제안 모형은 외부 GPU에서 29.3초/epoch로도 학습되었다.

(Fig. 2)는 <Table 4>의 검증구간 정확도와 <Table 5>의 비용을 한 평면에 놓은 것이다. 지연시간만 보면 분위
헤드 모형이 더 왼쪽에 있으나, 정확도와 비용을 함께 보면 제안 모형이 파레토 전선 위에 있다.

![Fig. 2](images/fig2_cost_accuracy.png)

Fig. 2. Forecast accuracy against inference cost. The vertical axis is the pinball
improvement over the unconditional-quantile baseline on validation (<Table 4>), the
horizontal axis is per-stock inference latency, and the marker area is activation memory
for a 120-day input (<Table 5>). The original point-forecast PatchTST is omitted because
its pinball loss falls far below the plotted range.

### 5.3 기권 행태

세 분위 모형에 4.4절의 동일한 기권 규칙을 적용하고 검정구간 전체에서 집계하였다. 점 예측 모형은 임계할
폭이 없어 기권이 정의되지 않으므로 제외하였다. 자료 수집이
이어져 4.3절의 검정표본에 2거래일이 더해진 550거래일·106,490 윈도우가 집계 대상이다. 기권이 어디서
일어나는가와 보류된 판단이 실제로 더 나빴는가를 함께 본다.

Table 6. Abstention on the test split (550 days, 106,490 windows). The threshold is the 95th percentile of the width distribution over the trailing 90 calendar days, pooled across all stocks; Q1 and Q4 are the lowest and highest quartiles of realized volatility at decision time. Pinball loss is reported separately for abstained and retained candidates. The abstention figures in this table, and the interval figures in Section 5.4, come from one checkpoint per model: the two baselines were trained on a Mac MPS backend and the proposed model on an external GPU, so the devices are mixed here and differ from those behind <Table 4>

| Model | Abstain | Q1 | Q4 | Pinball, abstained | Pinball, retained |
|---|---|---|---|---|---|
| PatchTST + quantile | 4.70% | 0.40% | 12.91% | 0.02387 | 0.01806 |
| TFT, original | 5.33% | 0.00% | 17.32% | 0.02617 | 0.01768 |
| **Ours** | 5.99% | 0.20% | **20.57%** | **0.02646** | **0.01726** |

<Table 6>은 기권 요구사항이 규칙이 아니라 구조에 의해 충족되었음을 보여준다. 세 행의 기권 규칙은
동일하며 예측 분포만 다르다. 분위 헤드를 부착하면 기권이 작동하기 시작하고, 변수선택·정적 문맥·매크로
입력이 더해지면 최상위 변동성 분위의 기권율이 20.57%인
반면 최하위 분위는 0.20%에 머물러 기권이 고변동 구간에 집중된다.

보류된 판단이 실제로 더 나빴는지는 따로 재야 한다. 제안 모형이 기권한 6,381건의 핀볼 손실은 0.02646으로
남긴 100,109건의 0.01726보다 1.53배 크다. 날짜 단위로 짝지은 손실차는 +0.00804(*t* = 19.74, 550일)이며,
분위 헤드만 부착한 모형과 TFT 원본에서도 같은 방향이다(*t* = 13.44, 18.61). 반면 중앙값 부호의 방향
적중률은 기권 0.553 대 유지 0.548로 거의 같다. 보류된 판단은 방향을 틀리기 쉬운 판단이 아니라 오차의
크기가 큰 판단이며, 이는 기권이 폭에 걸려 있다는 설계와 일치한다.

다만 핀볼 손실은 변동성이 큰 구간에서 원래 크므로, 고변동 구간을 피하는 규칙이라면 무엇이든 이 비가 1을
넘는다. 규칙과 분위를 그대로 두고 점수만 예측 폭에서 직전 20일 실현변동성으로 바꿔 끼운 통제군과
대조하였다. 통제군의 비는 1.343(기권율 5.66%)으로 제안 모형의 1.533(5.99%)보다 낮고, 남긴 후보의
손실도 0.01747로 제안 모형의 0.01726보다 높다. TFT 원본도 1.480으로 자기 통제군(1.358)을 상회하였으나, 분위 헤드만
부착한 모형은 1.322로 통제군(1.366)에 미치지 못하였다. 통제군의 값이 1.343과 1.366으로 다른 것은 통제군을
모형마다 따로 맞춘 탓이 아니다. 실현변동성 점수는 모형과 무관하여 통제군의 기권율이 세 모형 모두 5.66%로
같고, 보류된 후보의 손실만 각 모형의 예측으로 재기 때문이다.

기권율 자체가 다르면 보류 집단이 분포의 꼬리로 더 들어간 효과가 섞인다. 제안 모형은 5.99%, 통제군은
5.66%였으므로 기권율을 맞춘 비교를 따로 수행하였다. 이 비교에는 4.4절의 운용 규칙이 아니라 비교를 위해
별도로 정의한 규칙을 쓴다. 날짜마다 두 점수의 날짜별 횡단면 순위 상위 r%를 보류시키면 같은 날 같은 개수를
보류하게 되어 기권율이 일치한다. 보류와 유지의 핀볼 비는 r = 2, 5, 6, 10, 20, 30%에서 각각 1.574 대 1.485,
1.484 대 1.395, 1.475 대 1.368, 1.401 대 1.307, 1.333 대 1.258, 1.278 대 1.216이었다. 여섯 수준 전부에서
폭 기준이 높았고, 날짜 블록 부트스트랩 1,000회의 95% 신뢰구간도 여섯 수준 모두 0을 포함하지 않았다
(6%에서 차이 +0.107, 구간 [+0.079, +0.138]). TFT 원본은 5% 이상에서 같은 방향이나 2%에서는 구간이 0을
포함하였고, 분위 헤드만 부착한 모형은 여섯 수준 전부에서 통제군보다 낮았다. 기권율을 맞춘 뒤에도 제안
모형의 폭은 단순 변동성 지표보다 나은 보류 기준이며, 그 이점은 분위 헤드를 부착하는 것만으로는 생기지
않았다.

전체 기권율의 수준을 정하는 임계값 분위는 검정구간을 참조하여 선택되었으므로, 5.99%라는 절대 수준은
표본 밖 수치가 아니다. 다만 분위를 95에서 90, 75로 낮추면 제안 모형의 전체 기권율이 5.99%, 11.61%,
28.17%로 오르는 동안 최상위 변동성 분위의 기권율도 20.57%, 36.42%, 69.36%로 함께 올라 분위별 집중이
유지되며, 세 분위 모두에서 최상위 분위 기권율의 모형 간 순서(제안 모형 > TFT 원본 > 분위 헤드)도 보존된다. 임계값 선택이
기권 행태에 관한 결론을 만들어낸 것은 아니다.

기권 임계값의 변화도 같은 방향을 가리킨다. 검정구간의 마지막 12개월(244거래일)에서 제안 모형의
임계값은 0.189에서 0.329로 74% 상승한 반면 TFT 원본은 0.175→0.187로 7% 오르는 데 그쳤고, 분위 헤드만 부착한 모형은 0.189→0.166으로 오히려
내렸다. 임계값이 직전 90일 폭 분포의 분위이므로 이는 제안 모형의 폭이 시장 변동성 변화에 더 크게 반응한다는
뜻이다. 같은 12개월 구간의 평균 폭 자체도 제안 모형이 0.1615로 TFT(0.1298)와 분위 헤드 모형(0.1274)보다
넓다(5.4절의 0.1410은 검정구간 전체 값이다). 변동성이 오른 구간에서 폭이 함께
넓어져 기권이 늘어난 결과가 <Table 6>의 Q4 기권율이다. 폭의 반응성 자체를 직접 비교한 실험은 수행하지
않았다.

### 5.4 예측구간의 교정과 폭 예측

<Table 7>은 예측구간의 교정 상태와 폭 예측력을 검정구간 106,094 윈도우에서 측정한 결과이다. 폭 정보계수는
예측 구간의 폭 q90 − q10 과 앞으로 5일 실현 초과수익의 절댓값 사이의 상관이다. 뒤의 단순 기준 대조도
같은 목표를 쓰고 점수만 바꾼 것이다.

Table 7. Interval calibration and width predictive power (23,086 validation and 106,094 test windows)

| Measure | Validation | Test |
|---|---|---|
| Interval coverage (nominal 80%) | 77.5% | 71.4% |
| Coverage after conformal correction (+0.034) | - | 74.2% |
| **Width IC** (interval against absolute realized 5-day excess return) | 0.2710 (*t* = 30.19) | **0.1767 (*t* = 20.27)** |

폭 정보계수는 날짜별 횡단면 상관을 먼저 구한 뒤 그 시계열 평균으로 정의하며, *t* 값은 일별 계열의
표준오차에서 계산하였다. 목표가 5일 겹쳐 일별 계열에 자기상관이 남아 있는데 Newey-West 보정은 적용하지
않았으므로 *t* 값은 다소 낙관적이다.

변동성은 군집성 때문에 그 자체로 예측하기 쉬우므로 폭 정보계수의 절대 크기만으로는 모형의 기여를 가릴 수
없다. 단순 기준과 나란히 두면 가릴 수 있다. 5.3절과 같은 실행에서 직전 20일 실현변동성을 점수로 쓰는 기준의
정보계수는 +0.1450(*t* = 17.08)이었다. 같은 실행에서 제안 모형의 폭은 +0.1737(*t* = 19.45)로 이를
상회하였고, 분위 헤드만 부착한 모형은 +0.0962에 그쳐 단순 기준에 미치지 못하였으며, TFT 원본은
+0.1946(*t* = 24.05)으로 제안 모형보다 높았다. 폭에 변동성 정보를 담는 능력은 분위 헤드만으로는 생기지
않았으며, 변수선택·정적 문맥·매크로 입력이 함께 더해진 구조에서 나타났다. 어느 구성요소의 기여인지는 제거
실험 없이 가릴 수 없다(6장). 제안 모형은 그 능력을 TFT를 넘어서지는 못한 채 약 4분의 1의 비용으로 유지한다. <Table 7>의 폭 정보계수(+0.1767)와 이 단락의 값(+0.1737)이 다른 것은 표본이 2거래일 차이 나기
때문이다.

구간의 절대 교정은 좋지 않다. 명목 80% 구간이 검정구간에서 실제로 포함한 비율은 71.4%이고, 검증구간에서
산출한 등각예측 보정계수 +0.034를 적용해도 74.2%에 그친다. 본 모형의 구간은 체계적으로 좁다.

원인은 구간 간 변동성 차이에 있다. 폭은 변동성에 반응하지만 검정구간의 변동성 상승폭을 다 따라가지
못한다. 검정구간의 실현 p10~p90 폭이 0.1641인 데 비해 평균 예측 폭은 0.1410에 그친다. 검증구간에서는 두 값이 0.1080과 0.1100으로
거의 일치하여 포함률이 77.5%로 명목에 근접한다. 다만 4.4절의 기권 규칙은 폭의 절대 수준이 아니라 최근 분포
안에서의 상대 순위에 걸려 있다. 보정이 모든 폭에 같은 상수를 더하는 연산이므로 순위는 구성상 바뀌지 않고,
기권 선택도 그대로다. 기권에는 영향이 없으나 구간을 신뢰구간으로 읽어서는 안 된다.

## 6. 결론 및 향후 연구

본 연구는 주가 예측에서 "지금 판단해도 되는가"를 함께 답하는 모형을, 매일 201종목을 평가할 수 있는 비용
안에서 만드는 것을 목표로 하였다. 계산이 가벼운 패치 기반 트랜스포머에 TFT의 분위 예측 기능과 변수선택망을
결합하였다. 변수선택망은 입력 채널을 인코더 앞에서 하나로 축약하여 인코더 실행 횟수를 1회로 줄이는 데
사용하였다. 예측 구간이 이례적으로 넓을 때 판단을 보류하는 기권 규칙도 함께 적용하였다. 코스피200 구성종목
201개의 일별 자료로 평가하여 다음 두 가지를 확인하였다.

첫째, 제안 모형은 다섯 번의 반복 실험 모두에서 TFT보다 오차가 작았고, 그 정확도를 약 4분의 1의 비용으로
달성하였다. 조기종료에 쓰인 검증구간에서는 격차가 개선폭의 16%로 작아 두 모형을 대등하다고 보았다. 그
선택에 쓰이지 않은 평가 구간에서는 기준선 대비 개선폭이 +4.95% 대 +2.28%로 2.2배 벌어졌고, 날짜 단위 DM
검정도 다섯 시드 모두 유의하였다. 학습 시간은 3.8배, 종목당 추론 시간은 4.8배 짧았고, 활성값 메모리는
5.1배 적었다. 둘째, 예측 구간이 시장 변동성에 반응하여 판단 보류가 변동성이 큰 시기에
집중되었다. 보류된 예측은 실행된 예측보다 오차가 약 1.5배 컸다. 기권율을 맞춘 비교에서도 최근 변동성만 보고 보류하는
단순 규칙보다 큰 오차를 더 잘 걸러냈다. 하나의 예측값만 내는 기존 패치 기반 모형으로는 이러한 판단 보류
자체를 할 수 없다.

본 연구에는 다음 한계가 있다.

- 평가 기간의 자료가 입력 변수 구성과 기권 기준 선택에 참고되었으므로, 보고 수치는 완전한 사후 검증 값이
아니다. 다만 기권 기준 분위를 95에서 90, 75로 바꾸어도 판단 보류가 고변동 구간에 집중되는 경향은 유지되었다.
- 오르내림 방향을 맞히는 능력은 동일 조건의 그래디언트 부스팅 기준모형보다 낮았다. 기준모형은 LightGBM[19]의 방식을 따른 scikit-learn 구현이다. 공통 표본 101,745건에서
랭크 정보계수가 +0.0288 대 +0.0197이다. 제안 모형은 종목을 고르는 데보다 판단을 미룰 시점을 정하는 데
적합하다.
- 변수선택망의 가중치는 비용 절감 장치로만 사용하였다. 별도 진단에서 그 중요도는 개별 변수의 단변량
예측력과 일치하지 않았고 연도 간 거의 변하지 않았으므로, 매매 근거로는 제시하지 않는다.
- <Table 6>의 기권 지표와 폭 정보계수는 모형마다 체크포인트 하나씩으로 산출한 값이다. <Table 4>와 DM
검정은 두 모형을 같은 장비에서 다섯 시드씩 학습해 얻었으나, 기권·구간 지표는 시드 평균이 아니다.
- 구성요소를 하나씩 제거하는 실험과 TFT의 크기를 맞춘 비교를 수행하지 않아 개선이 어느 부분에서 왔는지는
가리지 못하였다. 종목 채널의 변수선택망을 매크로 경로와 같은 선형 합치기로 바꾼
변형도 비교하지 않았으므로, 채널 축약에 게이트가 필요했는지는 열려 있다.
- 유니버스를 현재 시점에서 추출하여 2015년 이후 상장폐지된 기업이 후보가 된 적이 없다. 시가총액 구간은 학습
종료 시점의 값으로 고정하였으므로 검증·검정 구간에는 사전 정보이고 학습 행에는 사후 값이다.
- 평가 기간은 한 시장의 한 구간이며, 구간을 옮겨 가며 재학습하는 walk-forward 검증은 수행하지 않았다.
코스피200은 구간 전체로 +172.6%이나 2024년 하반기 −17.1%, 2026년 7월 고점 대비 −40.9% 하락을 포함한다.
- 본 연구의 비교는 트랜스포머 계열 분위 모형 안으로 한정된다.
- 입력이 모두 공개된 가격·거래량 지표여서, 같은 재료를 쓴 모형들이 같은 종목을 지목하기 쉽다. 이는 국내 선행
연구[20]에서도 지적된 제약이다.

본 논문의 결과는 과거 자료에 대한 평가이며 투자 권유를 의도하지 않는다.

향후 연구에서는 비교 이후 새로 쌓인 자료를 평가에 쓰지 않은 구간으로 두고, 여러 시장 국면에서 다시 검증한다.

## References

[1] O. B. Sezer, M. U. Gudelek, and A. M. Ozbayoglu, "Financial time series forecasting with deep learning: A systematic literature review: 2005-2019," Applied Soft Computing, Vol.90, 106181, 2020.

[2] T. S. Kim and S. Lee, "Deep learning-based stock fluctuation prediction according to overseas indices and trading trends," KIPS Transactions on Software and Data Engineering, Vol.10, No.9, pp.367-374, 2021.

[3] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention is all you need," in Advances in Neural Information Processing Systems, 2017, pp.5998-6008.

[4] H. Zhou, S. Zhang, J. Peng, S. Zhang, J. Li, H. Xiong, and W. Zhang, "Informer: Beyond efficient transformer for long sequence time-series forecasting," in Proceedings of the AAAI Conference on Artificial Intelligence, 2021, pp.11106-11115.

[5] H. Wu, J. Xu, J. Wang, and M. Long, "Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting," in Advances in Neural Information Processing Systems, 2021.

[6] A. Zeng, M. Chen, L. Zhang, and Q. Xu, "Are transformers effective for time series forecasting?," in Proceedings of the AAAI Conference on Artificial Intelligence, 2023, pp.11121-11128.

[7] Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, "A time series is worth 64 words: Long-term forecasting with transformers," in Proceedings of the International Conference on Learning Representations, 2023.

[8] B. Lim, S. O. Arik, N. Loeff, and T. Pfister, "Temporal fusion transformers for interpretable multi-horizon time series forecasting," International Journal of Forecasting, Vol.37, No.4, pp.1748-1764, 2021.

[9] I. K. Kim, D. H. Kim, and J. Lee, "Temporal fusion transformers and deep learning methods for multi-horizon time series forecasting," KIPS Transactions on Software and Data Engineering, Vol.11, No.2, pp.81-86, 2022.

[10] D.-K. Kim and K. Kim, "Style-based transformer for time series forecasting," KIPS Transactions on Software and Data Engineering, Vol.10, No.12, pp.579-586, 2021.

[11] T. Kim, J. Kim, Y. Tae, C. Park, J.-H. Choi, and J. Choo, "Reversible instance normalization for accurate time-series forecasting against distribution shift," in Proceedings of the International Conference on Learning Representations, 2022.

[12] R. Koenker and G. Bassett, "Regression quantiles," Econometrica, Vol.46, No.1, pp.33-50, 1978.

[13] Y. Romano, E. Patterson, and E. J. Candes, "Conformalized quantile regression," in Advances in Neural Information Processing Systems, 2019.

[14] Y. Wang, H. Wu, J. Dong, G. Qin, H. Zhang, Y. Liu, Y. Qiu, J. Wang, and M. Long, "TimeXer: Empowering transformers for time series forecasting with exogenous variables," in Advances in Neural Information Processing Systems, 2024.

[15] T. Fischer and C. Krauss, "Deep learning with long short-term memory networks for financial market predictions," European Journal of Operational Research, Vol.270, No.2, pp.654-669, 2018.

[16] S. Gu, B. Kelly, and D. Xiu, "Empirical asset pricing via machine learning," The Review of Financial Studies, Vol.33, No.5, pp.2223-2273, 2020.

[17] H. J. Kim and Y. S. Jung, "A stock price prediction based on recurrent convolution neural network with weighted loss function," KIPS Transactions on Software and Data Engineering, Vol.8, No.3, pp.123-128, 2019.

[18] X. Yang, W. Liu, D. Zhou, J. Bian, and T.-Y. Liu, "Qlib: An AI-oriented quantitative investment platform," arXiv preprint arXiv:2009.11189, 2020.


[19] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, "LightGBM: A highly efficient gradient boosting decision tree," Advances in Neural Information Processing Systems, Vol.30, pp.3146-3154, 2017.

[20] M. K. Kwon and J. W. Nam, "AI and asset management: A review of the possibility of a paradigm shift," Issue Report 25-21, Korea Capital Market Institute, 2025 (in Korean).

<!-- AUTHORS:BEGIN -->
<!--
저자 소개 — 게재 확정 후 최종본에만 들어간다(규정 2.1). 양식 마지막 장 구성:
  이름(중고딕 11pt) / 사진 / ORCID / e-mail / 학력 / 경력 / 관심분야(신명조 8.5pt)
ORCID 는 orcid.org 에서 무료 발급. 관심분야는 게재본 관례대로 영문으로 적는다.
박성준 외 세 명은 본인이 직접 채운다 — 추정해 넣지 않았다.
사진은 게재논문 5편 중 4편이 저자 수만큼 넣었고 1편은 1인 생략 — 관례이나 절대 요건은 아닌 것으로 보인다.
-->

#### 박 성 준

![박성준](images/author_parksungjun.png)

https://orcid.org/0009-0004-9626-5165

e-mail : appop001@naver.com

2021년～현  재 동아대학교 컴퓨터공학과 학사과정

관심분야 : Time-Series Forecasting, Quantile Regression, Interpretable Deep Learning, Algorithmic Trading

#### 김 지 민

![김지민](images/author_kimjimin.jpeg)

https://orcid.org/0009-0004-4645-227X

e-mail : bckck@naver.com

2020년～현  재 동아대학교 컴퓨터공학과 학사과정

관심분야 : Time-Series Forecasting, Quantile Regression, Interpretable Deep Learning, Algorithmic Trading

#### 권 오 재

![권오재](images/author_kwonohjae.jpeg)

https://orcid.org/0009-0002-9537-5317

e-mail : rnjsdhwo4@naver.com

2021년～현  재 동아대학교 컴퓨터공학과 학사과정

관심분야 : Time-Series Forecasting, Quantile Regression, Interpretable Deep Learning, Algorithmic Trading

#### 송 하 얼

![송하얼](images/author_songhaeol.jpeg)

https://orcid.org/0009-0007-5429-4154

e-mail : 2337310@donga.ac.kr

2023년～현  재 동아대학교 컴퓨터공학과 학사과정

관심분야 : Time-Series Forecasting, Quantile Regression, Interpretable Deep Learning, Algorithmic Trading

#### 서 정 일

[사진]

https://orcid.org/0000-0000-0000-0000

e-mail : jeongilseo@dau.ac.kr

20○○년 ○○대학교 ○○학과(학사)

20○○년 ○○대학교 ○○학과(석·박사)

20○○년～현  재 동아대학교 컴퓨터공학과 부교수

관심분야 :

#### 유 주 한

[사진]

https://orcid.org/0000-0000-0000-0000

e-mail : unchinto@dau.ac.kr

20○○년 ○○대학교 ○○학과(학사)

20○○년 ○○대학교 ○○학과(석·박사)

20○○년～현  재 동아대학교 컴퓨터공학과 조교수

관심분야 :
<!-- AUTHORS:END -->
