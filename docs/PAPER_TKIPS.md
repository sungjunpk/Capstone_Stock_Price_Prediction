<!--
정보처리학회 논문지 (The Transactions of the Korea Information Processing Society, TKIPS) 투고 원고
(KTSDE·KTCCS 는 2024년 TKIPS 로 통합된 옛 논문지다)
양식 출처: docs/kips/paper-format_TKIPS.hwp (쪽·단·글꼴 설정은 옛 KTSDE 양식과 같다)
언어: 국문 본문. 규정 2.1 에 따라 제목·저자·소속·초록·키워드·표·그림·감사의 글·
참고문헌은 영문으로 작성한다.
1쪽 구성은 실제 게재논문(정보처리학회논문지 제13권 제1호)을 따랐다.
저자 정보는 AUTHORS 표시 구간에 모아 두었고, 심사용 원고에서는 자동으로 빠진다.
⚠️ hwp 최종본 작업 시: 식 (1)~(6)이 여기서는 일반 텍스트다. 게재본은 모두 수식 개체를
쓰므로 hwp 수식 편집기로 다시 입력할 것.
-->

# A Lightweight Quantile Forecasting Model for Stock Returns with Abstention

<!-- 제목 74자. 양식 영문 지침에는 "A manuscript title should be no more than 40
characters in length." 가 있으나, docs/kips/samples/ 의 게재본 5편 영문 제목이 62~103자다
(전력수요 80 · 비디오 보간 80 · e-러닝 103 · 로봇팔 62). 문서에는 있고 실무에서는 지켜지지
않는 규정으로 보여 게재본 관행과 국문 제목과의 대응을 택했다. 사무국 문의 시 함께 확인할 것. -->

<!-- AUTHORS:BEGIN -->
Seongjun Park†⋅Jimin Kim†⋅Ohjae Kwon†⋅Haeol Song†⋅Jeongil Seo††⋅Juhan Yoo†††
<!-- AUTHORS:END -->

## ABSTRACT

Stock price forecasting has been studied in economics, statistics and computer
science, and recent work increasingly trains artificial intelligence models on
price and trading data to predict price movements. Most of these models,
however, report a single predicted value and therefore cannot tell the user
whether that prediction should be trusted at the present moment. This paper
proposes a deep learning model that identifies the moments it is unsure about
and makes no prediction at those moments, which is called abstention. The
proposed model uses a lightweight architecture and reports, instead of a single
value, the range in which the future return is expected to fall; when that range
is among the widest 5% of the ranges observed over the past 90 days, the model
withholds its decision. The proposed model and the comparison models were
evaluated on daily data for 201 KOSPI 200 stocks since 2015, in terms of both
forecast quality and abstention quality, and the proposed model performed better
than the comparison models. In particular, withheld forecasts carried about 1.5
times the error of those acted on, and when the same share of forecasts was
withheld, the proposed criterion filtered out large errors better than a simple
rule based on recent price movement. These results indicate that the
contribution of the proposed model lies not in the accuracy of the predicted
value but in selecting the moments at which a prediction should not be acted on.

<!-- 239 단어 (공백 분리 기준; 제한 300). 한 문단, 표·참고문헌·장 번호 없음 -->

**Keywords :** Stock Return Forecasting, Patch-Based Transformer, Quantile Regression,
Forecast Uncertainty, Abstention

## 판단 보류가 가능한 경량 주가 수익률 분위 예측 모형

<!-- AUTHORS:BEGIN -->
박 성 준†⋅김 지 민†⋅권 오 재†⋅송 하 얼†⋅서 정 일††⋅유 주 한†††
<!-- AUTHORS:END -->

### 요     약

주가 예측은 경제학, 통계학, 컴퓨터공학 등 여러 분야에서 연구되는 주제이며, 최근에는 가격과 거래 자료로부터 인공지능 모형을 학습하여 주가의
변동을 예측하는 연구가 활발해지고 있다. 그러나 대부분의 모형은 예측값을 하나만 내놓기 때문에, 그 예측을 지금 믿어도 되는지는 알려주지
못한다. 본 연구에서는 확신하기 어려운 시점을 스스로 가려내고 그 시점에는 예측을 내놓지 않는, 판단 보류가 가능한 딥러닝 모형을 제안한다.
제안 모형은 계산이 가벼운 구조를 사용하여 예측값 하나 대신 미래 수익률이 들어올 범위를 함께 산출하고, 그 범위가 최근 90일 동안 나온 범위
가운데 넓은 쪽 5%에 들면 판단을 보류한다. 2015년 이후 코스피200 구성종목 201개의 일별 자료로 제안 모형과 비교 모형들의 예측
성능과 보류 성능을 비교한 결과, 제안 모형이 비교 모형들보다 우수한 성능을 보였다. 특히 보류한 예측의 오차는 실행한 예측보다 약 1.5배
컸고, 같은 비율만큼 보류하도록 맞춘 비교에서도 최근 변동폭만 보는 단순한 규칙보다 오차가 큰 예측을 더 잘 걸러냈다. 이 결과는 제안 모형의
기여가 예측값의 정확도가 아니라 예측을 실행하지 말아야 할 시점의 선별에 있음을 보여준다.

**키워드 :** 주가 수익률 예측, 패치 기반 트랜스포머, 분위회귀, 예측 불확실성, 판단 보류

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

주가 예측 모형은 과거의 가격과 거래 자료에서 규칙을 찾아 미래의 수익률을 추정한다. 그러나 주식 시장은 시기에 따라 움직이는 방식이 달라진다.
한동안 들어맞던 규칙이 다음 시기에는 들어맞지 않는다. 그러므로 어떤 예측 모형도 모든 시점에서 같은 정확도를 유지하지 못한다.

문제는 모형이 언제 틀릴지를 사용자에게 알려주지 않는다는 데 있다. 대부분의 예측 모형은 미래 값을 하나의 수치로 내놓는다. 이러한 방식을 점
예측(point forecast)이라고 한다. 점 예측은 값 하나만 제시하므로, 사용자는 그 값이 모형이 잘 맞히는 시점에서 나온 것인지 판단할
수 없다. 투자에서는 틀린 예측을 실행했을 때의 손실이 크기 때문에, 이 결함은 정확도가 조금 낮은 것보다 더 큰 문제가 된다.

따라서 예측 모형은 자신이 확신하지 못하는 시점을 스스로 가려내고, 그 시점에는 예측을 내놓지 않을 수 있어야 한다. 모형이 일부 시점에 대하여
답을 내지 않는 것을 판단 보류(abstention)라고 한다. 본 연구는 주가 예측에서 판단 보류를 수행하는 모형을 다룬다.

판단 보류를 하려면 모형이 자신의 불확실성을 수치로 내놓아야 한다. 본 연구는 그 수치를 예측 분포에서 얻는다. 모형이 값 하나 대신 여러 개의
분위수(quantile)를 출력하도록 하면 된다. 분위수는 실제 값이 그보다 작을 확률이 정해진 수준이 되는 값이다. 예를 들어 모형이 하위
10% 분위수와 상위 10% 분위수를 함께 출력하면, 두 값 사이의 구간이 미래 수익률이 놓일 범위가 된다. 이 구간이 넓다는 것은 모형이 그
시점의 결과를 좁게 특정하지 못한다는 뜻이다. 본 연구는 구간이 평소보다 넓은 시점에서 판단을 보류한다.

기존 연구에는 이 두 가지가 함께 갖추어져 있지 않다. 예측 분포를 산출하는 모형은 이미 제안되어 있다. Temporal Fusion
Transformer(TFT)는 여러 분위수를 동시에 출력한다. 그러나 TFT는 입력 기간의 모든 시점을 서로 비교하는 연산을 수행하므로 학습에
드는 시간과 메모리가 크다. 반대로 연산량을 줄이도록 설계된 최근의 시계열 예측 모형들은 예측값을 하나만 내놓는다. 이 모형들은 판단 보류의
근거가 되는 구간을 제공하지 못한다.

본 연구는 연산량이 적은 예측 모형이 여러 분위수를 함께 출력하도록 바꾸고, 그 출력에서 판단 보류 규칙을 정의하였다. 코스피200 구성종목
201개의 2015년 이후 일별 자료로 이 모형을 평가하였다. 본 연구의 기여는 세 가지이다. 첫째, 예측 구간의 폭을 기준으로 판단을 보류하면
오차가 큰 예측이 걸러진다. 보류한 예측의 오차는 실행한 예측의 약 1.5배였다. 이 결과는 보류 비율을 같게 맞춘 상태에서 최근 변동성만 보고
보류하는 규칙과 비교하여 얻었다. 둘째, 이 성질은 분위수 출력을 붙이는 것만으로는 생기지 않는다. 분위수 출력만 추가한 모형은 같은 비교에서
변동성 규칙을 넘어서지 못하였다. 셋째, 제안 모형의 중앙값 예측은 상수 기준선보다 정확하지 않았다. 본 연구의 기여는 예측값의 정확도 향상이
아니다. 예측을 실행하지 말아야 할 시점을 선별하는 데 있다.

이 논문의 구성은 다음과 같다. 2장에서는 판단 보류에 필요한 요소를 기준으로 기존 연구를 정리한다. 3장에서는 제안하는 모형과 판단 보류
규칙을 설명한다. 4장에서는 실험에 사용한 자료를, 5장에서는 실험 결과를 제시한다. 6장에서 결론을 내리고, 7장에서 후속 연구를 밝힌다.

## 2. 관련 연구

### 2.1 주가 예측에 쓰인 딥러닝 모형

금융 시계열에 딥러닝을 적용한 연구는 순환신경망과 합성곱신경망을 중심으로 축적되었다[1]. 장단기기억 신경망으로 미국 주식의 등락 방향을 예측한
연구[16], 여러 기계학습 기법을 자산가격 결정에 적용한 연구[17]가 대표적이다. 국내에서는 해외지수와 투자자별 매매 동향을 입력으로 삼은
등락 예측[2], 가중 손실 함수를 적용한 순환 합성곱 구조[18]가 보고되었다. 이 연구들은 입력 변수와 신경망 구조를 바꾸어 정확도를 높이는
데 집중하였고, 출력은 모두 점 예측이다. 따라서 모형이 어느 시점에서 확신하지 못하는지를 사용자가 알 수 없고, 1장에서 제기한 판단 보류도
정의되지 않는다.

### 2.2 예측 분포를 내는 모형

값 하나 대신 여러 분위수를 추정하는 방법은 분위회귀[11]에서 시작되었다. Temporal Fusion Transformer(TFT)[8]는
이를 딥러닝에 적용하여, 입력 변수의 중요도를 학습하는 장치와 시점별 어텐션을 결합하고 여러 분위수를 동시에 출력한다. 국내에서도 TFT를 여러
시점 앞 예측에 적용한 연구[9]가 있다. 이 계열은 판단 보류에 필요한
예측 구간을 제공한다. 단점은 연산량이다. TFT는 입력 기간의 모든 시점을 서로 비교하므로 학습 시간과 메모리가 입력 길이에 민감하다. 분위수
예측이 명목 확률을 지키는지는 별도의 문제이며, 사후에 구간을 교정하는 방법이 제안되어 있다[12].

### 2.3 연산량을 줄인 시계열 트랜스포머

트랜스포머[3]는 모든 시점 쌍을 비교하므로 연산량이 입력 길이의 제곱에 비례한다. 이를 줄이려고 어텐션 대상을 일부만 고르는 구조[4]와
계열을 추세와 주기로 나누는 구조[5]가 제안되었으나, 단순한 선형 모형이 이들을 앞선다는 보고[6]도 있다. PatchTST[7]는 시계열을
시점이 아니라 일정 길이의 구간 단위로 묶어 하나의 토큰으로 만든다. 비교 대상이 줄어 연산량이 감소하고, 변수마다 따로 처리하여 변수 수에
선형으로 확장된다. 이 계열의 단점은 출력이 점 예측 하나라는 것이며, 판단 보류의 근거가 되는 구간이 나오지 않는다.

### 2.4 판단 보류를 다룬 연구

모형이 일부 입력에 답을 내지 않는 선택지는 기각 옵션(reject option)으로 형식화되어 있다[13]. 오답 비용과 기각 비용의
교환관계로 최적 기각 영역을 정의한 이 틀은 선택적 예측(selective prediction)으로 일반화되어, 답을 낸 비율과 그 표본에서의
오차를 함께 보고하는 평가 방식으로 자리잡았다[14]. 심층 신경망에서는 이미 학습된 모형의 신뢰도 점수로 기각 영역을 정하는 방법이
제안되었다[15]. 이 연구들은 분류 문제를 대상으로 하고 선택 기준으로 예측 확률을 쓴다. 회귀 문제에는 그 확률이 없어 같은 방법을 그대로
적용할 수 없다.

### 2.5 본 연구의 위치

2.1절부터 2.3절까지의 세 계열은 각각 하나씩을 갖추지 못한다. 주가 예측에 쓰인 딥러닝 모형은 예측값 하나만 내므로 보류의 근거가 없다.
예측 분포를 내는 모형은 그 근거를 주지만 학습에 드는 시간과 메모리가 크다. 연산량을 줄인 구조는 가볍지만 다시 예측값 하나만 낸다.
2.4절의 선택적 예측은 보류의 틀을 주지만 분류를 전제하므로, 회귀에서는 선택 기준을 새로 정해야 한다.

본 연구는 이 넷을 하나로 모은다. 연산량이 적은 구간 단위 토큰 구조는 그대로 두고, 출력 부분만 여러 분위수를 내도록 바꾼다. 그러면 가벼운
모형이 예측 구간을 함께 내놓는다. 보류의 기준으로는 그 구간의 폭을 쓴다. 폭은 모형이 이미 내놓은 분위수를 빼기만 하면 나오므로, 보류를
위해 별도의 신경망을 학습시키거나 같은 예측을 여러 번 반복할 필요가 없다. 선택적 예측 연구가 분류에서 쓰던 확률을 대신할 기준을, 회귀에서는
이 폭에서 얻는 것이다.

비교 대상에는 트리 계열 기준모형도 포함하였다. 잘 조정된 단순 기준모형과 견주지 않으면 개선폭이 과장되기 쉽다는 지적이 예측 문헌에서 반복되어
왔기 때문이다[19]. 같은 자료와 같은 분할에서 그래디언트 부스팅 결정트리(gradient boosting decision tree,
GBDT)를 함께 학습하여 5장에 보고한다.

## 3. 제안하는 방법

### 3.1 설계 요구사항

<Table 1>은 1장에서 밝힌 요구사항과 그것을 충족하기 위해 고른 구성요소를 정리한 것이다.

Table 1. Requirements and corresponding components

| Requirement | Required property | Part chosen |
|---|---|---|
| Abstention | A predicted distribution, not a single value | Quantile output layer, pinball loss |
| Scale differs by stock and era | Normalization that assumes no fixed price level | RevIN, with channel exceptions |
| Local equities track global macro | A way to inject an exogenous series | Macro cross-attention (design option, Sec. 5.4) |
| A limited compute budget | Fewer tokens, and one encoder pass over the channels | Patch tokenization (120 to 24), channel reduction before the encoder (Sec. 5.4) |

### 3.2 전체 구조

제안 모형의 전체 구조는 (Fig. 1)과 같다. 입력은 세 갈래다. 첫째는 종목마다 매일 계산되는 지표 30개를 120일치 모은 것이다.
둘째는 시장 전체를 나타내는 지표 13개의 같은 기간 계열이다. 셋째는 섹터, 시가총액 구간, 요일처럼 한 번 정해지면 윈도우 안에서 바뀌지
않는 값 3종이다. 셋째 입력은 문맥 벡터로 바뀌어 앞의 두 경로가 어떤 변수에 비중을 둘지를 조건화한다. 예측에는 가장 최근 구간의 표현만
쓴다.

![Fig. 1](images/fig1_architecture.png)

Fig. 1. Architecture of the proposed model (a) with component detail (b-e). Colors in (a) mark
the PatchTST lineage (yellow), the TFT components (blue), the macro cross-attention added
in this work (teal) and the static context path (orange). The variable selection network and the
cross-attention are design options; removing either did not change the results (Sec. 5.4).
Panels (b)-(d) are adapted from [3] and [8].

입력은 윈도우마다 정규화한다. 각 윈도우의 통계로 입력을 표준화하고 출력에서 되돌리는 방식이며[10], 종목과 시대에 따라 가격 수준의 의미가
달라지는 자료에서는 필수적이다. 다만 횡단면 순위 채널에는 적용하지 않았다. 이 채널은 정보가 값의 수준 자체에 담겨 있어, 윈도우 안에서
표준화하면 그 정보가 사라진다.

모형의 출력은 세 개의 분위수다. 분위수를 학습시키는 손실은 분위회귀[11]에서 온 핀볼 손실이며, 분위 *τ*에 대하여 식 (1)과 같이
정의된다.

&nbsp;&nbsp;&nbsp;&nbsp;L(*τ*; *y*, *ŷ*) = max[ *τ*(*y* − *ŷ*), (*τ* − 1)(*y* − *ŷ*) ]&nbsp;&nbsp;&nbsp;&nbsp;(1)

본 연구는 *τ* ∈ {0.1, 0.5, 0.9}를 동시에 학습한다. 세 값이 한 번의 계산으로 나오므로 불확실성을 따로 추정하기 위한 추가
연산이 필요하지 않다. 사후 교정을 적용한 결과는 6장에 밝힌다.

### 3.3 설계 선택

아래 1)과 2)는 구조의 필수 요소가 아니라 비교 검토한 설계 선택지이며, 각각의 기여는 5.4절의 제거 실험으로
확인한다. 3)은 보류 규칙이 의존하는 구조적 요건이다.

#### 1) 변수선택망의 역할과 배치

구간 단위로 토큰을 만들면 채널마다 24개의 토큰이 생긴다. 이 채널 축을 하나로 줄여야 인코더를 한 번만 실행할 수 있다. 매크로 경로에는
채널을 선형 변환으로 합쳤고, 종목 채널에는 채널별 가중치를 학습하는 변수선택망을 두었다. 가중치가 정적 문맥에 따라 달라져 섹터와 시가총액
구간별로 다른 비중을 줄 수 있기 때문이다(식 2, 3).

&nbsp;&nbsp;&nbsp;&nbsp;*v*_{n} = softmax( GRN( [ flat(*E*_{n}) ; *c*_{s} ] ) ) ∈ ℝ^{C},&nbsp;&nbsp; *n* = 1, …, *N*&nbsp;&nbsp;&nbsp;&nbsp;(2)

&nbsp;&nbsp;&nbsp;&nbsp;*Z*_{n} = Σ_{i} *v*_{n,i} · GRN_{i}(*E*_{n,i})&nbsp;&nbsp;&nbsp;&nbsp;(3)

여기서 *C*는 채널 수, *N*은 패치 수, *E*_{n,i}는 패치 *n*·채널 *i*의 임베딩이고, flat(*E*_{n})은 그 패치의
모든 채널 임베딩을 이어 붙인 벡터, *c*_{s}는 정적 임베딩에서 얻은 문맥 벡터, GRN은 TFT의 게이트 잔차
신경망이다.

변수선택망을 인코더 뒤에 두면 인코더가 채널 수만큼 반복되지만, 앞에 두면 채널이 먼저 하나로 합쳐져 한 번만 실행된다. 다만 종목 채널까지
선형 합치기로 바꾼 변형도 정확도와 보류 행태가 같았다(5.4절). 연산을 줄이는 것은 인코더 앞의 채널 축약 자체이며, 학습된 가중치가
정확도를 더하지는 않았다.

#### 2) 매크로 크로스어텐션의 방향

매크로 지표는 크로스어텐션으로 종목 표현에 주입한다. 종목 표현을 질의로, 매크로 부호화를 키와 값으로 둔다. 방향을 반대로 두면 출력이 매크로
시퀀스의 형상을 따라 종목별 예측이 나오지 않는다. 외생 변수를 어텐션으로 주입하는 발상은 TimeXer[20]에도 있으며, 본 연구는 매크로를
패치 단위로 유지한다는 점이 다르다. 매크로 경로는 13채널을 하나로 합치는 선형 변환으로 처리하였다. 이 모듈의 정확도 기여는 5.4절의 제거
실험에서 확인되지 않았다.

#### 3) 분위 단조성의 구조적 보장과 출력층 초기화

분위 교차를 손실 벌점이 아니라 누적 소프트플러스 구조로 원천 차단하였다(식 4).

&nbsp;&nbsp;&nbsp;&nbsp;*q*_{10} = *b*,&nbsp;&nbsp; *q*_{50} = *q*_{10} + softplus(δ_{1}),&nbsp;&nbsp; *q*_{90} = *q*_{50} + softplus(δ_{2})&nbsp;&nbsp;&nbsp;&nbsp;(4)

*b*와 δ_{1}, δ_{2}는 출력층 마지막 선형층의 세 출력이다. 교차가 발생하면 보류 폭이 음수가 되어 하위 규칙 전체가 무효화되므로
구조적 보장이 필요하다. 5.1절의 세 트랜스포머 모형은 이 출력층을 공유하며, 다섯 시드의 검증·시험 구간 647,880건에서 교차는 한 건도
없었다. 이는 구조가 구현에서 지켜졌음을 확인한 것이지 보장이 없는 모형과 비교한 결과가 아니다.

출력층의 초기화도 성능에 영향을 주었다. 기본 초기화에서는 softplus(0) = 0.693이라 초기 구간 폭이 목표 분포와
어긋난 지점에 놓여 기준선에 미달하므로, 최종층 편향을 학습자료의 무조건부 분위수로 설정하여 해소하였다.

### 3.4 판단 보류 규칙

비교하는 모든 모형에 같은 규칙을 적용한다. 예측 구간의 폭이 최근 분포의 95분위를 넘으면 그 시점의 판단을 보류한다. 여기서 최근 분포는
판단일로부터 거슬러 달력 90일, 거래일로는 약 60일 안에 나온 폭을 모은 것이며, 종목별로 나누지 않고 전 종목을 합쳐 만든다. 구간 전체의
폭 분포를 쓰면 뒤쪽 날짜의 폭이 앞쪽 날짜의 판단에 흘러들어 미래 정보가 섞인다. 임계값으로 95분위를 고른 근거와 다른 값으로 바꿨을 때의
영향은 5.3절에서 다룬다.

### 3.5 최종 설정

<Table 2>는 이후 모든 실험에 사용한 설정이다. 정규화는 3.2절에서 기술한 방식을 따른다.

Table 2. Final model configuration

| Item | Value |
|---|---|
| Lookback / patch / stride | 120 / 5 / 5, giving 24 tokens |
| d_model / heads / layers / d_ff / dropout | 32 / 2 / 1 / 64 / 0.5 |
| Quantiles | 0.1, 0.5, 0.9 |
| Dynamic / macro / static inputs | 30 channels / 13 channels / 3 categoricals |
| Normalization | RevIN, applied selectively by channel |
| Parameters | 0.50M |

## 4. 데이터셋

### 4.1 자료

분석 자료는 코스피200 구성종목 201개의 2015년 1월 2일부터 2026년 10월 16일까지 일별 자료 519,617행이며,
종목당 최대 2,892거래일이다. 종목 목록은 2026년 9월 8일 지수 구성종목 조회 결과이고, 지수와 투자자 수급 계열을 포함해 모두 증권사 Open API로 수집하였다.
거래정지일은 거래량이 0이고 시가·고가·저가·종가가 같게 기록되는데, 그대로 두면 허위의 0 수익률이 생겨 변동성 추정을 왜곡하므로 지표 산출
전에 제거하였다. 종목마다 거래일 수가 크게 다르고, 가장 짧은 종목은 35거래일뿐이다. 201종목이 모두 존재하는 공통 달력을 강제하면 전체
행의 98.6%가 사라지므로, 윈도우는 각 종목의 자기 행 위에서 구성하였다. 모형과 평가 코드는 요청에 따라 제공하며, 원자료는 API 이용
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

정적 범주형 세 개는 섹터, 시가총액 구간, 요일이다. 시가총액 구간은 학습구간 종료일의 값으로 고정하였고,
요일은 사전 알려진 공변량이나 구현에서는 윈도우 마지막 날의 값을 정적 입력으로 넣었다.

가격 수준은 직접 사용하지 않는다. 이동평균은 종가 대비 괴리율로, 수렴·발산 지표와 ATR은 종가로 나누어,
유계 지표는 재척도하여 입력한다. 목표변수는 코스피200 대비 초과분으로 정의한 5일 누적 로그수익률이다(식 5).

&nbsp;&nbsp;&nbsp;&nbsp;*y*_{t} = Σ_{k=1}^{5} *r*_{t+k} − Σ_{k=1}^{5} *r*^{index}_{t+k}&nbsp;&nbsp;&nbsp;&nbsp;(5)

*r*_{t}는 일별 로그수익률이고 *r*^{index}_{t}는 같은 날 코스피200의 로그수익률이다.

### 4.3 분할과 기준선

분할은 종목이 아니라 날짜 기준 전역 분할이다. 2022년까지 학습, 2023년 검증, 2024년 이후 시험이며, 목표가
5일 앞을 보므로 각 경계에 5일 격리구간을 둔다. 정규화 통계는 학습 구간 행에서만 추정한다.

윈도우는 120일 입력이 해당 구간 안에 온전히 들어갈 때만 만든다. 입력이 앞 구간에 걸쳐도 격리구간이 목표값 중첩을 막으므로 누수는
아니지만, 그 경우 검증 표본의 입력 대부분이 학습에서 이미 본 행이 된다. 그 대가로 각 구간의 처음 120거래일이 소모된다. 검증 표본은
2023년 7월 3일부터 12월 28일까지 122거래일·190종목·23,086건, 시험 표본은 2024년 7월 3일부터 2026년 10월
8일까지 550거래일·198종목·106,490건이다. 자료는 2026년 10월 16일 수집분으로 동결하였다.

인접 표본은 120일 중 119일의 입력과 5일 중 4일의 목표를 공유하므로 명목 학습 행 264,653건의 실질 규모는
훨씬 작다. 이 때문에 모수를 늘리면 1~2회차에 과적합되어 0.50M으로 고정하였다.

모든 실험은 학습 자료의 무조건부 분위수, 즉 입력을 무시하고 세 상수를 출력하는 예측기를 기준선으로 평가한다.
이를 상회하지 못하면 초매개변수 조정 문제가 아니라 변수·목표 설계의 실패로 간주한다.

## 5. 실험 결과

### 5.1 동일조건 정확도 비교

판단 보류는 모형이 내놓은 예측 구간에 의존하므로, 그 구간을 만드는 분포가 쓸 만한지를 먼저 확인한다. 세 모형은 자료·분할·학습절차·조기종료
규칙을 공유하고 기준선도 0.012175로 같다. 두 가지 제약이 있다. PatchTST 계열 대조군은
원 구조에 해당 개념이 없어 매크로와 정적 변수를 입력받지 못한다. 초매개변수는 제안 모형에 맞춘 값을 대조군에도 그대로 적용하였으므로 대조군의
최적값이 아니다.

세 트랜스포머 모형은 같은 혼합정밀도 설정으로 캐글 GPU에서 학습하고 한 장비에서
평가하였으며, 분위 GBDT는 적합에 난수 요소가 없어 한 번만 학습하였다. 점 예측을 그대로 내는 원본
PatchTST는 표에서 제외하였다. 세 분위를 같은 값으로 채점하면 기준선보다 47.79% 나쁜데, 이는 구조의
비교가 아니라 손실함수의 차이를 보는 것이기 때문이다.

Table 4. Forecast accuracy on the validation and test splits (three-quantile pinball loss; mean over five seeds 42-46 with the standard deviation, single run for the quantile GBDT). Lower is better, and 'vs baseline' is the improvement over the unconditional-quantile predictor, which scores 0.012175 on validation and 0.018754 on test

| Model | Params | Validation | vs baseline | Test | vs baseline | Best epoch |
|---|---|---|---|---|---|---|
| PatchTST + quantile head | 0.035M | 0.012166 ± 0.000022 | +0.07% | 0.018686 ± 0.000130 | +0.36% | 1-5 |
| TFT, original | 0.687M | 0.011935 ± 0.000020 | +1.97% | 0.018325 ± 0.000091 | +2.28% | 1-2 |
| Quantile GBDT | 375 trees | 0.011890 | +2.34% | 0.018107 | +3.45% | - |
| **Ours** | 0.500M | 0.011889 ± 0.000012 | +2.35% | **0.017826 ± 0.000059** | **+4.95%** | 2-10 |

제안 모형은 두 구간 모두에서 TFT 원본보다 손실이 낮았고, 시드를 짝지은 비교에서도 구간마다 다섯 시드 전부가
같은 방향이었다(구간별 부호검정 *p* = 0.031 단측, 두 구간 각각). 두 구간은 같은 체크포인트에서 나오므로
열 번을 독립 시행으로 합치지 않는다. 검증구간의 격차는 핀볼 손실 0.000046으로 기준선 대비
개선폭의 16%에 그치므로 우위가 아니라 대등하거나 근소하게 낫다는 수준으로 해석한다. 조기종료와 최적 회차
선택에 쓰이지 않은 시험구간에서는 격차가 0.000499로 벌어지며 다섯 시드의 범위가 겹치지 않는다
(0.017739~0.017895 대 0.018205~0.018407). 평균 차이에 대한 Welch 검정은 검증구간 *t* = 4.47(*p* = 0.004),
시험구간 *t* = 10.33(*p* < 0.001)이다. 같은 표본의 날짜별 손실 계열에 Diebold-Mariano 검정[21](Newey-West 4시차[22],
Harvey-Leybourne-Newbold 소표본 보정[23])을 적용하면 550거래일의 시험구간에서는 다섯 시드 모두 유의하나(*p* ≤ 0.022) 122거래일의 검증구간에서는 다섯 시드 모두 유의하지 않다(*p*
= 0.16~0.56). 날짜를 표본으로 보는 검정이므로 짧은 구간에서는 검정력이 약하다. 분위수 출력층만 부착한
PatchTST는 검증구간 +0.07%, 시험구간 +0.36%로 상수 예측을 간신히 상회하는 데 그쳤고, 다섯 시드 가운데
네 시드는 첫 회차에서 최적값에 도달하였다. 격차가 시험구간에서 벌어지는 이유의 하나는 학습 곡선이다. TFT는 다섯 시드 모두
1~2회차에서 최적값에 도달한 뒤 과적합으로 돌아섰고, 제안 모형의 최적 회차는 2~10회차에 분포하였다. 세 모형을 모두 단정밀도로 맞추어 한 번씩 학습하면 격차는 좁아지지만 순서는 바뀌지 않는다. 중앙값만 떼어 평균절대오차로 재면 제안 모형은 0.053439 ± 0.000142로 상수
기준선(0.053447)과 사실상 같고 TFT 원본은 0.053707로 오히려 나쁘므로, 핀볼 손실의 개선은 중앙값의 점
정확도가 아니라 예측 구간에서 온다.

트리 계열 기준모형으로 분위 손실을 쓰는 히스토그램 그래디언트 부스팅(LightGBM[24]의 방식을 따른
scikit-learn[25] 구현)을 같은 분할과 같은 윈도우 표본에서 평가하였다. 입력은 동적 30채널의 당일 값과 1·5·20일 시차, 매크로 13채널, 정적 범주형 3종의 136개이고, 분위마다 모형을 하나씩 두어 검증구간
손실로 트리 수를 정하였다(75·25·275개). 검증구간 손실은 제안 모형과 같았고(0.011890 대 0.011889), 시험구간 손실은 TFT
원본보다 작고 제안 모형보다 컸다. 제안 모형의 다섯 시드는 모두 이 값보다 낮았다(최대 0.017895 대 0.018107).

### 5.2 계산 비용

자원 사용량은 본 연구의 기여가 아니라, 제안 모형이 기존 분포 예측 모형을 대신할 수 있는지를 확인하는 조건이다. Table 5.
Computational cost measured on one device (Mac, Metal GPU backend; batch 64, 120-day input, 201 stocks). The quantile GBDT row is one full fit of the three quantile models on the CPU (8 cores), so it has neither epochs nor activation memory. The text below gives the measurement procedure

| Model | s / epoch | Latency / stock | Universe pass | Activation | Param memory |
|---|---|---|---|---|---|
| PatchTST (+ quantile head) | 110.8 | 0.092 ms | 18.6 ms | 172.8 MB | 0.14 MB |
| TFT, original | 550.8 | 0.693 ms | 139.4 ms | 679.1 MB | 2.75 MB |
| Ours | 143.5 | 0.143 ms | 28.7 ms | 132.9 MB | 2.00 MB |
| Quantile GBDT (CPU) | 26.1 (full fit) | 0.068 ms | 13.4 ms | - | 2.13 MB |

epoch당 시간은 40배치의 학습 스텝 시간 중앙값에 학습 윈도우가 만드는 4,135배치를 곱한 값이다. 전 종목 1회 통과는 201종목
순전파 21회의 중앙값이고, 종목당 지연시간은 그 값을 201로 나눈 것이다. 활성값은 역전파 그래프를 남기는 순전파가 추가로 할당한 장치
메모리다.

제안 모형은 TFT 원본보다 학습 시간이 3.8배, 활성값 메모리가 5.1배 적으면서 손실은 더 낮았다. 입력을 구간 단위로 묶어 어텐션 대상
길이를 120에서 24로 줄이고, 채널을 인코더 앞에서 하나로 합쳐 반복 실행을 없앤 결과다. 같은 배율은 장비가 달라도 재현되어, 캐글
GPU에서는 제안 모형 41.8초/epoch, TFT 원본 157.1초/epoch였다.
분위 GBDT의 비용은 이보다도 낮다. 세 분위 모형 전체를 한 번 학습하는 데 CPU에서 26.1초로 제안 모형의 1 epoch보다
짧았고, 종목당 추론도 0.068 ms로 절반 이하였다. 따라서 본 연구의 경량은 트랜스포머 분위 모형, 특히 TFT
대비의 의미이며, 제안 모형이 GBDT 대비 얻는 것은 비용이 아니라 시험구간 손실의 추가 개선(+4.95% 대
+3.45%)이다.

(Fig. 2)는 <Table 4>의 검증 정확도와 <Table 5>의 비용을 한 평면에 놓은 것이다. 세 트랜스포머 모형 가운데 제안 모형이
파레토 전선 위에 있고, 분위 GBDT는 검증 정확도가 같으면서 지연이 더 짧다.

![Fig. 2](images/fig2_cost_accuracy.png)

Fig. 2. Forecast accuracy against inference cost. The vertical axis is the pinball
improvement over the unconditional-quantile baseline on validation, the split the models
were selected on; the test-split gap is wider (<Table 4>). The
horizontal axis is per-stock inference latency, and the circle area is activation memory
for a 120-day input (<Table 5>). The quantile GBDT (open diamond) is measured on the CPU
and has no activation memory, so its marker size does not apply. The original point-forecast
PatchTST is omitted because its pinball loss falls far below the plotted range.

### 5.3 판단 보류 행태

네 분위 모형에 3.4절의 같은 보류 규칙을 적용하고 시험구간 전체에서 집계하였다. 점 예측 모형은 임계값을 적용할 폭이 없어 보류가 정의되지 않으므로 제외하였다.

Table 6. Abstention on the test split (550 days, 106,490 windows), mean over five seeds with the standard deviation, single run for the quantile GBDT. The threshold is the 95th percentile of the width distribution over the trailing 90 calendar days, pooled across all stocks. Q1 and Q4 are the lowest and highest quartiles of realized volatility at decision time, taken over the whole test split

| Model | Abstain | Q1 | Q4 | Pinball, abstained | Pinball, retained |
|---|---|---|---|---|---|
| PatchTST + quantile | 5.24 ± 0.32% | 3.45 ± 2.04% | 8.02 ± 2.75% | 0.02127 ± 0.00171 | 0.01855 ± 0.00022 |
| Quantile GBDT | 5.36% | 0.00% | 17.04% | 0.02622 | 0.01765 |
| TFT, original | 5.28 ± 0.13% | 0.00 ± 0.00% | 16.71 ± 0.41% | 0.02584 ± 0.00104 | 0.01791 ± 0.00014 |
| **Ours** | 5.91 ± 0.11% | 0.21 ± 0.05% | **20.07 ± 0.71%** | 0.02612 ± 0.00040 | **0.01730 ± 0.00005** |

<Table 6>은 보류 요구사항이 규칙이 아니라 구조에 의해 충족되었음을 보여준다. 네 행의 보류 규칙은 동일하며
예측 분포만 다르다. 분위수 출력층을 부착하면 보류가 작동하기 시작하고, 변수선택·정적 문맥·매크로 입력이
더해지면 최상위 변동성 분위의 보류율이 20.07%인 반면 최하위 분위는 0.21%에 머물러 보류가 고변동
구간에 집중된다. 이 집중은 세 구성요소 가운데 하나를 빼도 유지된다(5.4절). 예측 폭과 실현 변동의 상관 자체는 TFT 원본이 더 높았다(0.186 대 0.170). 보류가 고변동
구간에 더 몰리는 것과 폭이 변동을 더 잘 따라가는 것은 별개의 성질이다. 보류한 판단은 실제로 더 나빴다. 제안 모형이 보류한 예측의 핀볼 손실은 남긴 예측의
1.510 ± 0.023배이며, 날짜 단위로 짝지은 손실차의 *t* 값은 19.41 ± 1.02이다. 반면 중앙값 부호의 방향
적중률은 보류 0.56 ± 0.02 대 유지 0.55 ± 0.02로 거의 같다. 보류한 판단은 방향을 틀리기 쉬운 판단이 아니라 오차의 크기가
큰 판단이며, 이는 보류가 폭에 걸려 있다는 설계와 일치한다.

핀볼 손실은 변동성이 큰 구간에서 원래 크므로, 고변동 구간을 피하는 규칙이라면 무엇이든 이 비가 1을
넘는다. 규칙과 분위를 그대로 두고 점수만 예측 폭에서 직전 20일 실현변동성으로 바꿔 끼운 통제군과
대조하였고, 보류율 차이가 섞이지 않도록 날짜마다 두 점수의 횡단면 순위 상위 r%를 보류시켜 같은 날 같은
개수를 보류하게 맞추었다. r = 2, 5, 6, 10, 20, 30% 여섯 수준 전부에서 다섯 시드 모두 폭 기준의 비가
통제군보다 높았고(차이 최소 +0.030, 6%에서 +0.064 ± 0.029), 날짜 블록 부트스트랩 1,000회의 95%
신뢰구간도 0을 포함하지 않았다. TFT 원본은 2~6% 구간에서 시드에 따라 부호가 바뀌었고, 분위수 출력층만 부착한
모형은 통제군에 미치지 못하였다. 예측 폭이 단순 변동성 지표보다 나은 보류 기준이라는 결과는 분위수 출력층을
부착하는 것만으로는 생기지 않았다. 분위 GBDT의 폭도 같은 비교에서 통제군보다 나았다(6%에서 +0.071, 95%
신뢰구간 [+0.033, +0.107]). 제안 모형(+0.064 ± 0.029)과 차이가 없으므로 이 성질은 트랜스포머 구조에
고유하지 않다. 제안 모형에 남는 차이는 최상위 변동성 분위로의 집중(20.07% 대 17.04%)과 남긴 후보의
손실(0.01730 대 0.01765)이다.

보류율의 절대 수준을 정하는 임계값 분위는 시험구간을 참조하여 선택되었으므로 5.91%라는 값은 표본 밖
수치가 아니다. 다만 앞 문단의 통제군 비교는 날짜마다 같은 개수를 보류시켜 보류율을 맞춘 상태에서 여섯 수준
전부로 수행하였으므로, 이 비교의 결론은 임계값 선택에 의존하지 않는다. 분위를 95에서 90, 75로 낮추면 전체 보류율이 11.71%, 28.08%로 오르는 동안 최상위 변동성 분위의 보류율도 36.22%,
68.72%로 함께 올라 분위별 집중과 모형 간 순서가 모두 유지된다. 

### 5.4 구성요소 제거

분위수 출력층만 단 PatchTST와 제안 모형 사이에는 변수선택망, 정적 문맥, 매크로 크로스어텐션이 한꺼번에 더해져
있다. 개선이 어디서 오는지 가리기 위해 셋을 하나씩 제거한 변형을 같은 설정과 시드로 학습하였다. 변수선택망은
매크로 경로와 같은 선형 합치기로 바꾸었고, 매크로 제거 변형은 크로스어텐션 없이 종목 표현을 출력층으로 보내며,
정적 문맥 제거 변형은 문맥 벡터를 0으로 두어 변수선택의 조건화와 결합 뒤 더하기를 함께 껐다. 초매개변수는
<Table 2> 그대로이며, 보고 지표는 실험 전에 시험구간 핀볼 손실, 최상위 변동성 분위의 보류율, 보류율 6%에서
맞춘 통제군 대비 비 차이의 셋으로 정하였다. 결과는 <Table 7>과 같다.

Table 7. Removing one component at a time (test split, mean over five seeds 42-46 with the standard deviation). 'Ours lower' counts the seeds on which the proposed model has the lower test loss; 'Ratio gap at 6%' is the abstained-to-retained loss ratio of the width rule minus that of the trailing 20-day volatility control at a matched 6% abstention rate (Section 5.3)

| Variant | Params | Test pinball | vs baseline | Ours lower | Q4 abstain | Ratio gap at 6% |
|---|---|---|---|---|---|---|
| Ours | 0.500M | 0.017826 ± 0.000059 | +4.95% | - | 20.07 ± 0.71% | +0.064 ± 0.029 |
| Variable selection to linear merge | 0.117M | 0.017837 ± 0.000078 | +4.89% | 3/5 | 19.90 ± 0.18% | +0.057 ± 0.007 |
| No macro cross-attention | 0.468M | 0.017786 ± 0.000029 | +5.16% | 2/5 | 20.51 ± 0.46% | +0.080 ± 0.035 |
| No static context | 0.456M | 0.017797 ± 0.000010 | +5.10% | 1/5 | 20.18 ± 0.30% | +0.067 ± 0.021 |

세 변형 모두 세 지표에서 제안 모형과 구별되지 않았다. 시험구간 손실의 Welch 검정은 *p* = 0.81, 0.21, 0.33이고, 짝지은
비교에서 제안 모형이 낮은 시드는 다섯 가운데 셋, 둘, 하나였다. 보류율 6%의 통제군 비교는 열다섯 실행 가운데 열넷에서 95% 신뢰구간이
0을 포함하지 않았다. 따라서 개선은 어느 한 구성요소에서 오지 않는다. 설명은 둘로 남는다. 세 변형이 공유하는 채널 축약 구조 자체가
원인이거나, 구성요소들이 서로를 대신하는 것이다. 둘 이상을 함께 제거하지 않았으므로 이 둘은 가려지지 않는다. 선형 합치기 변형은 모수가
4분의 1이면서 결과가 같았으므로, 운용에서는 이 변형이 더 가벼운 대안이다.

## 6. 결론

본 연구는 주가 예측 모형이 확신하지 못하는 시점을 스스로 가려내는 문제를 다루었다. 연산량이 적은 예측 모형이 여러 분위수를 함께 출력하도록
바꾸고, 예측 구간의 폭이 평소보다 넓은 시점에서 판단을 보류하는 규칙을 적용하였다. 코스피200 구성종목 201개의 2015년 이후 일별
자료로 평가하였다.

판단 보류는 의도한 대로 작동하였다. 보류는 변동성이 큰 시기에 몰렸고, 최상위 변동성 분위에서 20.07%인 반면 최하위 분위에서는
0.21%였다. 보류한 예측의 오차는 실행한 예측의 1.5배였다. 이 차이는 보류 비율을 같게 맞춘 여섯 수준 전부에서 최근 변동성만 보고
보류하는 규칙보다 컸다. 같은 성질이 분위수 출력층만 부착한 모형에서는 나타나지 않았으므로, 출력 형식을 바꾸는 것만으로는 얻어지지 않는다.
다만 분위 그래디언트 부스팅도 같은 비교에서 대등하였으므로 이 성질이 트랜스포머 구조에 고유한 것은 아니다.

예측 분포의 정확도와 자원 사용은 다음과 같다. 제안 모형의 분위 손실은 검증구간과 시험구간 모두에서 TFT 원본보다 낮았고, 조기종료에 쓰이지
않은 시험구간에서는 다섯 시드 전부가 같은 방향이었으며 날짜 단위 DM 검정도 다섯 시드 모두 유의하였다. 학습 시간은 3.8배, 활성값
메모리는 5.1배 적었다. 그러나 분위 그래디언트 부스팅은 그보다도 적은 자원으로 학습되므로, 자원 절감은 트랜스포머 계열 안에서의 비교다.
중앙값만 떼어 평균절대오차로 재면 제안 모형은 상수 기준선과 같다. 본 연구가 더하는 것은 예측값의 정확도가 아니라 예측을 실행하지 말아야 할
시점의 선별이다.

본 연구는 예측 손실로 평가하였으며 거래비용을 포함한 투자 성과는 측정하지 않았다. 유니버스는 현재 시점의 구성종목이고, 입력 설계와 보류
임계값 선택에 평가 기간 자료가 일부 참고되었다. 구성요소를 하나씩 제거하는 실험으로는 개선이 채널 축약 구조 자체에서 오는지 구성요소 간
대체에서 오는지 가리지 못하였다(5.4절). 예측 구간의 포함률은 71.6%로 명목 80%에 못 미쳐(등각 분위회귀 보정[12] 후 73.9%)
구간을 신뢰구간으로 해석할 수는 없으나, 보류는 폭의 순위만 쓰므로 영향을 받지 않는다.

## 7. 후속 연구

후속 연구로 두 가지를 계획한다. 첫째, 구성요소를 둘 이상 함께 제거한 변형을 학습하여 개선의 출처가 채널 축약 구조인지 구성요소 간 대체인지
가린다. 둘째, 비교 이후 새로 쌓인 자료를 평가에 쓰지 않은 구간으로 두어 여러 시장 국면에서 다시 검증한다.

## References

[1] O. B. Sezer, M. U. Gudelek, and A. M. Ozbayoglu, "Financial time series forecasting with deep learning: A systematic literature review: 2005-2019," *Applied Soft Computing*, Vol.90, 106181, 2020.

[2] T. S. Kim and S. Lee, "Deep learning-based stock fluctuation prediction according to overseas indices and trading trends," *KIPS Transactions on Software and Data Engineering*, Vol.10, No.9, pp.367-374, 2021.

[3] A. Vaswani et al., "Attention is all you need," in *Advances in Neural Information Processing Systems*, pp.5998-6008, 2017.

[4] H. Zhou et al., "Informer: Beyond efficient transformer for long sequence time-series forecasting," in *Proceedings of the AAAI Conference on Artificial Intelligence*, pp.11106-11115, 2021.

[5] H. Wu, J. Xu, J. Wang, and M. Long, "Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting," in *Advances in Neural Information Processing Systems*, pp.22419-22430, 2021.

[6] A. Zeng, M. Chen, L. Zhang, and Q. Xu, "Are transformers effective for time series forecasting?," in *Proceedings of the AAAI Conference on Artificial Intelligence*, pp.11121-11128, 2023.

[7] Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, "A time series is worth 64 words: Long-term forecasting with transformers," in *Proceedings of the International Conference on Learning Representations*, 2023.

[8] B. Lim, S. O. Arik, N. Loeff, and T. Pfister, "Temporal fusion transformers for interpretable multi-horizon time series forecasting," *International Journal of Forecasting*, Vol.37, No.4, pp.1748-1764, 2021.

[9] I. K. Kim, D. H. Kim, and J. Lee, "Temporal fusion transformers and deep learning methods for multi-horizon time series forecasting," *KIPS Transactions on Software and Data Engineering*, Vol.11, No.2, pp.81-86, 2022.


[10] T. Kim, J. Kim, Y. Tae, C. Park, J.-H. Choi, and J. Choo, "Reversible instance normalization for accurate time-series forecasting against distribution shift," in *Proceedings of the International Conference on Learning Representations*, 2022.

[11] R. Koenker and G. Bassett, "Regression quantiles," *Econometrica*, Vol.46, No.1, pp.33-50, 1978.

[12] Y. Romano, E. Patterson, and E. J. Candes, "Conformalized quantile regression," in *Advances in Neural Information Processing Systems*, pp.3543-3553, 2019.

[13] C. K. Chow, "On optimum recognition error and reject tradeoff," *IEEE Transactions on Information Theory*, Vol.16, No.1, pp.41-46, 1970.

[14] R. El-Yaniv and Y. Wiener, "On the foundations of noise-free selective classification," *Journal of Machine Learning Research*, Vol.11, pp.1605-1641, 2010.

[15] Y. Geifman and R. El-Yaniv, "Selective classification for deep neural networks," in *Advances in Neural Information Processing Systems*, pp.4878-4887, 2017.

[16] T. Fischer and C. Krauss, "Deep learning with long short-term memory networks for financial market predictions," *European Journal of Operational Research*, Vol.270, No.2, pp.654-669, 2018.

[17] S. Gu, B. Kelly, and D. Xiu, "Empirical asset pricing via machine learning," *The Review of Financial Studies*, Vol.33, No.5, pp.2223-2273, 2020.

[18] H. J. Kim and Y. S. Jung, "A stock price prediction based on recurrent convolution neural network with weighted loss function," *KIPS Transactions on Software and Data Engineering*, Vol.8, No.3, pp.123-128, 2019.

[19] S. Makridakis, E. Spiliotis, and V. Assimakopoulos, "Statistical and machine learning forecasting methods: Concerns and ways forward," *PLOS ONE*, Vol.13, No.3, e0194889, 2018.

[20] Y. Wang et al., "TimeXer: Empowering transformers for time series forecasting with exogenous variables," in *Advances in Neural Information Processing Systems*, 2024.

[21] F. X. Diebold and R. S. Mariano, "Comparing predictive accuracy," *Journal of Business & Economic Statistics*, Vol.13, No.3, pp.253-263, 1995.

[22] W. K. Newey and K. D. West, "A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix," *Econometrica*, Vol.55, No.3, pp.703-708, 1987.

[23] D. Harvey, S. Leybourne, and P. Newbold, "Testing the equality of prediction mean squared errors," *International Journal of Forecasting*, Vol.13, No.2, pp.281-291, 1997.

[24] G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," in *Advances in Neural Information Processing Systems*, pp.3146-3154, 2017.

[25] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, Vol.12, pp.2825-2830, 2011.

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

관심분야 : Time-Series Forecasting, Quantile Regression, Deep Learning, Algorithmic Trading

#### 김 지 민

![김지민](images/author_kimjimin.jpeg)

https://orcid.org/0009-0004-4645-227X

e-mail : bckck@naver.com

2020년～현  재 동아대학교 컴퓨터공학과 학사과정

관심분야 : Deep Learning, Financial Time-Series Analysis, Uncertainty Estimation

#### 권 오 재

![권오재](images/author_kwonohjae.jpeg)

https://orcid.org/0009-0002-9537-5317

e-mail : rnjsdhwo4@naver.com

2021년～현  재 동아대학교 컴퓨터공학과 학사과정

관심분야 : Machine Learning, Time-Series Forecasting, Quantitative Finance

#### 송 하 얼

![송하얼](images/author_songhaeol.jpeg)

https://orcid.org/0009-0007-5429-4154

e-mail : haeolsong@gmail.com

2023년～현  재 동아대학교 컴퓨터공학과 학사과정

관심분야 : Deep Learning, Transformer Models, Data Analysis

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
