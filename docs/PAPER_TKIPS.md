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

Stock return forecasting is a notoriously difficult problem: because markets are
highly volatile, no model can be accurate at every moment. Existing artificial
intelligence models nevertheless report a single predicted value even at the moments
they are unsure about, which increases the risk borne by the investor. This paper
starts from the position that a stock forecasting model should not only try to be
accurate, but should also be able to withhold its decision when it cannot be
confident. To this end, the proposed model uses a lightweight deep learning
architecture that reports, together with its forecast, the width of the range in which
the future return is expected to fall as a measure of forecast uncertainty, and
withholds the decision at the moments when that uncertainty rises well above its usual
level. In an evaluation on daily data for 201 KOSPI 200 stocks since 2015, the moments
at which the model withheld its decision were indeed the moments at which forecast
errors were large. The proposed model gave the most accurate predictive distribution
among the four models compared, and its criterion selected high-risk moments better
than a simple rule based on recent market volatility. This study shows empirically that the
contribution of an artificial intelligence model for stock forecasting lies not in
improving point-forecast accuracy, but in reliably selecting the moments at which a
prediction should be withheld.

<!-- 229 단어 (공백 분리 기준; 제한 300). 한 문단, 표·참고문헌·장 번호 없음 -->

**Keywords :** Stock Return Forecasting, Patch-Based Transformer, Quantile Regression,
Forecast Uncertainty, Abstention

## 판단 보류가 가능한 경량 주가 수익률 분위 예측 모형

<!-- AUTHORS:BEGIN -->
박 성 준†⋅김 지 민†⋅권 오 재†⋅송 하 얼†⋅서 정 일††⋅유 주 한†††
<!-- AUTHORS:END -->

### 요     약

주가 예측은 대표적인 난제 중 하나로, 시장의 높은 변동성으로 인해 모든 시점에서 정확한 수익률을 맞히는 것은 불가능에 가깝다. 그럼에도 기존
인공지능 모형들은 불확실한 시점조차 단일 예측값을 강제로 출력하여 투자 위험을 키우는 한계가 있었다. 본 연구는 주가 예측 모형이 맞히는 것에만
집중하기보다 확신할 수 없는 순간 스스로 판단을 보류하는 능력을 갖추어야 한다는 문제 의식에서 출발한다. 이를 위해 경량화된 딥러닝 구조를
바탕으로 미래 수익률의 예측 불확실성을 나타내는 예측 폭을 함께 산출하고, 불확실성이 평소보다 크게 치솟는 구간에서는 예측을 스스로 보류하는
메커니즘을 구현했다. 2015년 이후 코스피200 구성종목 201개의 일별 자료를 통한 검증 결과, 모형이 보류를 선언한 시점은 실제로 예측
오차가 크게 발생하는 위험 구간임이 확인되었다. 제안 모형의 예측 분포는 비교한 네 모형 가운데 가장 정확했으며, 단순 시장 변동성 지표를
따르는 수준을 넘어 예측이 틀리기 쉬운 고위험 시점을 유의미하게 선별해냈다. 본 연구는 주가 예측 인공지능의 기여가 단일 값 예측의 정확도 갱신에 있는 것이 아니라 판단할 수 없는 시점을 신뢰성 있게 선별하여
예측을 보류하는 것에 있음을 실증적으로 입증한다.

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

주가 예측 모형은 과거 가격과 거래 자료에서 규칙을 학습하여 미래 수익률을 추정한다. 그러나 주식 시장의 동학은 시시각각 변화하므로 특정
시기에 유효했던 규칙이 다음 시기에는 적용되지 않을 수 있다. 따라서 어떠한 예측 모형도 모든 시점에서 일정한 정확도를 유지하기는 어렵다.

문제는 모형이 스스로의 오예측 가능성을 사용자에게 전달하지 못한다는 점이다. 대부분의 예측 모형은 미래 값을 단일 수치로 출력한다. 이러한
방식을 단일 값 예측(point forecast)이라 한다. 단일 값 예측은 하나의 값만 제시하므로 사용자는 해당 예측이 모형의 신뢰도가 높은
시점에서 산출된 것인지 판단할 수 없다. 금융 투자에서는 오예측에 따른 손실 위험이 크므로 이러한 결함은 단순한 정확도 저하보다 심각한 문제가
된다.

따라서 예측 모형은 확신할 수 없는 시점을 스스로 가려내어 예측을 출력하지 않는 메커니즘을 갖추어야 한다. 이처럼 모형이 일부 시점에 대해
예측을 제시하지 않는 것을 판단 보류(abstention)라 한다. 본 연구는 주가 예측에서 판단 보류를 수행하는 모형을 다룬다.

판단 보류를 실행하려면 모형이 스스로의 불확실성을 수치화하여 출력해야 한다. 본 연구는 해당 수치를 예측 분포에서 도출한다. 단일 값 대신
여러 분위수(quantile)를 동시에 출력하도록 모형을 구성하는 방식이다. 분위수는 실제 값이 해당 수치보다 작을 확률이 정해진 수준이 되는
값이다. 예를 들어 하위 10% 분위수와 상위 10% 분위수를 함께 출력하면 두 값 사이의 구간이 미래 수익률의 예상 범위가 된다. 이 구간이
넓다는 것은 모형이 해당 시점의 결과를 구체적으로 특정하지 못함을 의미한다. 본 연구는 예측 구간이 평소보다 넓은 시점에서 판단을 보류한다.

기존 연구에서는 두 기능이 함께 구현되지 않았다. 예측 분포를 산출하는 대표적 모형인 Temporal Fusion Transformer, 즉
TFT는 여러 분위수를 동시 출력하지만 입력 기간의 모든 시점을 비교하는 연산 구조로 인해 학습 시간과 메모리 소모가 크다. 반대로 연산량을
줄인 최신 시계열 예측 모형들은 단일 예측값만 출력하여 판단 보류의 근거가 되는 구간 정보를 제공하지 못한다.

본 연구는 연산량이 적은 예측 모형이 여러 분위수를 동시 출력하도록 구조를 개선하고 해당 출력에 기반한 판단 보류 규칙을 정의했다.
코스피200 구성종목 201개의 2015년 이후 일별 자료로 이를 평가했다. 본 연구의 기여는 세 가지다.

첫째, 예측 구간 폭을 기준으로 판단을 보류하면 오차가 큰 예측을 효과적으로 걸러낼 수 있다. 보류된 예측의 오차는 실행된 예측의 약
1.5배였다. 이 결과는 동일한 보류 비율 조건에서 최근 변동성 기준 통제군보다 우수했다.

둘째, 이러한 성질은 단순 분위수 출력층 추가만으로는 형성되지 않는다. 분위수 출력만 부착한 모형은 동일한 비교에서 변동성 통제군을 상회하지
못했다.

셋째, 제안 모형의 중앙값 예측은 상수 기준선보다 특별히 정확하지 않았다. 즉 본 연구의 기여는 단일 값 예측의 정확도 갱신에 있는 것이
아니라 예측을 실행하지 말아야 할 시점을 신뢰성 있게 선별하는 데 있다.

본 논문의 구성은 다음과 같다. 2장에서는 판단 보류에 필요한 요소를 기준으로 기존 연구를 정리한다. 3장에서는 제안 모형과 판단 보류 규칙을
설명한다. 4장에서는 실험에 사용한 자료를, 5장에서는 실험 결과를 제시한다. 6장에서 결론을 내리고 7장에서 후속 연구를 밝힌다.

## 2. 관련 연구

### 2.1 주가 예측에 쓰인 딥러닝 모형

금융 시계열에 딥러닝을 적용한 연구는 순환신경망과 합성곱신경망을 중심으로 축적되었다[1]. 장단기기억 신경망으로 미국 주식의 등락 방향을 예측한
연구[16], 여러 기계학습 기법을 자산가격 결정에 적용한 연구[17]가 대표적이다. 국내에서는 해외지수와 투자자별 매매 동향을 입력으로 삼은
등락 예측[2], 가중 손실 함수를 적용한 순환 합성곱 구조[18]가 보고되었다. 이 연구들은 입력 변수와 신경망 구조를 바꾸어 정확도를 높이는
데 집중하였고, 출력은 모두 단일 값 예측이다. 따라서 모형이 어느 시점에서 확신하지 못하는지를 사용자가 알 수 없고, 1장에서 제기한 판단 보류도
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
선형으로 확장된다. 이 계열의 단점은 출력이 단일 값 하나라는 것이며, 판단 보류의 근거가 되는 구간이 나오지 않는다.

### 2.4 판단 보류를 다룬 연구

기계학습에서 판단 보류는 오랜 기간 다뤄진 주제이다. 초기 연구에서는 모형이 확실하지 않을 때 답변을 피하는 선택지를
기각(reject)이라 불렀으며[13], 틀린 답을 냈을 때의 손실과 답변을 포기했을 때의 손실을 비교하여 어느 입력에서 답변을 유보할지
결정하는 방식을 취했다.

이후 이 개념은 선택적 예측(selective prediction)으로 확장되었으며, 모형이 실제로 답변한 비율(coverage)과 그 표본에서의
오차를 함께 평가하는 방식으로 발전했다[14]. 다만 답변 비율을 낮추면 남은 표본의 오차는 자연히 줄어들기 때문에, 두 지표를 함께
고려해야만 공정한 모형 비교가 가능하다. 심층 신경망 분야에서는 학습이 완료된 모형의 신뢰도 점수를 기준으로 답변 여부를 결정하는 방법이
제안되었다[15].

하지만 이러한 기존 연구들은 모두 분류(classification) 문제를 전제로 하며, 모형이 출력하는 각 범주별 확률을 이용해 판단 보류를
결정한다. 따라서 수치 하나를 정확히 예측해야 하는 회귀(regression) 문제에는 범주별 확률이 존재하지 않아, 기존의 기준을 그대로
적용하기 어렵다는 한계가 있다.

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
둘째는 시장 전체를 나타내는 지표 13개의 같은 기간 계열이며, 이하 매크로 계열이라 부른다. 셋째는 섹터, 시가총액 구간, 요일처럼 한 번 정해지면 윈도우 안에서 바뀌지
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
채널을 선형결합으로 축약했고, 종목 채널에는 채널별 가중치를 학습하는 변수선택망을 두었다. 가중치가 정적 문맥에 따라 달라져 섹터와 시가총액
구간별로 다른 비중을 줄 수 있기 때문이다(식 2, 3).

&nbsp;&nbsp;&nbsp;&nbsp;*v*_{n} = softmax( GRN( [ flat(*E*_{n}) ; *c*_{s} ] ) ) ∈ ℝ^{C},&nbsp;&nbsp; *n* = 1, …, *N*&nbsp;&nbsp;&nbsp;&nbsp;(2)

&nbsp;&nbsp;&nbsp;&nbsp;*Z*_{n} = Σ_{i} *v*_{n,i} · GRN_{i}(*E*_{n,i})&nbsp;&nbsp;&nbsp;&nbsp;(3)

여기서 *C*는 채널 수, *N*은 패치 수, *E*_{n,i}는 패치 *n*·채널 *i*의 임베딩이고, flat(*E*_{n})은 그 패치의
모든 채널 임베딩을 이어 붙인 벡터, *c*_{s}는 정적 임베딩에서 얻은 문맥 벡터, GRN은 TFT의 게이트 잔차
신경망이다.

변수선택망을 인코더 뒤에 두면 인코더가 채널 수만큼 반복되지만, 앞에 두면 채널이 먼저 하나로 합쳐져 한 번만 실행된다. 다만 종목 채널까지
선형결합으로 바꾼 변형도 정확도와 보류 행태가 같았다(5.4절). 연산을 줄이는 것은 인코더 앞의 채널 축약 자체이며, 학습된 가중치가
정확도를 더하지는 않았다.

#### 2) 매크로 크로스어텐션의 방향

매크로란 개별 종목이 아니라 시장 전체의 움직임을 나타내는 변수를 말한다. 본 연구가 쓴 13개 계열은 코스피·코스닥 지수 관련
상품과 해외 지수 상장지수펀드(ETF)의 일별 가격이며(4.2절), 국내 주식이 시장 전체나 해외 증시와 함께 움직이는 부분을 모형에
알려주는 몫을 한다. 모든 종목이 같은 날 같은 값을 보므로 종목별 지표와는 따로 처리해야 한다.

이 매크로 계열은 크로스어텐션으로 종목 표현에 주입한다. 종목 표현을 질의로, 매크로 부호화를 키와 값으로 둔다. 방향을 반대로 두면 출력이 매크로
시퀀스의 형상을 따라 종목별 예측이 나오지 않는다. 외생 변수를 어텐션으로 주입하는 발상은 TimeXer[20]에도 있으며, 본 연구는 매크로를
패치 단위로 유지한다는 점이 다르다.

#### 3) 분위 단조성의 구조적 보장과 출력층 초기화

분위 교차를 손실 벌점이 아니라 누적 소프트플러스 구조로 원천 차단하였다(식 4).

&nbsp;&nbsp;&nbsp;&nbsp;*q*_{10} = *b*,&nbsp;&nbsp; *q*_{50} = *q*_{10} + softplus(δ_{1}),&nbsp;&nbsp; *q*_{90} = *q*_{50} + softplus(δ_{2})&nbsp;&nbsp;&nbsp;&nbsp;(4)

*b*와 δ_{1}, δ_{2}는 출력층 마지막 선형층의 세 출력이다. 교차가 발생하면 보류 폭이 음수가 되어 하위 규칙 전체가 무효화되므로
구조적 보장이 필요하다. 5.1절의 세 트랜스포머 모형은 이 출력층을 공유하며, 다섯 시드의 검증·시험 구간 647,880건에서 교차는 한 건도
없었다. 이는 구조가 구현에서 지켜졌음을 확인한 것이지 보장이 없는 모형과 비교한 결과가 아니다.

출력층의 초기화도 성능에 영향을 주었다. 기본 초기화에서는 softplus(0) = 0.693이라 초기 구간 폭이 목표 분포와
어긋난 지점에 놓여 기준선에 미달하므로, 최종층 편향을 학습자료의 무조건부 분위수로 설정하여 해소하였다.

### 3.4 판단 보류 규칙

비교하는 모든 모형에 동일한 규칙을 적용한다. 예측 구간의 폭이 최근 분포의 95분위를 초과하면 해당 시점의 판단을 보류한다. 여기서 최근
분포란 판단일 기준 달력 90일(거래일 기준 약 60일) 동안의 폭을 전 종목 통합하여 모은 것이다. 전체 구간의 폭 분포를 사용하면 미래
날짜의 폭 정보가 과거 판단에 누수되므로, 이를 방지하기 위한 조치이다. 임계값으로 95분위를 선정한 근거와 다른 값으로 변경했을 때의 영향은
5.3절에서 다룬다.

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

자료는 종목이 아닌 날짜 기준으로 2022년까지 학습, 2023년 검증, 2024년 이후 시험 구간으로 분할했다. 종목 기준으로 나누면 학습과
시험이 같은 날짜를 공유하여 정보 누수(data leakage)가 발생하기 때문이다. 5일 뒤 수익률을 예측하므로 구간 경계마다 5거래일의 격리
기간을 두어 경계에 걸친 표본의 정답이 다음 구간으로 넘어가지 않게 했으며, 정규화 통계량은 학습 구간의 행에서만 계산했다.

입력 윈도우(120거래일)가 한 구간 내에 완전히 포함되는 경우만 표본으로 삼았다. 윈도우가 앞 구간에 걸쳐도 격리 기간이 정답의 중첩을 막으므로
누수는 아니지만, 그 경우 검증 표본의 입력 대부분이 학습에서 이미 본 행이 된다. 이 조건에 따라 각 구간의 초기 120거래일은 제외된다.
최종 검증 표본은 2023년 7월 3일부터 12월 28일까지 122거래일, 190종목, 23,086건이며, 시험 표본은 2024년 7월 3일부터
2026년 10월 8일까지 550거래일, 198종목, 106,490건이다. 종목 수 감소는 120거래일을 채우지 못한 종목이 제외된 결과이며,
자료는 2026년 10월 16일 수집분으로 동결했다.

학습 표본은 명목상 264,653건이나 연속된 표본 간 입력 120일 중 119일, 정답 5일 중 4일이 겹쳐 실질 정보량은 이보다 적다. 모수를
늘리면 1~2회차에 과적합되어 0.50M으로 고정했다.

모든 실험은 입력을 보지 않고 학습 자료 전체의 분위수 세 개를 상수로 출력하는 예측기를 기준선으로 삼는다. 모형이 이를 상회하지 못할 경우,
초매개변수 조정의 문제가 아니라 변수와 목표변수 설계의 실패로 간주한다.

## 5. 실험 결과

### 5.1 동일조건 정확도 비교

판단 보류는 모형이 출력하는 예측 구간에 의존하므로, 먼저 이를 생성하는 분포의 타당성을 검증한다. 비교 대상인 세 트랜스포머 모형은 동일한
자료, 분할, 학습 절차, 조기종료 규칙을 공유하며, 무조건부 분위수 기준선 손실도 0.012175로 동일하다. 다만 PatchTST 계열 대조군은
구조적 한계로 매크로 및 정적 변수를 입력받지 못하며, 초매개변수는 제안 모형의 설정값을 동일 적용하여 대조군의 최적값은 아니다.

세 트랜스포머 모형은 동일한 혼합정밀도 설정으로 캐글 GPU 환경에서 학습하고 단일 장비에서 평가했다. 분위 GBDT는 적합 과정에 난수 요소가
없어 1회만 학습했다. 단일 값 예측을 수행하는 원본 PatchTST는 비교 표에서 제외했다. 단일 예측값을 세 분위로 동일하게 복제하여 평가할
경우 기준선 대비 47.79% 열위한 성능을 보이는데, 이는 구조적 성능 차이가 아닌 손실함수 정의의 불일치에서 기인한다.

Table 4. Forecast accuracy on the validation and test splits (three-quantile pinball loss; mean over five seeds 42-46 with the standard deviation, single run for the quantile GBDT). Lower is better, and 'vs baseline' is the improvement over the unconditional-quantile predictor, which scores 0.012175 on validation and 0.018754 on test

| Model | Params | Validation | vs baseline | Test | vs baseline | Best epoch |
|---|---|---|---|---|---|---|
| PatchTST + quantile head | 0.035M | 0.012166 ± 0.000022 | +0.07% | 0.018686 ± 0.000130 | +0.36% | 1-5 |
| TFT, original | 0.687M | 0.011935 ± 0.000020 | +1.97% | 0.018325 ± 0.000091 | +2.28% | 1-2 |
| Quantile GBDT | 375 trees | 0.011890 | +2.34% | 0.018107 | +3.45% | - |
| **Ours** | 0.500M | 0.011889 ± 0.000012 | +2.35% | **0.017826 ± 0.000059** | **+4.95%** | 2-10 |

제안 모형은 검증 및 시험 구간 모두에서 TFT 원본보다 낮은 손실을 기록했다. 시드를 짝지은 비교에서도 모든 구간에서 다섯 시드 전체가 일관된
방향성을 보였다. 구간별 부호검정 *p* 값은 0.031 단측이다. 두 구간은 동일한 체크포인트에서 파생되었으므로 독립 시행으로 합산하지 않는다.

검증구간에서 두 모형 간 격차는 핀볼 손실 기준 0.000046으로, 기준선 대비 개선폭이 16% 수준에 그쳐 근소하게 우수한 정도이다. 반면
조기종료 및 최적 회차 선택의 영향이 배제된 시험구간에서는 격차가 0.000499로 벌어지며 다섯 시드의 손실 범위가 겹치지 않았다. 제안
모형의 범위는 0.017739부터 0.017895, TFT 원본은 0.018205부터 0.018407이다.

평균 차이에 대한 Welch 검정 결과 검증구간 *t* 값은 4.47, *p* 값은 0.004이며, 시험구간 *t* 값은 10.33, *p* 값은
0.001 미만이다. 동일 표본의 날짜별 손실 계열에 Newey-West 4시차[22] 및 Harvey-Leybourne-Newbold 소표본
보정[23]을 적용한 Diebold-Mariano 검정[21]을 실시한 결과, 550거래일의 시험구간에서는 다섯 시드 모두 *p* 값 0.022
이하로 유의했다. 반면 122거래일의 검증구간에서는 다섯 시드 모두 *p* 값이 0.16에서 0.56으로 유의하지 않았다. 이는 날짜 단위 표본
검정의 특성상 단기 구간에서 검정력이 저하되기 때문이다.

분위수 출력층만 추가한 PatchTST는 검증구간에서 +0.07%, 시험구간에서 +0.36% 개선되어 상수 예측을 미미하게 상회했으며, 다섯 시드
중 네 시드가 1회차에서 최적값에 도달했다. 시험구간에서 성능 격차가 벌어진 주된 원인은 학습 곡선의 차이에 있다. TFT 원본은 다섯 시드
모두 1회차 또는 2회차에서 최적값에 도달한 뒤 과적합을 보인 반면, 제안 모형의 최적 회차는 2회차에서 10회차 사이에 분포했다. 세 모형을
단정밀도로 맞추어 1회씩 학습하더라도 성능 순서는 유지된다.

중앙값 예측만을 이용해 평균절대오차로 평가할 경우 제안 모형은 0.053439 ± 0.000142로 상수 기준선인 0.053447과 대등하며,
TFT 원본은 0.053707로 기준선보다 오히려 나쁘다. 따라서 핀볼 손실의 성능 개선은 단일 값 예측의 정확도가 아닌 예측 구간에서 비롯된다.

트리 계열 기준모형으로는 분위 손실을 사용하는 히스토그램 그래디언트 부스팅(LightGBM[24]의 방식을 따른 scikit-learn[25]
구현)을 동일한 분할 및 윈도우 표본에서 평가했다. 입력 변수는 동적 30채널의 당일 값과 1·5·20일 시차, 매크로 13채널, 정적 범주형
3종을 포함해 총 136개이다. 분위별로 독립 모형을 구축하고 검증구간 손실을 기준으로 트리 수를 75개, 25개, 275개로 결정했다.
검증구간 손실은 0.011890으로 제안 모형의 0.011889와 유사했다. 시험구간 손실은 0.018107로 TFT 원본보다는 우수하지만 제안
모형에는 미치지 못했으며, 제안 모형의 다섯 시드는 최댓값 0.017895를 포함해 모두 이보다 우수한 성능을 나타냈다.

### 5.2 계산 비용

자원 사용량은 본 연구의 주된 기여라기보다, 제안 모형이 기존 분포 예측 모형을 대체할 실용성이 있는지 확인하는 조건이다.

Table 5. Computational cost measured on one device (Mac, Metal GPU backend; batch 64, 120-day input,
201 stocks). The quantile GBDT row is one full fit of the three quantile models on the CPU (8 cores), so it has
neither epochs nor activation memory. The text below gives the measurement procedure

| Model | s / epoch | Latency / stock | Universe pass | Activation | Param memory |
|---|---|---|---|---|---|
| PatchTST (+ quantile head) | 110.8 | 0.092 ms | 18.6 ms | 172.8 MB | 0.14 MB |
| TFT, original | 550.8 | 0.693 ms | 139.4 ms | 679.1 MB | 2.75 MB |
| Ours | 143.5 | 0.143 ms | 28.7 ms | 132.9 MB | 2.00 MB |
| Quantile GBDT (CPU) | 26.1 (full fit) | 0.068 ms | 13.4 ms | - | 2.13 MB |

epoch당 학습 시간은 40배치 학습 스텝 시간의 중앙값에 학습 윈도우가 생성하는 4,135배치를 곱해 산출했다. 전 종목 1회 통과 시간은
201개 종목 순전파 21회 수행의 중앙값이며, 종목당 지연시간은 이를 201로 나눈 값이다. 활성값(activation) 메모리는 역전파 그래프를
유지하기 위해 순전파 시 추가 할당된 장치 메모리를 의미한다.

![Fig. 2](images/fig2_cost_accuracy.png)

Fig. 2. Forecast accuracy against inference cost. The vertical axis is the pinball improvement
over the unconditional-quantile baseline on validation (<Table 4>); the test-split gap is wider.
The horizontal axis is per-stock inference latency and the circle area is activation memory
(<Table 5>). The quantile GBDT (open diamond) runs on the CPU, so its marker size does not
apply. The original point-forecast PatchTST is omitted; its pinball loss falls far below the
plotted range.

제안 모형은 TFT 원본 대비 학습 시간이 3.8배 짧고 활성값 메모리를 5.1배 적게 사용하면서도 더 낮은 손실을 기록했다. 이는 입력을 구간
단위로 묶어 어텐션 대상 길이를 120에서 24로 줄이고, 채널을 인코더 앞에서 하나로 통합하여 반복 연산을 제거한 결과이다. 이러한 효율성
차이는 환경이 바뀌어도 유지되어, 캐글 GPU 기준 epoch당 학습 시간은 제안 모형이 41.8초, TFT 원본이 157.1초를 기록했다.

한편 분위 GBDT의 계산 비용은 이보다 더 낮다. 세 분위 모형 전체를 1회 학습하는 데 CPU 기준 26.1초가 소요되어 제안 모형의
1 epoch 학습 시간보다 짧았으며, 종목당 추론 시간 역시 0.068 ms로 제안 모형의 절반 이하 수준이다. 따라서 본 연구의 경량화 효과는
트랜스포머 기반 분위 모형(특히 TFT) 대비 상대적 의미를 지니며, 제안 모형이 GBDT 대비 갖는 강점은 계산 비용이 아닌 시험구간 손실의
추가 개선(+4.95% 대 +3.45%)에 있다.

(Fig. 2)는 <Table 4>의 검증 정확도와 <Table 5>의 비용을 한 평면에 놓은 것이다. 세 트랜스포머 모형 가운데 제안 모형이
파레토 전선 위에 있고, 분위 GBDT는 검증 정확도가 같으면서 지연이 더 짧다.

### 5.3 판단 보류 행태

네 분위 모형에 3.4절의 같은 보류 규칙을 적용하고 시험구간 전체에서 집계하였다. 단일 값 예측 모형은 임계값을 적용할 폭이 없어 보류가 정의되지 않으므로 제외하였다.

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

### 5.4 구성요소 제거와 보류의 강건성

분위수 출력층만 결합한 PatchTST와 제안 모형 사이에는 변수선택망, 정적 문맥, 매크로 크로스어텐션이 동시에 추가되어 있다. 본 절에서는 두
가지 사항을 검증한다. 첫째, 판단 보류 특성이 특정 구성요소에 종속되는지 확인한다. 각 요소를 개별적으로 제거하더라도 보류 성질이
유지된다면 본 연구의 결론은 특정 구조적 선택에 국한되지 않는다. 둘째, 5.1절에서 확인된 성능 개선이 어떤 요소에서 기인하는지 규명한다.
세 요소가 동시에 도입되어 직접적인 기여도 산출이 불가능하므로, 동일한 설정과 시드를 적용하여 세 가지 변형 모형을 학습했다.

변수선택망 제거 변형은 해당 부품을 매크로 경로와 동일한 선형결합으로 대체했고, 매크로 제거 변형은 크로스어텐션 없이 종목 표현을 곧바로
출력층으로 전달했다. 정적 문맥 제거 변형은 문맥 벡터를 영벡터로 설정하여 변수선택의 조건화와 결합 후 덧셈 연산을 동시에 비활성화했다.
초매개변수는 <Table 2>의 설정과 동일하며, 사전에 정의한 평가지표는 시험구간 핀볼 손실, 최상위 변동성 분위의 보류율, 보류율 6%
조건에서 통제군 대비 비 차이의 세 가지이다. 실험 결과는 <Table 7>에 정리되어 있다.

Table 7. Removing one component at a time (test split, mean over five seeds 42-46 with the standard deviation). 'Ours lower' counts the seeds on which the proposed model has the lower test loss; 'Ratio gap at 6%' is the abstained-to-retained loss ratio of the width rule minus that of the trailing 20-day volatility control at a matched 6% abstention rate (Section 5.3)

| Variant | Params | Test pinball | vs baseline | Ours lower | Q4 abstain | Ratio gap at 6% |
|---|---|---|---|---|---|---|
| Ours | 0.500M | 0.017826 ± 0.000059 | +4.95% | - | 20.07 ± 0.71% | +0.064 ± 0.029 |
| Variable selection to linear combination | 0.117M | 0.017837 ± 0.000078 | +4.89% | 3/5 | 19.90 ± 0.18% | +0.057 ± 0.007 |
| No macro cross-attention | 0.468M | 0.017786 ± 0.000029 | +5.16% | 2/5 | 20.51 ± 0.46% | +0.080 ± 0.035 |
| No static context | 0.456M | 0.017797 ± 0.000010 | +5.10% | 1/5 | 20.18 ± 0.30% | +0.067 ± 0.021 |

세 변형 모두 세 지표에서 제안 모형과 구별되지 않았다. 시험구간 손실의 Welch 검정은 *p* = 0.81, 0.21, 0.33이고, 짝지은
비교에서 제안 모형이 낮은 시드는 다섯 가운데 셋, 둘, 하나였다. 보류 쪽도 같았다. 최상위 변동성 분위의 보류율은 세 변형에서
19.90~20.51%로 제안 모형의 20.07%와 다르지 않았다. 보류율 6% 조건의 통제군 비교에서는 총 15회 실행 중 14회에서 95%
신뢰구간이 0을 포함하지 않았다. 따라서 판단 보류 메커니즘은 특정 구성요소에 의존하지 않으며, 본 연구의 결론은 이러한 설계 선택에 국한되지
않는다.

반면 성능 개선의 정확한 출처는 본 실험만으로 명확히 규명되지 않았다. 세 변형 모형이 공유하는 채널 축약 구조 자체가 성능 향상의
원인이거나, 개별 구성요소들이 상호 대체 작용을 수행한 결과일 수 있다. 둘 이상의 요소를 동시 제거하는 추가 실험을 수행하지 않아 두
가능성을 완벽히 구분하기는 어렵다. 다만 변수선택망을 선형결합으로 대체한 변형 모형은 모수가 4분의 1 수준임에도 동일한 성능을 기록했으므로,
실제 운용 관점에서 더 가벼운 대안이 될 수 있다.

## 6. 결론

본 연구는 주가 예측 모형이 불확실한 시점을 스스로 가려내는 문제를 다루었다. 연산량이 적은 예측 모형이 여러 분위수를 동시 출력하도록
개선하고, 예측 구간 폭이 평소보다 넓을 때 판단을 보류하는 규칙을 적용했다. 코스피200 구성종목 201개의 2015년 이후 일별 자료로
이를 평가했다.

판단 보류는 의도대로 작동했다. 보류는 고변동 시기에 집중되어 최상위 변동성 분위에서 20.07%, 최하위 분위에서 0.21%를 기록했다.
보류된 예측의 오차는 실행된 예측의 1.5배였으며, 보류율을 동일하게 맞춘 6개 수준 전부에서 최근 변동성 기준 통제군보다 우수했다. 임계값
분위를 95에서 90, 75로 낮추어도 분위별 집중과 모형 간 순서가 유지되어, 이 결론은 임계값 선택에 의존하지 않는다. 이러한 특성은
분위수 출력층 부착만으로는 나타나지 않지만, 분위 그래디언트 부스팅에서도 동일하게 확인되어 트랜스포머 구조 고유의 성질은 아니다.

예측 분포의 정확도와 자원 효율성 측면에서, 제안 모형의 분위 손실은 검증 및 시험구간 모두에서 TFT 원본보다 낮았다. 특히 조기종료에 영향을
받지 않는 시험구간에서는 5개 시드 모두 일관된 방향성을 보였고 날짜 단위 DM 검정에서도 모두 유의했으며, 손실은 비교한 네 모형 가운데 가장
낮았다. 학습 시간은 3.8배, 활성값 메모리는 5.1배 적었으나, 분위 GBDT가 이보다 적은 자원을 소모하므로 자원 절감은 트랜스포머 계열 내
비교에 해당한다. 중앙값 기반 평균절대오차로 평가하면 제안 모형은 상수 기준선과 동등하다. 즉, 본 연구의 핵심 기여는 예측값의 점 정확도가
아니라 예측을 유보할 시점을 선별하는 데 있다.

본 연구는 예측 손실을 기준으로 평가했으며 거래비용을 포함한 투자 성과는 측정하지 않았다. 분석 대상 종목군은 현재 구성종목 기준이며 입력 설계와 보류
임계값 선택에 평가 기간 자료가 일부 반영되었다. 구성요소 제거 실험(5.4절)을 통해 성능 개선이 채널 축약 구조 자체에서 기인하는지 요소 간
대체 효과 때문인지는 명확히 규명하지 못했다. 예측 구간의 포함률은 71.6%(등각 분위회귀 보정[12] 후 73.9%)로 명목 수준인 80%에
미치지 못해 구간을 신뢰구간으로 해석할 수는 없으나, 보류 규칙은 폭의 순위 정보만 활용하므로 영향을 받지 않는다.

## 7. 후속 연구

후속 연구로는 두 가지를 계획한다. 첫째, 복수의 구성요소를 동시에 제거한 변형 모형을 통해 성능 개선의 원인이 채널 축약 구조인지 요소 간
대체 효과인지 규명한다. 둘째, 비교 시점 이후 추가로 축적된 자료를 미평가 구간으로 활용하여 다양한 시장 국면에서 모형의 타당성을 재검증한다.

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
