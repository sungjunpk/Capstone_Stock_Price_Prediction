# 참고문헌 검증 (2026-10-21)

지도교수 피드백 1단계. 원고 `docs/PAPER_TKIPS.md` 의 참고문헌 26편을 각각 검색해 실재 여부,
서지 정확성, 본문 인용 문장과의 내용 일치를 확인하였다. 기억이 아니라 검색 결과로만 판정하였다.

## 요약

| 판정 | 건수 | 번호 |
|---|---|---|
| 유지 (서지·인용 모두 일치) | 23 | [3]~[8], [10]~[15], [17]~[26] |
| 서지 수정 | 1 | [2] |
| 인용 문장 수정 | 2 | [1], [16] |
| 삭제 권고 | 0 | — |

실재가 확인되지 않은 문헌은 없다. 국내 4편([2], [9], [10], [19])은 KCI·KoreaScience·학회 PDF 로 확인하였다.

## 전체 표

| # | 실재 확인 출처 | 서지 대조 | 본문 인용과 내용 일치 | 판정 |
|---|---|---|---|---|
| [1] Sezer et al., Applied Soft Computing 90, 106181, 2020 | arXiv:1911.13288 / scirp 서지 | 일치 | **불일치.** 본문은 "LSTM·CNN을 거쳐 **트랜스포머 계열로 확장**되며 축적"이라고 썼으나, 이 리뷰는 2005–2019 범위이며 초록이 다루는 모형군은 CNN·DBN·LSTM 이다. 트랜스포머·어텐션 언급이 없다 | **인용 문장 수정** |
| [2] Kim & Lee, KIPS TSDE 10(9), 367-374, 2021 | KCI ART002761398 / KoreaScience JAKO202128837710024 / DOI 10.3745/KTSDE.2021.10.9.367 | **영문 제목 불일치.** 정식 제목은 "Deep Learning-Based Stock Fluctuation Prediction According to Overseas Indices and **Trading Trend by Investors**" (원고는 "trading trends" 로 끝남) | 일치. 해외지수(S&P500 등)와 투자자별 매매 동향을 입력으로 스택 오토인코더+LSTM 으로 등락 예측 | **서지 수정** |
| [3] Vaswani et al., NIPS 2017, pp.5998-6008 | papers.nips.cc/paper/7181 / ACM DL 10.5555/3295222.3295349 | 일치 | 일치 (모든 시점 어텐션 → 길이 제곱 비용) | 유지 |
| [4] Zhou et al., AAAI 2021, pp.11106-11115 | ojs.aaai.org (submissionId 17325) / github zhouhaoyi/Informer2020 | 일치. 정식 서지는 Vol.35, No.12 포함 | 일치 (ProbSparse 희소 어텐션) | 유지 (권호 보완은 선택) |
| [5] Wu et al., NeurIPS 2021, pp.22419-22430 | proceedings.neurips.cc (hash bcc0d400…) / dblp conf/nips/WuXWL21 | 일치 | 일치 (계열 분해) | 유지 |
| [6] Zeng et al., AAAI 2023, pp.11121-11128 | DOI 10.1609/aaai.v37i9.26317 | 일치. 정식 서지는 Vol.37, No.9 | 일치 (선형 모형이 트랜스포머를 상회) | 유지 (권호 보완은 선택) |
| [7] Nie et al., ICLR 2023 | mlanthology / openreview, arXiv:2211.14730 | 일치 | 일치 (패치 토큰화 + 채널 독립) | 유지 |
| [8] Lim et al., IJF 37(4), 1748-1764, 2021 | EconPapers RePEc:eee:intfor:v:37:y:2021:i:4:p:1748-1764 / research.google | 일치 | 일치 (변수선택망·GRN·정적 공변량 인코더·분위 출력) | 유지 |
| [9] Kim, Kim & Lee, KIPS TSDE 11(2), 81-86, 2022 | **1차 자료 직접 확인** — `docs/kips/samples/dl01-v11n2-TFT-multi-horizon-forecasting.pdf`, DOI 10.3745/KTSDE.2022.11.2.81 | 일치 (InKyung Kim, DaeHee Kim, Jaekoo Lee) | 일치 (TFT 를 주가·미세먼지·전력 수요의 다층 수평 예측에 적용) | 유지 |
| [10] Kim & Kim, KIPS TSDE 10(12), 579-586, 2021 | KCI ART002797299 / KISTI JAKO202106153181642 | 일치 (Kim Dong-Keon, Kim Kwangsu, 성균관대) | 느슨함. 본문은 "트랜스포머를 시계열 특성에 맞추어 변형"이라 했는데, 실제 기여는 예측 구간 **전체를 한 번에 생성하는 스타일 기반 디코더** 다 | 유지 (문장 구체화 권고) |
| [11] Kim et al., ICLR 2022 | iclr.cc/virtual/2022/poster/6034 / KAIST KOASAS | 일치 | 일치 (윈도우별 인스턴스 정규화 후 역변환) | 유지 |
| [12] Koenker & Bassett, Econometrica 46(1), 33-50, 1978 | econometricsociety.org / RePEc ecm:emetrp:v:46:y:1978:i:1 / JSTOR 1913643 | 일치 | 일치 | 유지 |
| [13] Romano et al., NeurIPS 2019, pp.3543-3553 | papers.nips.cc/paper/8613 | 일치 | 일치 (등각예측 + 분위회귀 결합) | 유지 |
| [14] Chow, IEEE TIT 16(1), 41-46, 1970 | DOI 10.1109/TIT.1970.1054406 / dblp journals/tit/Chow70 | 일치 | 일치 (오류·기각 교환관계로 최적 기각 규칙) | 유지 |
| [15] El-Yaniv & Wiener, JMLR 11, 1605-1641, 2010 | jmlr.org/papers/volume11/el-yaniv10a | 일치 | 일치 (risk–coverage 교환관계) | 유지 |
| [16] Geifman & El-Yaniv, NeurIPS 2017, pp.4878-4887 | proceedings.neurips.cc/paper/2017 (hash 4a8423d5…) / dblp conf/nips/GeifmanE17 | 일치 | **불일치.** 본문은 "선택 함수를 **별도 헤드로 학습**하는 방법"이라 했으나, 이 논문은 **이미 학습된** 신경망에서 softmax response·MC-dropout 신뢰도로 선택 분류기를 구성한다. 별도 헤드 학습은 같은 저자의 SelectiveNet(ICML 2019)이다 | **인용 문장 수정** |
| [17] Fischer & Krauss, EJOR 270(2), 654-669, 2018 | RePEc eee:ejores:v:270:y:2018:i:2 / DOI 10.1016/j.ejor.2017.11.054 | 일치 | 일치 (S&P500 구성종목 방향 예측) | 유지 |
| [18] Gu, Kelly & Xiu, RFS 33(5), 2223-2273, 2020 | RePEc oup:rfinst:v33y2020i5 / NBER w25398 | 일치 | 일치 | 유지 |
| [19] Kim & Jung, KIPS TSDE 8(3), 123-128, 2019 | KCI ART002448478 / KISTI JAKO201912761597466 | 일치 (김현진·정연승, 단국대) | 일치 (가중 손실 함수 + RCNN) | 유지 |
| [20] Makridakis et al., PLOS ONE 13(3), e0194889, 2018 | DOI 10.1371/journal.pone.0194889 | 일치 | 일치 (단순 기준모형 비교 부재 문제 제기) | 유지 |
| [21] Wang et al., NeurIPS 2024 | neurips.cc/virtual/2024/poster/95770 / arXiv:2402.19072 / proceedings.neurips.cc | 일치 (저자 9인 → et al. 표기 규정에 맞음) | 일치 (외생 변수는 변수 단위 토큰, 내생은 패치 토큰) | 유지 |
| [22] Diebold & Mariano, JBES 13(3), 253-263, 1995 | RePEc bes:jnlbes:v:13:y:1995:i:3 | 일치 | 일치 | 유지 |
| [23] Newey & West, Econometrica 55(3), 703-708, 1987 | JSTOR 1913610 / DOI 10.2307/1913610 | 일치 | 일치 | 유지 |
| [24] Harvey, Leybourne & Newbold, IJF 13(2), 281-291, 1997 | RePEc eee:intfor:v:13:y:1997:i:2 | 일치 | 일치 (소표본 보정) | 유지 |
| [25] Ke et al., NIPS 2017, pp.3146-3154 | neurips.cc/virtual/2017/poster/9099 / ACM DL 10.5555/3294996.3295074 | 쪽수 표기가 출처마다 3146-3154 / 3149-3157 로 갈린다. NeurIPS 공식 쪽은 3146-3154 | 일치 (scikit-learn 의 히스토그램 GBDT 가 LightGBM 방식을 따름) | 유지 |
| [26] Pedregosa et al., JMLR 12, 2825-2830, 2011 | jmlr.org/papers/volume12/pedregosa11a | 일치 | 일치 | 유지 |

## 고쳐야 할 문장 (원문 → 권고안)

**[1] 1장 2문단**

> 금융 시계열에 딥러닝을 적용한 연구는 장단기기억 신경망과 합성곱 구조를 거쳐 트랜스포머 계열로 확장되며 폭넓게 축적되어 왔다[1].

[1]은 2005–2019년 연구를 정리한 리뷰로 CNN·DBN·LSTM 을 다루며 트랜스포머를 다루지 않는다.
→ 트랜스포머 확장 부분을 빼고 [1]의 범위 안에서 서술하거나, 확장 부분은 [4]–[7]로 받친다.

**[16] 2.4절**

> 심층 신경망에서는 선택 함수를 별도 헤드로 학습하는 방법이 제안되었다[16].

[16]은 이미 학습된 신경망의 신뢰도 점수로 선택 분류기를 구성하는 방법이다.
→ "이미 학습된 신경망의 신뢰도 점수로 기각 영역을 정하는 방법이 제안되었다[16]" 로 고치거나,
별도 헤드 학습을 쓰려면 SelectiveNet(Geifman & El-Yaniv, ICML 2019)을 새로 검증해 추가한다.

**[10] 2.2절 (선택)**

> 트랜스포머를 시계열 특성에 맞추어 변형한 연구[10]

→ "예측 구간 전체를 한 번에 생성하는 디코더를 제안한 연구[10]" 로 구체화한다.

## 서지 수정

**[2]** 영문 제목 끝을 정식 표기로 바꾼다.

> T. S. Kim and S. Lee, "Deep learning-based stock fluctuation prediction according to overseas indices and **trading trend by investors**," *KIPS Transactions on Software and Data Engineering*, Vol.10, No.9, pp.367-374, 2021.
