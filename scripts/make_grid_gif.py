"""한미약품 10거래일 · '간격 재조정 + 기준가 재설정' 트랙 재생을 GIF 로 굽는다.

아티팩트(R6LLWx2zAcWx5EXbZ7tj8J)의 bothViews 데이터와 draw() 로직·패널 틀을 그대로 옮긴 것.
다크 테마 토큰 사용. 1배속 = 프레임마다 봉 1개 ≈ 60봉/초.
치수는 아티팩트 CSS px 에 SC 를 곱해 그린다 — SC=2 면 레티나 해상도.
"""
import json
import sys

from PIL import Image, ImageDraw, ImageFont

SC = 2                                      # 해상도 배율 (1 = CSS px 그대로)
sc = lambda v: v * SC

SP = "/private/tmp/claude-501/-Users-mac-Desktop-Capstone-Stock-Price-Prediction/68c52e55-632f-4cea-b0da-20c667eabc59/scratchpad"
D = json.load(open(f"{SP}/data.json"))
S = [x for x in D["stocks"] if x["name"] == "한미약품"][0]
V = S["bothViews"][2]                      # days=10
bars, frm, fills = S["bars"], V["from"], V["fills"]
segs_all = V["segs"]

# ── 아티팩트 패널 틀 ──────────────────────────────────────────
IMG_W, IMG_H = sc(1240), sc(605)
MG = sc(12)                                 # 바깥 여백
HD = sc(44)                                 # .panel-hd 높이
PX, PT = sc(14), sc(12)                     # .chart-wrap 패딩
CV_X, CV_Y = MG + PX, MG + HD + sc(1) + PT
CV_W, CV_H = IMG_W - 2 * MG - 2 * PX, sc(520)       # canvas { height: 520px }
padL, padR, padT, padB = sc(6), sc(62), sc(12), sc(24)   # draw() 내부 패딩
plotW, plotH = CV_W - padL - padR, CV_H - padT - padB

# ── 다크 테마 토큰. rung-soft 는 alpha 0.34 를 surface 위에 미리 섞은 값 ──
GROUND, SURF, LINE = "#16150f", "#201e19", "#3b372e"
INK, INK2, INK3 = "#f4f0e7", "#b2ab9c", "#857e70"
BUY, SELL, RUNG, RUNGSOFT = "#e8635c", "#4b93e8", "#9c8757", "#4a422e"

f = lambda p, s: ImageFont.truetype(p, sc(s))
GO, MO = "/System/Library/Fonts/AppleSDGothicNeo.ttc", "/System/Library/Fonts/Menlo.ttc"
F_H2, F_TAB, F_LEG, F_MONO, F_RESET = f(GO, 13), f(GO, 12), f(GO, 12), f(MO, 10), f(GO, 10)
F_TALLY = f(GO, 15)

lo = min(b[3] for b in bars[frm:])
hi = max(b[2] for b in bars[frm:])
for g in segs_all:
    lo = min(lo, *g["buyLevels"])
    hi = max(hi, *g["sellLevels"])
pad = (hi - lo) * 0.04
lo -= pad
hi += pad
n = len(bars) - frm
X = lambda i: CV_X + padL + (i - frm + 0.5) * (plotW / n)
Y = lambda p: CV_Y + padT + (hi - p) / (hi - lo) * plotH
bw = plotW / n
TOP, BOT = CV_Y + padT, CV_Y + padT + plotH
RIGHT = CV_X + padL + plotW


def dash(d, p0, p1, fill, width=1, on=3, off=4):
    (x0, y0), (x1, y1) = p0, p1
    width, on, off = sc(width), sc(on), sc(off)
    L = max(abs(x1 - x0), abs(y1 - y0))
    t = 0.0
    while t < L:
        e = min(t + on, L)
        d.line([x0 + (x1 - x0) * t / L, y0 + (y1 - y0) * t / L,
                x0 + (x1 - x0) * e / L, y0 + (y1 - y0) * e / L], fill=fill, width=width)
        t = e + off


def header(d):
    y = MG + HD / 2
    d.text((MG + sc(14), y), "한미약품 · 128940", font=F_H2, fill=INK2, anchor="lm")

    # 기간 탭 — 10일 선택
    tabs, tw, th = ["1일", "5일", "10일"], sc(48), sc(13)
    x0 = IMG_W / 2 - tw * len(tabs) / 2
    d.rounded_rectangle([x0, y - th, x0 + tw * len(tabs), y + th], sc(7), outline=LINE, width=SC)
    for k, t in enumerate(tabs):
        on = t == "10일"
        bx = x0 + tw * k
        if on:
            d.rounded_rectangle([bx, y - th, bx + tw, y + th], sc(7), fill=INK)
        if k:
            d.line([bx, y - th, bx, y + th], fill=LINE, width=SC)
        d.text((bx + tw / 2, y), t, font=F_TAB, fill=GROUND if on else INK2, anchor="mm")

    # 범례 — 오른쪽 정렬
    items = [("dot", BUY, "매수 체결"), ("dot", SELL, "매도 체결"),
             ("bar", RUNG, "그리드 칸"), ("vdash", RUNG, "간격·기준가 변경")]
    w = sum(sc(16) + d.textlength(t, font=F_LEG) + sc(14) for _, _, t in items) - sc(14)
    x = IMG_W - MG - sc(14) - w
    for kind, col, t in items:
        if kind == "dot":
            d.ellipse([x, y - sc(4.5), x + sc(9), y + sc(4.5)], fill=col)
        elif kind == "bar":
            d.rectangle([x - sc(2), y - sc(1), x + sc(12), y + sc(1)], fill=col)
        else:
            dash(d, (x + sc(4), y - sc(5.5)), (x + sc(4), y + sc(5.5)), col, width=2, on=2, off=2)
        x += sc(16)
        d.text((x, y), t, font=F_LEG, fill=INK2, anchor="lm")
        x += d.textlength(t, font=F_LEG) + sc(14)


def frame(upto):
    im = Image.new("RGB", (IMG_W, IMG_H), GROUND)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([MG, MG, IMG_W - MG, IMG_H - MG], sc(10), fill=SURF, outline=LINE, width=SC)
    d.line([MG, MG + HD, IMG_W - MG, MG + HD], fill=LINE, width=SC)
    header(d)

    segs = [g for g in segs_all if g["i"] <= upto]
    edge = max(CV_X + padL, min(RIGHT, X(min(max(upto, frm), len(bars) - 1)) + bw / 2))
    drawn_y = []

    for k, g in enumerate(segs):
        last = k == len(segs) - 1
        x0 = CV_X + padL if k == 0 else X(g["i"]) + bw / 2
        x1 = edge if last else X(segs[k + 1]["i"]) + bw / 2
        rungs = [(p, "buy") for p in g["buyLevels"]] + \
                [(p, "sell") for p in g["sellLevels"]] + [(g["center"], "center")]
        for p, kind in rungs:
            y = Y(p)
            if y < TOP - sc(2) or y > BOT + sc(2):
                continue
            if kind == "center":
                d.line([x0, y, x1, y], fill=RUNG, width=sc(2))
            else:
                dash(d, (x0, y), (x1, y), RUNGSOFT)
            # 오른쪽 가격 라벨 — 마지막 구간만, 겹치면 건너뛴다
            if last and x1 >= RIGHT - 0.5 and all(abs(y - q) > sc(11) for q in drawn_y):
                drawn_y.append(y)
                d.text((RIGHT + sc(7), y), f"{p:,}", font=F_MONO,
                       fill=INK if kind == "center" else INK3, anchor="lm")
        if k > 0:
            prev = segs[k - 1]
            moved = g["center"] != prev["center"]
            yA = Y(prev["center"]) if moved else Y(g["sellLevels"][-1])
            yB = Y(g["center"]) if moved else Y(g["buyLevels"][-1])
            dash(d, (x0, yA), (x0, yB), RUNG, width=2, on=4, off=3)
            if moved:
                up = yB < yA
                d.text((x0 + sc(4), yB + (-sc(3) if up else sc(3))),
                       "재설정 " + ("▲" if g["center"] > prev["center"] else "▼"),
                       font=F_RESET, fill=RUNG, anchor="ls" if up else "la")

    # 캔들 — 한국식: 상승 빨강 / 하락 파랑
    body = max(sc(1.4), bw * 0.62)
    for i in range(frm, min(upto, len(bars) - 1) + 1):
        _, o, hh, ll, c = bars[i]
        x = X(i)
        col = BUY if c >= o else SELL
        d.line([x, Y(hh), x, Y(ll)], fill=col, width=SC)
        y1, y2 = Y(max(o, c)), Y(min(o, c))
        d.rectangle([x - body / 2, y1, x + body / 2, max(y1 + SC, y2)], fill=col)

    # 체결 — 표면색 링으로 겹쳐도 구분되게
    for fl in fills:
        if fl["i"] > upto:
            continue
        x, y = X(fl["i"]), Y(fl["px"])
        r = sc(6.5) if fl["kind"] == "init" else sc(5)
        d.ellipse([x - r, y - r, x + r, y + r],
                  fill=BUY if fl["side"] == "buy" else SELL, outline=SURF, width=sc(2))

    if upto < len(bars) - 1:
        dash(d, (X(upto), TOP), (X(upto), BOT), INK2, on=2, off=3)

    # 체결 누계 — 우측 하단. 아티팩트 .tally 와 같은 색
    nb = sum(1 for x in fills if x["i"] <= upto and x["side"] == "buy")
    ns = sum(1 for x in fills if x["i"] <= upto and x["side"] == "sell")
    tx, ty = RIGHT - sc(10), BOT - sc(10)
    for txt, col in ((f"매도 {ns}", SELL), (f"매수 {nb}", BUY)):
        d.text((tx, ty), txt, font=F_TALLY, fill=col, anchor="rs")
        tx -= d.textlength(txt, font=F_TALLY) + sc(14)

    lastday = ""
    for i in range(frm, len(bars)):
        day = bars[i][0][5:10]
        if day == lastday:
            continue
        lastday = day
        if frm + 2 < i < len(bars) - 6:
            d.text((X(i), BOT + sc(6)), day, font=F_MONO, fill=INK3, anchor="ma")
            d.line([X(i) - bw / 2, TOP, X(i) - bw / 2, BOT], fill=RUNGSOFT, width=SC)
    return im


out = sys.argv[1]
if out.endswith(".png"):          # .png 로 부르면 마지막 프레임 한 장만
    frame(len(bars) - 1).save(out)
    print(out, (IMG_W, IMG_H))
    raise SystemExit

# 아티팩트 1배속 = requestAnimationFrame 마다 봉 1개 ≈ 60봉/초. 여기선 1.5배속 = 90봉/초.
# GIF 는 20ms 미만 지연을 100ms 로 올려버리는 뷰어가 있어 지연은 50ms 로 두고 봉 수로 속도를 맞춘다
# (프레임당 4.5봉 — 정수가 아니라 소수점을 누적해서 4,5,4,5… 로 나눠 넣는다).
SPEED, DELAY = 1.5, 50
step = 60 * SPEED * DELAY / 1000
idx, t = [], float(frm)
while t < len(bars):
    idx.append(int(t))
    t += step
idx += [len(bars) - 1] * 30
frames = [frame(i) for i in idx]
# 색이 열몇 개뿐이라 팔레트를 256 다 쓸 필요가 없다 — 첫 프레임 팔레트로 전부 양자화한다
pal = frames[0].quantize(colors=64, method=Image.MEDIANCUT)
frames = [im.quantize(palette=pal, dither=Image.NONE) for im in frames]
frames[0].save(out, save_all=True, append_images=frames[1:], duration=DELAY, loop=0,
               optimize=True, disposal=2)
print(out, len(frames), "frames", frames[0].size)
