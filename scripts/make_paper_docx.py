"""docs/PAPER_TKIPS.md -> docs/PAPER_TKIPS_*.docx (정보처리학회 논문지 TKIPS).

조판은 학회 양식 `docs/kips/paper-format_TKIPS.hwp` 그대로다. hwp 의 PAGE_DEF 를 직접
읽어 맞췄다: A4 / 여백 L20 R20 T25.5 B15.5mm / 머리·꼬리 10mm / 2단 / 단 폭 81mm /
단 간격 8mm (본문 폭 170 = 81+8+81). 국문 표제부도 양식대로 1쪽에 둔다.
hwp 는 머리말·꼬리말을 여백과 본문 사이의 별도 영역으로 두므로 본문 상자는
좌우 20mm / 위 35.5mm / 아래 25.5mm 안쪽이다 — Word 여백은 본문 기준이라 그 값을 넣는다.

    review.docx   투고용 — 규정 2.1 에 따라 저자 정보를 뺀다
    preview.docx  저자 정보까지 넣은 판 — 쪽수 확인과 게재 확정 후 최종본 대비용

    python3 scripts/make_paper_docx.py
"""
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Mm, Pt
from docx.oxml import OxmlElement

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs" / "PAPER_TKIPS.md"
REVIEW_OUT = ROOT / "docs" / "PAPER_TKIPS_review.docx"     # 투고용
PREVIEW_OUT = ROOT / "docs" / "PAPER_TKIPS_preview.docx"   # 게재본 조판 확인용(쪽수)
AUTHOR_BLOCK = re.compile(r"<!-- AUTHORS:BEGIN -->.*?<!-- AUTHORS:END -->", re.S)

# 양식에 표기된 글꼴·크기 (견명조 14pt 따위). 한글이 HNC 글꼴로 렌더한다.
# 뒤의 두 값은 양식 hwp 의 CHAR_SHAPE 에서 읽은 장평(%)·자간(글자 크기 대비 %)이다.
# 이걸 빼면 Word 기본값(100%/0%)으로 조판되어 게재논문보다 눈에 띄게 벌어진다.
F = {
    "title_en": ("견명조", 14, 93, -5),
    "author": ("중고딕", 10, 94, 0),
    "abs_head": ("태고딕", 10.5, 100, 0),
    "abs_body": ("신명조", 7.5, 95, -5),
    "abs_body_ko": ("신명조", 7.5, 98, -5),
    "kw": ("태고딕", 7.5, 97, 0),
    "title_ko": ("태명조", 16.5, 96, -5),
    "chapter": ("견고딕", 10, 97, 0),
    "section": ("중고딕", 9, 95, -5),
    "body": ("신명조", 9, 100, -3),
    "caption": ("중고딕", 8.5, 95, -5),
    "table": ("신명조", 8, 95, -3),
    "refs": ("신명조", 8.5, 95, -3),
    "footnote": ("신명조", 7, 95, -5),
    "bio_name": ("중고딕", 11, 100, 10),
    "bio": ("신명조", 8.5, 95, -3),
}
# 양식이 지정한 HNC 글꼴은 한글(HWP)에만 있어서 Word 는 전부 같은 글꼴로 대체해 버린다.
# 그 과정에서 명조/고딕 구분과 굵기가 사라져 장·절 제목이 본문과 구분되지 않는다
# (게재본과 나란히 놓고 확인). docx 로 내는 동안은 맥에 있는 같은 계열로 바꿔 둔다.
# hwp 최종본에서는 양식 글꼴 이름(F 의 키)으로 되돌릴 것.
HNC = {
    "신명조": "AppleMyungjo", "태명조": "AppleMyungjo", "견명조": "AppleMyungjo",
    "중고딕": "Apple SD Gothic Neo", "태고딕": "Apple SD Gothic Neo",
    "견고딕": "Apple SD Gothic Neo",
}

# 고정 줄간격(pt). 양식의 줄간격은 글자 크기 대비 %(본문 150, 참고문헌 154, 캡션·표 130,
# 각주 110)인데 HNC 글꼴의 줄 높이가 끼어 실제 인쇄본은 그보다 조금 넓다. 게재논문에서
# 잰 값(본문 14.0 / 초록 11.2 / 각주 8.4 / 저자소개 13.7)을 그대로 고정값으로 쓴다.
# 배수로 두면 Word 가 대체 글꼴의 줄 높이로 계산해 들쭉날쭉해진다.
LEADING = {
    "body": 14.0, "abs_body": 10.5, "abs_body_ko": 12.0, "kw": 11.2,
    "refs": 13.5, "caption": 11.0, "table": 10.4, "footnote": 8.4, "bio": 13.6,
    # 제목 계열도 고정값으로 둔다 — 배수로 두면 대체 글꼴의 줄 높이가 그대로 반영돼
    # 제목과 본문 사이가 들뜬다. 값은 게재본의 제목 줄 간격 실측.
    "title_en": 18.2, "title_ko": 20.0, "author": 13.0, "abs_head": 14.0,
    "chapter": 15.5, "section": 14.0, "bio_name": 15.0,
}
COL_W = Mm(81)      # 단 폭
BIO_W = Mm(25)      # 저자 소개 사진
FULL_W = Mm(170)    # 좌우 여백을 뺀 본문 폭 — 전폭 표가 쓴다
# 전폭 그림은 본문 폭보다 좁게 둔다. 170mm 로 넣으면 높이가 128.7mm 로 쪽 높이(236mm)의
# 55% 라, 2단 본문 뒤에 남은 자리에 못 들어가 다음 쪽으로 밀리고 그 쪽이 통째로 빈다.
FIG_W = Mm(105)     # 높이 79.5mm

# 표제부의 문단 간격·여백은 양식 hwp 의 PARA_SHAPE 값(1/200pt 단위)을 pt 로 환산한 것이다.
# 게재논문 실측 좌표와도 맞는다 — 저자줄 아래 22.7pt, 키워드 위 8.5pt, 초록 들여쓰기 10pt.
ABS_SIDE = Pt(17.5)     # 초록·키워드 좌우 여백
ABS_INDENT = Pt(10)     # 초록 첫 줄 들여쓰기


def page_setup(section, cols=1):
    section.page_width, section.page_height = Mm(210), Mm(297)
    section.left_margin = section.right_margin = Mm(20)
    # 위 25.5 + 머리말 10 / 아래 15.5 + 꼬리말 10 = 본문 상자 35.5~271.5mm.
    # 게재본 실측도 같다 (본문 첫 줄 38.2mm, 끝 줄 269.6mm, 머리말 27.8mm).
    section.top_margin, section.bottom_margin = Mm(35.5), Mm(25.5)
    section.header_distance, section.footer_distance = Mm(25.5), Mm(15.5)
    sectPr = section._sectPr
    for old in sectPr.findall(qn("w:cols")):
        sectPr.remove(old)
    el = OxmlElement("w:cols")
    el.set(qn("w:num"), str(cols))
    el.set(qn("w:space"), str(int(8 * 72 * 20 / 25.4)))  # 8mm -> twips
    el.set(qn("w:equalWidth"), "1")
    sectPr.append(el)


def korean_spacing(p):
    """국문 조판 보정 — autoSpaceDE/DN 을 끄면 '2015년'이 '2015 년'으로 벌어지지 않는다.

    wordWrap 0 (글자 단위 줄바꿈 허용) 은 쓰지 않는다. 양쪽 정렬의 남는 폭을 줄여주긴
    하지만 영문 단어와 숫자가 아무 데서나 잘린다 — 참고문헌이 'Proceed/ings',
    '202/3' 으로 끊겼다. 게재본은 단어 단위로 줄을 바꾼다.
    """
    pPr = p._p.get_or_add_pPr()
    for tag, val in (("w:autoSpaceDN", "0"), ("w:autoSpaceDE", "0")):
        el = OxmlElement(tag)
        el.set(qn("w:val"), val)
        pPr.insert(0, el)      # 스키마 순서: wordWrap → autoSpaceDE → autoSpaceDN → spacing/ind/jc


def footnote_frame(p, first):
    """저자·접수일 각주를 1쪽 좌측 단 하단에 붙인다.

    양식은 이것을 각주로 두어 본문과 분리하는데, docx 에서 그대로 흘리면 키워드 바로 뒤
    본문 첫머리에 끼어든다. 같은 자리에 놓이도록 단 폭 프레임으로 띄우고, 첫 줄 위에
    각주 구분선을 넣는다.
    """
    pPr = p._p.get_or_add_pPr()
    if first:
        bdr = OxmlElement("w:pBdr")
        top = OxmlElement("w:top")
        for k, v in (("w:val", "single"), ("w:sz", "6"),
                     ("w:space", "4"), ("w:color", "000000")):
            top.set(qn(k), v)
        bdr.append(top)
        pPr.insert(0, bdr)
    frame = OxmlElement("w:framePr")
    # vSpace 는 위쪽 본문과의 거리. 게재본은 본문 마지막 줄 기준선에서 각주 첫 줄까지
    # 24.3pt 이고, 그 중 본문 줄 높이 14pt 를 뺀 나머지가 이 여백 + 구분선 간격이다.
    for k, v in (("w:w", str(COL_W.twips)), ("w:hRule", "auto"),
                 ("w:wrap", "around"), ("w:vSpace", "160"),
                 ("w:vAnchor", "margin"),
                 ("w:hAnchor", "margin"), ("w:xAlign", "left"),
                 ("w:yAlign", "bottom")):
        frame.set(qn(k), v)
    pPr.insert(0, frame)        # 스키마 순서: framePr 가 pBdr·wordWrap 보다 앞


# rPr 자식 요소의 스키마 순서 — w:spacing(자간) 과 w:w(장평) 뒤에 오는 것들.
# 순서가 틀리면 Word 가 파일을 열면서 복구 대화상자를 띄운다.
RPR_AFTER = ("w:kern", "w:position", "w:sz", "w:szCs", "w:highlight", "w:u",
             "w:effect", "w:bdr", "w:shd", "w:fitText", "w:vertAlign", "w:rtl",
             "w:cs", "w:em", "w:lang", "w:eastAsianLayout", "w:specVanish", "w:oMath")


def set_rpr(rPr, tag, val):
    el = OxmlElement(tag)
    el.set(qn("w:val"), str(val))
    rPr.insert_element_before(el, *(("w:w",) + RPR_AFTER if tag == "w:spacing"
                                    else RPR_AFTER))


def style_run(run, key, bold=False, italic=False):
    name, size, scale, track = F[key]
    run.font.size = Pt(size)
    run.font.name = "Times New Roman"      # 라틴 글자
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:ascii"), "Times New Roman")
    rFonts.set(qn("w:hAnsi"), "Times New Roman")
    rFonts.set(qn("w:eastAsia"), HNC.get(name, name))   # 한글 글자
    set_rpr(rPr, "w:w", scale)                          # 장평
    set_rpr(rPr, "w:spacing", round(size * track / 5))  # 자간: % → 1/20pt
    run.bold = bold
    run.italic = italic


INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|_\{[^}]+\}|\^\{[^}]+\})")


def add_para(doc, text, key, align=None, space_after=3, indent=None, bold=False):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(0)
    leading = LEADING.get(key)
    pf.line_spacing = Pt(leading) if leading else 1.25
    if align is not None:
        p.alignment = align
    if indent is not None:
        pf.first_line_indent = indent
    korean_spacing(p)
    for piece in INLINE.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**"):
            style_run(p.add_run(piece[2:-2]), key, bold=True)
        elif piece.startswith("*") and piece.endswith("*") and len(piece) > 2:
            style_run(p.add_run(piece[1:-1]), key, italic=True)
        elif piece.startswith("_{") and piece.endswith("}"):
            r = p.add_run(piece[2:-1]); style_run(r, key, italic=True)
            r.font.subscript = True
        elif piece.startswith("^{") and piece.endswith("}"):
            r = p.add_run(piece[2:-1]); style_run(r, key, italic=True)
            r.font.superscript = True
        else:
            style_run(p.add_run(piece), key, bold=bold)
    return p


def three_line_borders(table, header=True):
    """학술지 3선표: 표 위·아래 굵은 선, 머리행 아래 가는 선, 세로선 없음."""
    def edge(cell, side, sz):
        tcPr = cell._tc.get_or_add_tcPr()
        borders = tcPr.find(qn("w:tcBorders"))
        if borders is None:
            borders = OxmlElement("w:tcBorders")
            tcPr.append(borders)
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:val"), "single" if sz else "none")
        el.set(qn("w:sz"), str(sz or 0))
        el.set(qn("w:color"), "000000")
        borders.append(el)

    rows = table.rows
    for r_i, row in enumerate(rows):
        for cell in row.cells:
            edge(cell, "left", 0)
            edge(cell, "right", 0)
            edge(cell, "top", 12 if r_i == 0 else 0)
            if r_i == 0 and header:
                edge(cell, "bottom", 6)
            elif r_i == len(rows) - 1:
                edge(cell, "bottom", 12)
            else:
                edge(cell, "bottom", 0)


def column_widths(rows, width):
    """열마다 가장 긴 칸의 길이에 비례해 폭을 나눈다.

    균등 분할이면 〈Table 7〉 처럼 "0.017837 ± 0.000078" 이 든 열이 줄바꿈된다.
    상한을 두는 이유는 변형 이름 같은 긴 라벨이 숫자 열의 폭까지 가져가지 않게 하려는 것이다.
    """
    w = [min(max(max(len(r[c]) for r in rows), 6), 28) for c in range(len(rows[0]))]
    total = sum(w)
    width = width - Mm(1.5)     # 단 폭에 딱 맞추면 테두리 때문에 밀린다
    return [int(width * x / total) for x in w]


def table_layout(table, widths):
    """tblW·tblGrid·여백을 명시한다.

    python-docx 가 표를 만들 때 넣는 tblGrid 는 셀 폭(tcW)과 무관한 값이라, 고정 레이아웃에서
    Word 가 grid 를 따르면 표가 단 밖으로 삐져나간다. 둘을 같은 값으로 맞춘다.
    기본 셀 여백(좌우 각 1.9mm)도 폭 계산에 들어가도록 줄인다.
    """
    tblPr = table._tbl.tblPr
    for tag, attrs in (("w:tblW", {"w:w": str(sum(widths) // 635), "w:type": "dxa"}),
                       ("w:tblLayout", {"w:type": "fixed"})):
        for old in tblPr.findall(qn(tag)):
            tblPr.remove(old)
        el = OxmlElement(tag)
        for k, v in attrs.items():
            el.set(qn(k), v)
        tblPr.append(el)
    mar = OxmlElement("w:tblCellMar")
    for side, val in (("left", 40), ("right", 40)):      # 40 twip = 0.7mm
        e = OxmlElement(f"w:{side}")
        e.set(qn("w:w"), str(val))
        e.set(qn("w:type"), "dxa")
        mar.append(e)
    tblPr.append(mar)
    grid = table._tbl.find(qn("w:tblGrid"))
    for col in list(grid):
        grid.remove(col)
    for w in widths:
        gc = OxmlElement("w:gridCol")
        gc.set(qn("w:w"), str(w // 635))                 # EMU -> twip
        grid.append(gc)


def keep_together(table):
    """행이 쪽 사이에서 갈라지지 않게 하고, 표 전체를 한 쪽에 묶는다."""
    for row in table.rows[:-1]:
        for para in (p for c in row.cells for p in c.paragraphs):
            para.paragraph_format.keep_with_next = True
    for row in table.rows:
        trPr = row._tr.get_or_add_trPr()
        trPr.append(OxmlElement("w:cantSplit"))


def add_table(doc, rows, width):
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    widths = column_widths(rows, width)
    for r_i, row in enumerate(rows):
        for c_i, text in enumerate(row):
            cell = table.cell(r_i, c_i)
            cell.width = widths[c_i]
            cell.text = ""
            p = cell.paragraphs[0]
            korean_spacing(p)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.line_spacing = Pt(LEADING["table"])
            bold = r_i == 0 or text.startswith("**")
            clean = text.replace("**", "")
            for piece in INLINE.split(clean):
                if not piece:
                    continue
                if piece.startswith("*") and piece.endswith("*") and len(piece) > 2:
                    style_run(p.add_run(piece[1:-1]), "table", bold=bold, italic=True)
                else:
                    style_run(p.add_run(piece), "table", bold=bold)
    three_line_borders(table)
    table_layout(table, widths)
    keep_together(table)
    return table


def clear_metadata(doc):
    """문서 속성을 비운다 — 기본값이 작성자 'python-docx', 메모 'generated by python-docx' 라
    심사본에 그대로 실린다. 익명 심사라 저자 단서를 남기지 않는 편이 낫다."""
    cp = doc.core_properties
    cp.author = cp.last_modified_by = cp.comments = ""
    cp.title = cp.subject = cp.category = cp.keywords = ""


def switch(doc, cols):
    """단 수를 바꾸고 적용된 단 수를 돌려준다. 전폭 표·그림만 1단 구역으로 빠진다."""
    page_setup(doc.add_section(WD_SECTION.CONTINUOUS), cols)
    return cols


def parse(md):
    """마크다운을 블록 리스트로. (kind, payload)"""
    blocks, lines, i = [], md.split("\n"), 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("<!--"):
            while i < len(lines) and "-->" not in lines[i]:
                i += 1
            i += 1
            continue
        if not ln.strip() or ln.strip() == "---":
            i += 1
            continue
        if ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            blocks.append(("table", rows))
            continue
        m = re.match(r"!\[[^\]]*\]\(([^)]+)\)", ln.strip())
        if m:
            blocks.append(("image", m.group(1)))
            i += 1
            continue
        if ln.startswith("#"):
            level = len(ln) - len(ln.lstrip("#"))
            blocks.append((f"h{level}", ln.lstrip("#").strip()))
            i += 1
            continue
        buf = []
        while (i < len(lines) and lines[i].strip()
               and not lines[i].startswith(("|", "#", "!", "<!--"))):
            buf.append(lines[i].strip())
            i += 1
        blocks.append(("p", " ".join(buf)))
    return blocks


def group_bios(blocks):
    """References 뒤의 저자 소개를 저자별 한 덩어리로 묶는다.

    게재본 조판이 좌측 사진 / 우측 이름·경력·관심분야의 2단 표라서,
    h4 와 뒤따르는 사진·본문을 한 블록으로 합쳐야 표로 낼 수 있다.
    """
    out, bio, cur = [], False, None
    for kind, payload in blocks:
        if kind == "h2":
            if cur:
                out.append(("bio", cur)); cur = None
            bio = payload == "References"
            out.append((kind, payload))
            continue
        if bio and kind == "h4":
            if cur:
                out.append(("bio", cur))
            cur = {"name": payload, "img": None, "lines": []}
            continue
        if cur is not None:
            if kind == "image":
                cur["img"] = payload
            elif kind == "p":
                cur["lines"].append(payload)
            continue
        out.append((kind, payload))
    if cur:
        out.append(("bio", cur))
    return out


def build(blind):
    md = SRC.read_text(encoding="utf-8")
    if blind:           # 심사용 초기 투고 원고에는 저자 정보를 넣지 않는다 (규정 2.1)
        md = AUTHOR_BLOCK.sub("", md)
    blocks = group_bios(parse(md))
    full_w = FULL_W
    doc = Document()
    page_setup(doc.sections[0], cols=1)
    mode = 1          # 현재 단 수
    expect_author = False
    in_bio = False    # References 이후는 저자 소개 블록
    in_body = False   # 본문(2단) 시작 여부
    fn_started = False  # 1쪽 각주 블록의 첫 줄인가 (구분선은 한 번만)

    def next_is_wide_table(i):
        """다음 블록이 전폭이 필요한 표인가 — 캡션부터 1단으로 넘기려고 미리 본다."""
        nxt = blocks[i + 1] if i + 1 < len(blocks) else None
        return bool(nxt and nxt[0] == "table" and len(nxt[1][0]) >= 5)

    for bi, (kind, payload) in enumerate(blocks):
        was_author, expect_author = expect_author, False
        # --- 표제부 ---
        if kind == "h1":
            add_para(doc, payload, "title_en", WD_ALIGN_PARAGRAPH.CENTER, 19.8)
            expect_author = True
            continue
        if kind == "h2" and payload == "ABSTRACT":
            add_para(doc, "ABSTRACT", "abs_head", WD_ALIGN_PARAGRAPH.CENTER, 14.2)
            continue
        # 번호 없는 h2 중 영문 표제부가 아닌 것 = 국문 제목 (제목 문자열에 의존하지 않는다)
        if (kind == "h2" and not re.match(r"^\d+\.", payload)
                and payload not in ("ABSTRACT", "References", "Acknowledgments")):
            if mode != 1:
                switch(doc, 1)
                mode = 1
            in_body = False          # 이후 요약 단락은 7.5pt
            in_bio = False
            add_para(doc, payload, "title_ko", WD_ALIGN_PARAGRAPH.CENTER,
                     17.0).paragraph_format.space_before = Pt(10)
            expect_author = True
            continue
        if kind == "h3" and payload.startswith("요"):
            add_para(doc, "요     약", "abs_head", WD_ALIGN_PARAGRAPH.CENTER, 14.2)
            continue

        if kind == "p" and was_author:
            add_para(doc, payload, "author", WD_ALIGN_PARAGRAPH.CENTER, 22.7)
            continue
        if kind == "p" and payload.startswith(("**Keywords :**", "**키워드 :**")):
            pf = add_para(doc, payload.replace("**", ""), "kw",
                          None, 28.4).paragraph_format
            pf.space_before = Pt(8.5)
            pf.left_indent = pf.right_indent = ABS_SIDE
            continue

        # --- 장·절 제목: 첫 장에서 2단으로 전환 ---
        if kind == "h2":
            if not in_body:
                mode, in_body = switch(doc, 2), True
            if mode != 2:
                mode = switch(doc, 2)
            ch = add_para(doc, payload, "chapter", None, 14, bold=True)
            ch.paragraph_format.space_before = Pt(18)
            ch.paragraph_format.keep_with_next = True
            in_bio = payload == "References"
            continue
        if kind == "h3":
            if mode != 2:
                mode = switch(doc, 2)
            add_para(doc, payload, "section", None, 1, bold=True) \
                .paragraph_format.keep_with_next = True
            continue
        if kind == "h4":
            add_para(doc, payload, "bio_name" if in_bio else "body", None, 3)
            continue

        # --- 저자 소개: 좌 사진 / 우 이름·경력·관심분야 ---
        if kind == "bio":
            if mode != 2:
                mode = switch(doc, 2)
            table = doc.add_table(rows=1, cols=2)
            table.autofit = False
            table.alignment = WD_TABLE_ALIGNMENT.LEFT
            left, right = table.rows[0].cells
            left.width, right.width = Mm(28), Mm(53)
            for cell in (left, right):
                cell.vertical_alignment = WD_ALIGN_VERTICAL.TOP
                blank = cell.paragraphs[0]._p
                blank.getparent().remove(blank)
            if payload["img"]:
                ph = left.add_paragraph()
                ph.paragraph_format.space_after = Pt(0)
                ph.add_run().add_picture(str(ROOT / "docs" / payload["img"]),
                                         width=BIO_W)
            else:                       # 사진 미정인 저자는 자리만 남긴다
                add_para(left, "[사진]", "bio", None, 0)
            add_para(right, payload["name"], "bio_name",
                     WD_ALIGN_PARAGRAPH.CENTER, 2)
            for line in payload["lines"]:
                if line != "[사진]":
                    add_para(right, line, "bio", None, 1)
            doc.add_paragraph().paragraph_format.space_after = Pt(4)
            continue

        # --- 그림: 단 폭을 넘으므로 1단 구간으로 ---
        if kind == "image":
            full = "fig1" in payload and not in_bio   # 구조도만 전폭, 나머지는 단 폭
            if full and mode != 1:
                mode = switch(doc, 1)
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(2)
            p.add_run().add_picture(
                str(ROOT / "docs" / payload),
                width=BIO_W if in_bio else (
                    FIG_W if full else COL_W))
            continue

        # --- 표: 열이 많으면 1단 구간 ---
        if kind == "table":
            wide = len(payload[0]) >= 5
            if wide and mode != 1:
                mode = switch(doc, 1)
            add_table(doc, payload, full_w if mode == 1 else COL_W)
            doc.add_paragraph().paragraph_format.space_after = Pt(4)
            if wide and in_body:
                mode = switch(doc, 2)   # 전폭 표가 끝나면 곧바로 2단 본문으로 되돌린다
            continue

        # --- 본문 단락 ---
        text = payload
        if text.startswith(("Table ", "Fig. ")):
            if text.startswith("Table ") and next_is_wide_table(bi) and mode != 1:
                mode = switch(doc, 1)
            if text.startswith("Fig. ") and mode == 1:
                add_para(doc, text, "caption", WD_ALIGN_PARAGRAPH.LEFT, 8)
                mode = switch(doc, 2)   # 그림 캡션 다음은 본문으로 복귀
            else:
                cap = add_para(doc, text, "caption", None, 2)
                if text.startswith("Table "):
                    cap.paragraph_format.keep_with_next = True
            continue
        if text.startswith(("※", "†", "* Corresponding", "Manuscript Received",
                            "First Revision", "Accepted")):
            footnote_frame(add_para(doc, text, "footnote", None, 1),
                           first=not fn_started)
            fn_started = True
            continue
        if re.match(r"^\[\d+\]", text):
            ref = add_para(doc, text, "refs", None, 2, indent=Pt(-14))
            ref.paragraph_format.left_indent = Pt(14)   # 번호 내어쓰기
            continue
        if in_bio:
            add_para(doc, text, "bio", None, 1)
            continue
        if "&nbsp;" in text:        # 수식 행
            add_para(doc, text.replace("&nbsp;", " ").strip(), "body",
                     WD_ALIGN_PARAGRAPH.CENTER, 4)
            continue
        key = "body" if in_body else (
            "abs_body_ko" if re.search(r"[가-힣]", text) else "abs_body")
        # 국문 논문 관례: 본문 단락은 첫 줄을 한 글자 들여 쓴다
        para = add_para(doc, text, key, WD_ALIGN_PARAGRAPH.JUSTIFY,
                        4 if key == "body" else 0,
                        indent=Pt(F[key][1]) if key == "body" else ABS_INDENT)
        if key.startswith("abs_body"):   # 초록은 본문보다 좌우를 좁혀 조판한다
            para.paragraph_format.left_indent = ABS_SIDE
            para.paragraph_format.right_indent = ABS_SIDE

    dst = REVIEW_OUT if blind else PREVIEW_OUT
    clear_metadata(doc)
    doc.save(dst)
    print(f"wrote {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    build(blind=True)
    build(blind=False)
    sys.exit(0)
