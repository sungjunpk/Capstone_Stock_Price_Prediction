"""docs/PAPER_TKIPS.md -> docs/PAPER_TKIPS_*.docx (정보처리학회 논문지 TKIPS).

기본은 **투고 서식**이다 — TKIPS 투고 안내 5.2: 10pt, 2배 줄간격, A4, 상하좌우 여백 2.5cm.
5.4 의 순서에 따라 국문 제목·요약·키워드는 참고문헌 뒤로 옮긴다. 1단이다.
    review.docx  심사용(저자 정보 없음)    final.docx  저자 정보 포함

--preview 는 게재본 모양(2단)을 PAPER_TKIPS_preview.docx 로 낸다. 조판값은 옛 KTSDE
양식에서 실측했고 docs/kips/paper-format_TKIPS.hwp 와 같다:
A4 / 여백 L20 R20 T25.5 B15.5mm / 2단 / 단 간격 8mm / 단 폭 81mm.

    python3 scripts/make_paper_docx.py [--preview]
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
# 심사용은 규정 2.1 에 따라 저자 정보를 뺀다. 최종본은 넣는다.
OUT = {True: ROOT / "docs" / "PAPER_TKIPS_review.docx",
       False: ROOT / "docs" / "PAPER_TKIPS_final.docx"}
PREVIEW_OUT = ROOT / "docs" / "PAPER_TKIPS_preview.docx"
SUBMIT = True       # 투고 서식(1단·10pt·2배 줄간격). --preview 일 때만 False
AUTHOR_BLOCK = re.compile(r"<!-- AUTHORS:BEGIN -->.*?<!-- AUTHORS:END -->", re.S)

# 양식에 표기된 글꼴·크기 (견명조 14pt 따위). 한글이 HNC 글꼴로 렌더한다.
F = {
    "title_en": ("견명조", 14),
    "author": ("중고딕", 10),
    "abs_head": ("태고딕", 10.5),
    "abs_body": ("신명조", 7.5),
    "kw": ("태고딕", 7.5),
    "title_ko": ("태명조", 19),
    "chapter": ("견고딕", 10),
    "section": ("중고딕", 9),
    "body": ("신명조", 9),
    "caption": ("중고딕", 8.5),
    "table": ("신명조", 8),
    "refs": ("신명조", 8.5),
    "footnote": ("신명조", 7),
    "bio_name": ("중고딕", 11),
    "bio": ("신명조", 8.5),
}
LEADING = {                 # 샘플 게재논문 실측 기반 행간 배수
    "body": 1.42, "abs_body": 1.35, "refs": 1.25, "caption": 1.20,
    "table": 1.10, "footnote": 1.15,
}
COL_W = Mm(81)      # 단 폭
BIO_W = Mm(25)      # 저자 소개 사진
FULL_W = Mm(170)    # 좌우 여백을 뺀 본문 폭


def page_setup(section, cols=1):
    section.page_width, section.page_height = Mm(210), Mm(297)
    if SUBMIT:
        cols = 1
        section.left_margin = section.right_margin = Mm(25)
        section.top_margin = section.bottom_margin = Mm(25)
    else:
        section.left_margin = section.right_margin = Mm(20)
        section.top_margin, section.bottom_margin = Mm(25.5), Mm(15.5)
    section.header_distance = section.footer_distance = Mm(10)
    sectPr = section._sectPr
    for old in sectPr.findall(qn("w:cols")):
        sectPr.remove(old)
    el = OxmlElement("w:cols")
    el.set(qn("w:num"), str(cols))
    el.set(qn("w:space"), str(int(8 * 72 * 20 / 25.4)))  # 8mm -> twips
    el.set(qn("w:equalWidth"), "1")
    sectPr.append(el)


def korean_spacing(p):
    """국문 조판 보정.

    autoSpaceDE/DN 을 끄면 '2015년'이 '2015 년'으로 벌어지지 않는다.
    wordWrap 0 은 긴 영문 토큰이 줄 안에서 끊기게 하여, 양쪽 정렬이 남는 폭을
    한글 어절 사이로 몰아 벌리는 현상을 막는다.
    """
    pPr = p._p.get_or_add_pPr()
    for tag, val in (("w:autoSpaceDN", "0"), ("w:autoSpaceDE", "0"), ("w:wordWrap", "0")):
        el = OxmlElement(tag)
        el.set(qn("w:val"), val)
        pPr.insert(0, el)      # 스키마 순서: wordWrap → autoSpaceDE → autoSpaceDN → spacing/ind/jc


def style_run(run, key, bold=False, italic=False):
    name, size = F[key]
    run.font.size = Pt(max(size, 10) if SUBMIT else size)
    run.font.name = "Times New Roman"      # 라틴 글자
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:ascii"), "Times New Roman")
    rFonts.set(qn("w:hAnsi"), "Times New Roman")
    rFonts.set(qn("w:eastAsia"), name)     # 한글 글자는 양식 글꼴로
    run.bold = bold
    run.italic = italic


INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|_\{[^}]+\}|\^\{[^}]+\})")


def add_para(doc, text, key, align=None, space_after=3, indent=None, bold=False):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(0)
    pf.line_spacing = 2.0 if SUBMIT else LEADING.get(key, 1.25)
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


def add_table(doc, rows, width):
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    col_w = int(width / len(rows[0]))
    for r_i, row in enumerate(rows):
        for c_i, text in enumerate(row):
            cell = table.cell(r_i, c_i)
            cell.width = col_w
            cell.text = ""
            p = cell.paragraphs[0]
            korean_spacing(p)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.space_before = Pt(1)
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
    return table


def switch(doc, cols):
    s = doc.add_section(WD_SECTION.CONTINUOUS)
    page_setup(s, cols)


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


def korean_to_end(blocks):
    """국문 제목부터 국문 키워드까지를 참고문헌 뒤(저자 소개 앞)로 옮긴다 — 투고 안내 5.4."""
    start = next(i for i, (k, p) in enumerate(blocks) if k == "h2" and not re.match(r"^\d+\.", p)
                 and p not in ("ABSTRACT", "References", "Acknowledgments"))
    end = next(i for i, (k, p) in enumerate(blocks) if k == "p" and p.startswith("**키워드 :**"))
    ko, rest = blocks[start:end + 1], blocks[:start] + blocks[end + 1:]
    at = next((i for i, (k, _) in enumerate(rest) if k == "bio"), len(rest))
    return rest[:at] + ko + rest[at:]


def build(blind):
    md = SRC.read_text(encoding="utf-8")
    if blind:
        md = AUTHOR_BLOCK.sub("", md)
    blocks = group_bios(parse(md))
    if SUBMIT:
        blocks = korean_to_end(blocks)
    full_w = Mm(160) if SUBMIT else FULL_W
    doc = Document()
    page_setup(doc.sections[0], cols=1)
    mode = 1          # 현재 단 수
    expect_author = False
    in_bio = False    # References 이후는 저자 소개 블록
    in_body = False   # 본문(2단) 시작 여부

    def next_is_wide_table(i):
        """다음 블록이 전폭이 필요한 표인가 — 캡션부터 1단으로 넘기려고 미리 본다."""
        nxt = blocks[i + 1] if i + 1 < len(blocks) else None
        return bool(nxt and nxt[0] == "table" and len(nxt[1][0]) >= 5)

    for bi, (kind, payload) in enumerate(blocks):
        was_author, expect_author = expect_author, False
        # --- 표제부 ---
        if kind == "h1":
            add_para(doc, payload, "title_en", WD_ALIGN_PARAGRAPH.CENTER, 6)
            expect_author = True
            continue
        if kind == "h2" and payload == "ABSTRACT":
            add_para(doc, "ABSTRACT", "abs_head", WD_ALIGN_PARAGRAPH.CENTER, 4)
            continue
        # 번호 없는 h2 중 영문 표제부가 아닌 것 = 국문 제목 (제목 문자열에 의존하지 않는다)
        if (kind == "h2" and not re.match(r"^\d+\.", payload)
                and payload not in ("ABSTRACT", "References", "Acknowledgments")):
            if mode != 1:
                switch(doc, 1)
                mode = 1
            in_body = False          # 이후 요약 단락은 7.5pt
            if SUBMIT and in_bio:
                doc.add_page_break()     # 참고문헌 뒤 국문 표제부는 새 쪽에서
            in_bio = False
            add_para(doc, payload, "title_ko", WD_ALIGN_PARAGRAPH.CENTER, 6)
            expect_author = True
            continue
        if kind == "h3" and payload.startswith("요"):
            add_para(doc, "요     약", "abs_head", WD_ALIGN_PARAGRAPH.CENTER, 4)
            continue

        if kind == "p" and was_author:
            add_para(doc, payload, "author", WD_ALIGN_PARAGRAPH.CENTER, 8)
            continue
        if kind == "p" and payload.startswith(("**Keywords :**", "**키워드 :**")):
            add_para(doc, payload.replace("**", ""), "kw", None, 10)
            continue

        # --- 장·절 제목: 첫 장에서 2단으로 전환 ---
        if kind == "h2":
            if not in_body:
                switch(doc, 2)
                mode, in_body = 2, True
            if mode != 2:
                switch(doc, 2)
                mode = 2
            add_para(doc, payload, "chapter", None, 4, bold=True)
            in_bio = payload == "References"
            continue
        if kind == "h3":
            if mode != 2:
                switch(doc, 2)
                mode = 2
            add_para(doc, payload, "section", None, 3)
            continue
        if kind == "h4":
            add_para(doc, payload, "bio_name" if in_bio else "body", None, 3)
            continue

        # --- 저자 소개: 좌 사진 / 우 이름·경력·관심분야 ---
        if kind == "bio":
            if mode != 2:
                switch(doc, 2)
                mode = 2
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
                switch(doc, 1)
                mode = 1
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(2)
            p.add_run().add_picture(
                str(ROOT / "docs" / payload),
                width=BIO_W if in_bio else (full_w if full else COL_W))
            continue

        # --- 표: 열이 많으면 1단 구간 ---
        if kind == "table":
            wide = len(payload[0]) >= 5
            if wide and mode != 1:
                switch(doc, 1)
                mode = 1
            add_table(doc, payload, full_w if mode == 1 else COL_W)
            doc.add_paragraph().paragraph_format.space_after = Pt(4)
            if wide and in_body:
                switch(doc, 2)      # 전폭 표가 끝나면 곧바로 2단 본문으로 되돌린다
                mode = 2
            continue

        # --- 본문 단락 ---
        text = payload
        if text.startswith(("Table ", "Fig. ")):
            if text.startswith("Table ") and next_is_wide_table(bi) and mode != 1:
                switch(doc, 1)
                mode = 1
            if text.startswith("Fig. ") and mode == 1:
                add_para(doc, text, "caption", WD_ALIGN_PARAGRAPH.LEFT, 8)
                switch(doc, 2)      # 그림 캡션 다음은 본문으로 복귀
                mode = 2
            else:
                add_para(doc, text, "caption", None, 2)
            continue
        if text.startswith(("※", "†", "* Corresponding", "Manuscript Received",
                            "First Revision", "Accepted")):
            add_para(doc, text, "footnote", None, 1)
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
        key = "abs_body" if not in_body else "body"
        # 국문 논문 관례: 본문 단락은 첫 줄을 한 글자 들여 쓴다
        add_para(doc, text, key, WD_ALIGN_PARAGRAPH.JUSTIFY, 4,
                 indent=Pt(F[key][1]) if key == "body" else None)

    dst = OUT[blind] if SUBMIT else PREVIEW_OUT
    doc.save(dst)
    print(f"wrote {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    if "--preview" in sys.argv:
        SUBMIT = False
        build(blind=False)
        sys.exit(0)
    build(blind=True)
    build(blind=False)
    sys.exit(0)
