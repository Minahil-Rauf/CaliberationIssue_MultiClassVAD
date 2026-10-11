"""Put the paper (main.tex) into the official ICEET 2026 Word template.

    python paper/build_docx.py

Reads main.tex (text, tables, captions), main.bbl (formatted references, produced by
`tectonic --keep-intermediates main.tex`), figures/*.png and ICEET-2026-Template.docx,
and writes ICEET2026_Right_but_Unsure.docx. All styles, page setup, header and footer
come from the template; headings, captions and references use its numbered styles.
The template is Strict OOXML, so every element is written in Strict namespaces.
"""
from __future__ import annotations

import re
import shutil
import struct
import sys
import tempfile
import unicodedata
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "ICEET-2026-Template.docx"
OUT = HERE / "ICEET2026_Right_but_Unsure.docx"

AFFILIATION = ["School of Electrical Engineering and Computer Science (SEECS)",
               "National University of Sciences and Technology (NUST)", "Islamabad, Pakistan"]
AUTHORS = [  # (name, lines below the name); bracketed lines are highlighted for the authors to fill
    ("Minahil Rauf", AFFILIATION + ["mrauf.bese23seecs@seecs.edu.pk"]),
    ("Muhammad Ijaz", AFFILIATION + ["muhammad.bese23seecs@seecs.edu.pk"]),
]

TEXT_W_PT, COL_W_PT = 521.3, 252.0      # full text width and one column (template margins)
# Word has no float placement: a full-width float sits exactly where it is inserted. Floats listed
# here are emitted just before the subsection with the given title instead, so that a page does not
# end in a large gap when the float does not fit.
DEFER_WIDE = {"fig:method": "Calibration measures and post-hoc calibration"}
# Column floats listed here are emitted after the given number of following body paragraphs.
DEFER_PARAS = {"fig:reliability": 1}
FULL_FIG_IN, COL_FIG_IN = 7.0, 3.45
EMU_PER_IN = 914400
IMG_REL = "http://purl.oclc.org/ooxml/officeDocument/relationships/image"
SPMU = "speech+music"


# ------------------------------------------------------------------ XML helpers
def esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def run(text: str, b=False, i=False, sub=False, sup=False, size=None, highlight=None, font=None) -> str:
    rpr = ""
    if font:
        rpr += f'<w:rFonts w:ascii="{font}" w:hAnsi="{font}" w:cs="{font}"/>'
    if b:
        rpr += "<w:b/><w:bCs/>"
    if i:
        rpr += "<w:i/><w:iCs/>"
    if size:
        rpr += f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'
    if highlight:
        rpr += f'<w:highlight w:val="{highlight}"/>'
    if sub or sup:
        rpr += f'<w:vertAlign w:val="{"subscript" if sub else "superscript"}"/>'
    rpr = f"<w:rPr>{rpr}</w:rPr>" if rpr else ""
    return f'<w:r>{rpr}<w:t xml:space="preserve">{esc(text)}</w:t></w:r>'


def runs(segs, **extra) -> str:
    return "".join(run(t, **{**f, **extra}) for t, f in segs if t)


def para(content: str, style: str | None = None, ppr: str = "", sect: str = "") -> str:
    p = (f'<w:pStyle w:val="{style}"/>' if style else "") + ppr + sect
    return f"<w:p>{'<w:pPr>' + p + '</w:pPr>' if p else ''}{content}</w:p>"


def section_break(cols: int) -> str:
    """Tiny empty paragraph that ends a continuous section with `cols` columns."""
    cols_xml = '<w:cols w:num="2" w:space="18pt"/>' if cols == 2 else '<w:cols w:space="36pt"/>'
    sect = ('<w:sectPr><w:type w:val="continuous"/><w:pgSz w:w="612pt" w:h="792pt" w:code="1"/>'
            '<w:pgMar w:top="54pt" w:right="45.35pt" w:bottom="72pt" w:left="45.35pt" w:header="36pt" '
            f'w:footer="36pt" w:gutter="0pt"/>{cols_xml}<w:docGrid w:linePitch="360"/></w:sectPr>')
    ppr = ('<w:spacing w:before="0pt" w:after="0pt" w:line="1pt" w:lineRule="exact"/>'
           '<w:rPr><w:sz w:val="2"/><w:szCs w:val="2"/></w:rPr>')
    return para("", ppr=ppr, sect=sect)


# ------------------------------------------------------------------ LaTeX -> text segments
GREEK = {"tau": "τ", "gamma": "γ", "varepsilon": "ε", "epsilon": "ε"}
MATH_SYM = {"pm": "±", "approx": "≈", "in": "∈", "odot": "⊙", "dots": "…", "uparrow": "↑",
            "downarrow": "↓", "geq": "≥", "leq": "≤", "times": "×"}
BINARY = {"=", "<", ">", "≈", "∈", "±", "+", "−", "⊙", "≥", "≤"}


def _group(s: str, i: int) -> tuple[str, int]:
    """Balanced {...} starting at s[i] == '{'; returns (inner, index after)."""
    assert s[i] == "{", s[i:i + 30]
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
    raise ValueError("unbalanced braces: " + s[i:i + 40])


def math_segs(m: str) -> list:
    out, i = [], 0

    def add(t, **f):
        out.append((t, f))

    def script(i, kind):
        if m[i] == "{":
            inner, j = _group(m, i)
        else:
            inner, j = m[i], i + 1
        for t, f in math_segs(inner):
            out.append((t, {**f, kind: True}))
        return j

    while i < len(m):
        c = m[i]
        if c == "\\":
            mm = re.match(r"\\([a-zA-Z]+|.)", m[i:])
            cmd, i = mm.group(1), i + len(mm.group(0))
            if cmd in GREEK:
                add(GREEK[cmd], i=True)
            elif cmd in MATH_SYM:
                sym = MATH_SYM[cmd]
                add(f" {sym} " if sym in BINARY else sym)
            elif cmd in ("{", "}"):
                add(cmd)
            elif cmd == "max":
                add("max")
            elif cmd in ("mathbf", "mathrm", "hat"):
                inner, i = _group(m, i)
                if cmd == "mathbf":
                    add(inner, b=True)
                elif cmd == "mathrm":
                    add(inner)
                else:
                    add(unicodedata.normalize("NFC", inner + "\u0302"), i=True)
            elif cmd == ",":
                add("\u00a0")
            else:
                raise ValueError(f"unhandled math command \\{cmd} in {m!r}")
        elif c == "_":
            i = script(i + 1, "sub")
            if m[i:i + 1] == " ":   # e.g. \max_k p
                add(" ")
        elif c == "^":
            i = script(i + 1, "sup")
        elif c.isalpha():
            add(c, i=True)
            i += 1
        elif c == "-":
            add("−" if not out else " − ")
            i += 1
        elif c in "=<>+":
            add(f" {c} ")
            i += 1
        elif c == ",":
            add(", ")
            i += 1
        elif c == " ":
            i += 1
        else:
            add(c)
            i += 1
    # merge, and drop doubled spaces created by adjacent operators
    merged = []
    for t, f in out:
        if merged and merged[-1][1] == f:
            merged[-1] = (merged[-1][0] + t, f)
        else:
            merged.append((t, f))
    return [(re.sub(r"  +", " ", t), f) for t, f in merged]


class Converter:
    def __init__(self, cite_num: dict[str, int], labels: dict[str, str]):
        self.cite_num, self.labels = cite_num, labels

    def cite(self, keys: str) -> str:
        nums = sorted(self.cite_num[k.strip()] for k in keys.split(","))
        parts, start = [], 0
        while start < len(nums):
            end = start
            while end + 1 < len(nums) and nums[end + 1] == nums[end] + 1:
                end += 1
            if end - start >= 2:
                parts.append(f"[{nums[start]}]–[{nums[end]}]")
            else:
                parts.extend(f"[{n}]" for n in nums[start:end + 1])
            start = end + 1
        return ", ".join(parts)

    def segs(self, s: str, fmt: dict | None = None) -> list:
        fmt = fmt or {}
        out, buf, i = [], "", 0

        def flush():
            nonlocal buf
            if buf:
                out.append((buf, dict(fmt)))
                buf = ""

        while i < len(s):
            c = s[i]
            if s.startswith("---", i):
                buf += "—"; i += 3
            elif s.startswith("--", i):
                buf += "–"; i += 2
            elif s.startswith("``", i):
                buf += "“"; i += 2
            elif s.startswith("''", i):
                buf += "”"; i += 2
            elif s.startswith("{,}", i):
                buf += ","; i += 3
            elif c == "~":
                buf += "\u00a0"; i += 1
            elif c == "$":
                j = s.index("$", i + 1)
                flush()
                out.extend((t, {**fmt, **f}) for t, f in math_segs(s[i + 1:j]))
                i = j + 1
            elif c == "\\":
                mm = re.match(r"\\([a-zA-Z]+|.)", s[i:])
                cmd, i = mm.group(1), i + len(mm.group(0))
                if cmd in ("%", "&", "#", "_", "{", "}"):
                    buf += cmd
                elif cmd == ",":
                    buf += "\u00a0"
                elif cmd == " ":
                    buf += " "
                elif cmd == "dots":
                    buf += "…"
                elif cmd == "spmu":
                    buf += SPMU
                    if s.startswith("{}", i):
                        i += 2
                elif cmd in ("textbf", "emph"):
                    inner, i = _group(s, i)
                    flush()
                    key = "b" if cmd == "textbf" else "i"
                    out.extend(self.segs(inner, {**fmt, key: not fmt.get(key, False)}))
                elif cmd == "cite":
                    inner, i = _group(s, i)
                    buf += self.cite(inner)
                elif cmd == "ref":
                    inner, i = _group(s, i)
                    buf += self.labels[inner]
                elif cmd == "label":
                    _, i = _group(s, i)
                else:
                    raise ValueError(f"unhandled text command \\{cmd}: {s[i - 20:i + 30]!r}")
            elif c in "{}":
                i += 1
            elif c in " \n\t":
                if not buf.endswith(" ") and not (not buf and out and out[-1][0].endswith(" ")):
                    buf += " "
                i += 1
            else:
                buf += c; i += 1
        flush()
        return out


# ------------------------------------------------------------------ bibliography
ACCENT = {"'": "\u0301", "`": "\u0300", '"': "\u0308", "^": "\u0302", "~": "\u0303"}


def bbl_entries(path: Path) -> list[tuple[str, list]]:
    s = path.read_text(encoding="utf-8")
    s = s[s.index(r"\bibitem"):s.index(r"\end{thebibliography}")]
    entries = []
    for chunk in s.split(r"\bibitem")[1:]:
        key, rest = re.match(r"\{([^}]+)\}(.*)", chunk, re.S).groups()
        rest = re.sub(r"\\BIBentry(ALT|STD)interwordspacing", "", rest).replace(r"\newblock", "")
        rest = re.sub(r"\\url\{([^}]*)\}", r"\1", rest)
        rest = rest.replace(r"{\ss}", "ß").replace(r"{\O}", "Ø").replace(r"{\o}", "ø")
        rest = re.sub(r"\{?\\(['`\"^~])\{?([A-Za-z])\}?\}?",
                      lambda m: unicodedata.normalize("NFC", m.group(2) + ACCENT[m.group(1)]), rest)
        entries.append((key, Converter({}, {}).segs(" ".join(rest.split()))))
    return entries


# ------------------------------------------------------------------ tables
def parse_colspec(spec: str) -> list[dict]:
    cols, i, bar = [], 0, False
    while i < len(spec):
        c = spec[i]
        if c == "@" or c == ">":
            _, i = _group(spec, i + 1)
            continue
        if c == "|":
            bar = True
        elif c in "lcrYX":
            cols.append({"type": c, "bar_left": bar}); bar = False
        elif c == "p":
            w, j = _group(spec, i + 1)
            cols.append({"type": "p", "width_cm": float(w.replace("cm", "")), "bar_left": bar}); bar = False
            i = j
            continue
        i += 1
    return cols


def col_widths(cols: list[dict], total_pt: float, first_pt: float | None = None) -> list[float]:
    fixed = {k: c["width_cm"] * 28.3465 for k, c in enumerate(cols) if c["type"] == "p"}
    if first_pt and 0 not in fixed:
        fixed[0] = first_pt
    if any(c["type"] in "XY" for c in cols):
        narrow = {k: 34.0 for k, c in enumerate(cols) if c["type"] in "lcr" and k not in fixed}
        fixed.update(narrow)
        flex = [k for k, c in enumerate(cols) if c["type"] in "XY"]
        rest = (total_pt - sum(fixed.values())) / len(flex)
        return [fixed.get(k, rest) for k in range(len(cols))]
    free = [k for k in range(len(cols)) if k not in fixed]
    rest = (total_pt - sum(fixed.values())) / len(free)
    return [fixed.get(k, rest) for k in range(len(cols))]


def table_xml(conv: Converter, spec: str, body: str, total_pt: float, first_pt: float | None) -> str:
    cols = parse_colspec(spec)
    widths = col_widths(cols, total_pt, first_pt)
    rows, rule_before = [], set()
    for line in re.split(r"\\\\", body):
        line = line.strip()
        while True:
            m = re.match(r"\\(toprule|midrule|bottomrule|hline)\s*", line)
            if not m:
                break
            rule_before.add(len(rows))
            line = line[m.end():]
        if line:
            rows.append(line)
    border = '<w:{0} w:val="single" w:sz="{1}" w:space="0" w:color="000000"/>'
    xml = [f'<w:tbl><w:tblPr><w:tblW w:w="{total_pt:.1f}pt" w:type="dxa"/><w:jc w:val="center"/>'
           '<w:tblLayout w:type="fixed"/><w:tblCellMar><w:start w:w="2.5pt" w:type="dxa"/>'
           '<w:end w:w="2.5pt" w:type="dxa"/></w:tblCellMar>'
           '<w:tblLook w:firstRow="0" w:lastRow="0" w:firstColumn="0" w:lastColumn="0" w:noHBand="1" w:noVBand="1"/>'
           "</w:tblPr><w:tblGrid>"]
    xml += [f'<w:gridCol w:w="{round(w * 20)}"/>' for w in widths]
    xml.append("</w:tblGrid>")
    n_rows = len(rows)
    for r, line in enumerate(rows):
        cells = re.split(r"(?<!\\)&", line)
        xml.append('<w:tr><w:trPr><w:cantSplit/><w:jc w:val="center"/></w:trPr>')
        col = 0
        for cell in cells:
            cell = cell.strip()
            span, cspec = 1, None
            m = re.match(r"\\multicolumn\{(\d+)\}\{([^}]*)\}", cell)
            if m:
                span, cspec = int(m.group(1)), m.group(2)
                cell, _ = _group(cell, m.end())
            c0 = cols[col]
            align = {"l": "start", "c": "center", "r": "end"}.get((cspec or c0["type"]).strip("|")[:1], "start")
            width = sum(widths[col:col + span])
            b = ""
            if r in rule_before:
                b += border.format("top", 8 if r == 0 else 4)
            if c0["bar_left"] and col > 0:
                b += border.format("start", 4)
            if r == n_rows - 1:
                b += border.format("bottom", 8)
            elif (r + 1) in rule_before:
                b += border.format("bottom", 4)
            if cspec and cspec.endswith("|"):
                b += border.format("end", 4)
            tcpr = (f'<w:tcW w:w="{width:.1f}pt" w:type="dxa"/>' + (f'<w:gridSpan w:val="{span}"/>' if span > 1 else "")
                    + (f"<w:tcBorders>{b}</w:tcBorders>" if b else "") + '<w:vAlign w:val="center"/>')
            content = runs(conv.segs(cell))
            ppr = f'<w:spacing w:before="1pt" w:after="1pt"/><w:jc w:val="{align}"/>'
            xml.append(f"<w:tc><w:tcPr>{tcpr}</w:tcPr>{para(content, 'tablecopy', ppr)}</w:tc>")
            col += span
        xml.append("</w:tr>")
    xml.append("</w:tbl>")
    return "".join(xml)


# ------------------------------------------------------------------ figures
class Media:
    def __init__(self):
        self.items, self.next_id = [], 1

    def inline(self, png: Path, width_in: float) -> str:
        w_px, h_px = struct.unpack(">II", png.read_bytes()[16:24])
        cx = int(width_in * EMU_PER_IN)
        cy = int(cx * h_px / w_px)
        rid, n = f"rIdFig{self.next_id}", self.next_id
        self.items.append((rid, png))
        self.next_id += 1
        a, pic = "http://purl.oclc.org/ooxml/drawingml/main", "http://purl.oclc.org/ooxml/drawingml/picture"
        return (f'<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
                f'<wp:docPr id="{1000 + n}" name="Figure {n}"/><wp:cNvGraphicFramePr>'
                f'<a:graphicFrameLocks xmlns:a="{a}" noChangeAspect="1"/></wp:cNvGraphicFramePr>'
                f'<a:graphic xmlns:a="{a}"><a:graphicData uri="{pic}"><pic:pic xmlns:pic="{pic}">'
                f'<pic:nvPicPr><pic:cNvPr id="{n}" name="{png.name}"/><pic:cNvPicPr/></pic:nvPicPr>'
                f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
                f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
                f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData>'
                "</a:graphic></wp:inline></w:drawing></w:r>")


ECE_OMML = (
    '<m:oMath><m:r><m:rPr><m:sty m:val="p"/></m:rPr><m:t>ECE</m:t></m:r><m:r><m:t>=</m:t></m:r>'
    '<m:nary><m:naryPr><m:chr m:val="∑"/><m:limLoc m:val="undOvr"/></m:naryPr>'
    '<m:sub><m:r><m:t>b=1</m:t></m:r></m:sub><m:sup><m:r><m:t>B</m:t></m:r></m:sup><m:e>'
    '<m:f><m:num><m:d><m:dPr><m:begChr m:val="|"/><m:endChr m:val="|"/></m:dPr><m:e>'
    '<m:sSub><m:e><m:r><m:t>S</m:t></m:r></m:e><m:sub><m:r><m:t>b</m:t></m:r></m:sub></m:sSub>'
    '</m:e></m:d></m:num><m:den><m:r><m:t>N</m:t></m:r></m:den></m:f>'
    '<m:d><m:dPr><m:begChr m:val="|"/><m:endChr m:val="|"/></m:dPr><m:e>'
    '<m:r><m:rPr><m:sty m:val="p"/></m:rPr><m:t>acc</m:t></m:r>'
    '<m:d><m:e><m:sSub><m:e><m:r><m:t>S</m:t></m:r></m:e><m:sub><m:r><m:t>b</m:t></m:r></m:sub></m:sSub></m:e></m:d>'
    '<m:r><m:t>−</m:t></m:r><m:r><m:rPr><m:sty m:val="p"/></m:rPr><m:t>conf</m:t></m:r>'
    '<m:d><m:e><m:sSub><m:e><m:r><m:t>S</m:t></m:r></m:e><m:sub><m:r><m:t>b</m:t></m:r></m:sub></m:sSub></m:e></m:d>'
    "</m:e></m:d></m:e></m:nary></m:oMath>")


# ------------------------------------------------------------------ main conversion
def strip_comments(tex: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in tex.splitlines())


def build_body(tex: str, conv: Converter, media: Media, refs: list) -> list[str]:
    body = tex[tex.index(r"\end{IEEEkeywords}") + len(r"\end{IEEEkeywords}"):tex.index(r"\bibliographystyle")]
    out, deferred, pending = [], {}, []
    i = 0
    while i < len(body):
        while i < len(body) and body[i] in " \n\t":
            i += 1
        if i >= len(body):
            break
        rest = body[i:]
        m = re.match(r"\\(section|subsection)\{", rest)
        if m:
            title, j = _group(body, i + m.end() - 1)
            for label, before in DEFER_WIDE.items():
                if before == title and label in deferred:
                    out.extend(deferred.pop(label))
            out.append(para(runs(conv.segs(title)), "Heading1" if m.group(1) == "section" else "Heading2"))
            i = j
            lm = re.match(r"\s*\\label\{[^}]*\}", body[i:])
            if lm:
                i += lm.end()
            continue
        m = re.match(r"\\begin\{([a-z*]+)\}", rest)
        if m:
            env = m.group(1)
            end = body.index(r"\end{" + env + "}", i)
            inner = body[i + m.end():end]
            i = end + len(r"\end{" + env + "}")
            block = env_xml(env, inner, conv, media)
            label = re.search(r"\\label\{([^}]+)\}", inner)
            if label and label.group(1) in DEFER_WIDE:
                deferred[label.group(1)] = block
            elif label and label.group(1) in DEFER_PARAS:
                pending.append([DEFER_PARAS[label.group(1)], block])
            elif block and block[0] == "WIDE_START" and out and 'w:val="Heading1"' in out[-1]:
                out[-1:-1] = block   # a wide float right after a section heading goes above the heading
            else:
                out.extend(block)
            continue
        j = i
        while j < len(body):
            if body.startswith("\n\n", j) or re.match(r"\n\s*\n", body[j:j + 3]) or \
                    re.match(r"\\(section|subsection|begin)\{", body[j:]):
                break
            j += 1
        text = body[i:j].strip()
        if text:
            out.append(para(runs(conv.segs(text)), "BodyText"))
            for item in pending:
                item[0] -= 1
            out.extend(b for n, block in pending if n <= 0 for b in block)
            pending[:] = [item for item in pending if item[0] > 0]
        i = j
    out.append(para(runs([("References", {})]), "Heading5"))
    for n, (_, segs) in enumerate(refs):
        out.append(("REF", runs(segs)))
    return out


def env_xml(env: str, inner: str, conv: Converter, media: Media) -> list:
    if env == "enumerate":
        items = [it.strip() for it in inner.split(r"\item")[1:]]
        ppr = '<w:ind w:start="18pt" w:hanging="18pt"/>'
        return [para(run(f"{k})\u00a0") + runs(conv.segs(it)), "BodyText", ppr) for k, it in enumerate(items, 1)]
    if env == "equation":
        content = (run("\t") + ECE_OMML + run("\t(" + "1" + ")", font="Times New Roman"))
        return [para(content, "equation")]
    if env in ("figure", "figure*"):
        wide = env.endswith("*")
        g = re.search(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", inner).group(1)
        png = (HERE / g).with_suffix(".png")
        cap = _group(inner, inner.index(r"\caption{") + len(r"\caption"))[0]
        pic = para(media.inline(png, FULL_FIG_IN if wide else COL_FIG_IN), None,
                   '<w:keepNext/><w:spacing w:before="6pt" w:after="0pt"/><w:jc w:val="center"/>')
        caption = para(runs(conv.segs(cap)), "figurecaption")
        return ["WIDE_START", pic, caption, "WIDE_END"] if wide else [pic, caption]
    if env in ("table", "table*"):
        wide = env.endswith("*")
        cap = _group(inner, inner.index(r"\caption{") + len(r"\caption"))[0]
        m = re.search(r"\\begin\{(tabularx|tabular)\}", inner)
        k = m.end()
        if m.group(1) == "tabularx":
            _, k = _group(inner, k)
        spec, k = _group(inner, k)
        tbody = inner[k:inner.index(r"\end{" + m.group(1) + "}")]
        total = TEXT_W_PT if wide else COL_W_PT
        first = None if wide else 64.0
        if wide and "tab:main" in inner:
            first = 66.0
        head = para(runs(conv.segs(cap)), "tablehead", '<w:keepNext/>')
        tbl = table_xml(conv, spec, tbody, total, first)
        after = para("", None, '<w:spacing w:before="0pt" w:after="6pt"/>')
        return ["WIDE_START", head, tbl, after, "WIDE_END"] if wide else [head, tbl, after]
    raise ValueError("unhandled environment " + env)


def labels_from(tex: str) -> dict[str, str]:
    roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
    labels, sec, fig, tab = {}, 0, 0, 0
    for m in re.finditer(r"\\section\{|\\begin\{(figure|table)\*?\}|\\label\{([^}]+)\}", tex):
        if m.group(0).startswith(r"\section"):
            sec += 1; last = roman[sec - 1]
        elif m.group(1) == "figure":
            fig += 1; last = str(fig)
        elif m.group(1) == "table":
            tab += 1; last = roman[tab - 1]
        elif m.group(2):
            labels[m.group(2)] = last
    return labels


def author_block() -> str:
    parts = []
    for k, (name, lines) in enumerate(AUTHORS):
        if k:
            parts.append('<w:r><w:br w:type="column"/></w:r>')
        parts.append(run(name, size=22))
        for j, line in enumerate(lines):
            filler = line.startswith("[")
            parts.append('<w:r><w:br/></w:r>')
            parts.append(run(line, size=18, i=(j < 2), highlight="yellow" if filler else None))
    sect = ('<w:sectPr><w:type w:val="continuous"/><w:pgSz w:w="612pt" w:h="792pt" w:code="1"/>'
            '<w:pgMar w:top="54pt" w:right="44.65pt" w:bottom="72pt" w:left="44.65pt" w:header="36pt" '
            'w:footer="36pt" w:gutter="0pt"/><w:cols w:num="2" w:space="10.80pt"/><w:docGrid w:linePitch="360"/></w:sectPr>')
    return para("".join(parts), "Author", '<w:spacing w:before="5pt" w:beforeAutospacing="1"/>', sect)


def top_level(body: str) -> list[str]:
    els, depth, start = [], 0, 0
    for m in re.finditer(r"<(/?)w:(p|tbl|sectPr)\b[^>]*?(/?)>", body):
        close, selfclose = m.group(1), m.group(3)
        if selfclose:
            if depth == 0:
                els.append(body[m.start():m.end()])
            continue
        if not close:
            if depth == 0:
                start = m.start()
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                els.append(body[start:m.end()])
    return els


def main() -> None:
    tex = strip_comments((HERE / "main.tex").read_text(encoding="utf-8"))
    refs = bbl_entries(HERE / "main.bbl")
    conv = Converter({k: n for n, (k, _) in enumerate(refs, 1)}, labels_from(tex))
    media = Media()

    title = " ".join(_group(tex, tex.index(r"\title{") + len(r"\title"))[0].split())
    abstract = tex[tex.index(r"\begin{abstract}") + 16:tex.index(r"\end{abstract}")]
    keywords = tex[tex.index(r"\begin{IEEEkeywords}") + 20:tex.index(r"\end{IEEEkeywords}")]

    work = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(TEMPLATE) as z:
        z.extractall(work)
    doc_path = work / "word" / "document.xml"
    doc = doc_path.read_text(encoding="utf-8")
    head, rest = doc.split("<w:body>")
    tpl_body, tail = rest.rsplit("</w:body>", 1)
    els = top_level(tpl_body)
    assert len(els) == 98 and "papertitle" in els[0] and "w:sectPr" in els[3], "unexpected template layout"

    title_p = re.sub(r"(</w:pPr>).*(</w:p>)", lambda m: m.group(1) + run(title) + m.group(2), els[0], flags=re.S)
    abstract_p = para(run("Abstract", i=True) + run("—") + runs(conv.segs(abstract.strip())), "Abstract")
    keywords_p = para(run("Keywords—") + runs(conv.segs(keywords.strip())), "Keywords")

    content = build_body(tex, conv, media, refs)
    final_2col = re.search(r"<w:sectPr\b.*?</w:sectPr>", els[95], re.S).group(0)
    final_2col = re.sub(r' w:rsid\w*="[^"]*"', "", final_2col)

    body_xml = [title_p, els[3], author_block(), els[5], els[6], abstract_p, keywords_p]
    refs_left = sum(1 for c in content if isinstance(c, tuple))
    for c in content:
        if c == "WIDE_START":
            body_xml.append(section_break(2))
        elif c == "WIDE_END":
            body_xml.append(section_break(1))
        elif isinstance(c, tuple):
            refs_left -= 1
            ppr = '<w:ind w:start="17.70pt" w:hanging="17.70pt"/>'
            body_xml.append(para(c[1], "references", ppr, final_2col if refs_left == 0 else ""))
        else:
            body_xml.append(c)
    body_xml += [els[96], els[97]]
    # consecutive WIDE_END/WIDE_START pairs would leave an empty 2-column section between wide items: merge them
    joined = "".join(body_xml).replace(section_break(1) + section_break(2), "")
    doc_path.write_text(head + "<w:body>" + joined + "</w:body>" + tail, encoding="utf-8")

    media_dir = work / "word" / "media"
    media_dir.mkdir(exist_ok=True)
    rels_path = work / "word" / "_rels" / "document.xml.rels"
    rels = rels_path.read_text(encoding="utf-8")
    new_rels = "".join(f'<Relationship Id="{rid}" Type="{IMG_REL}" Target="media/{png.name}"/>'
                       for rid, png in media.items)
    for _, png in media.items:
        shutil.copy(png, media_dir / png.name)
    rels_path.write_text(rels.replace("</Relationships>", new_rels + "</Relationships>"), encoding="utf-8")
    ct_path = work / "[Content_Types].xml"
    ct = ct_path.read_text(encoding="utf-8")
    if 'Extension="png"' not in ct:
        ct = ct.replace("<Default Extension=\"xml\"", '<Default Extension="png" ContentType="image/png"/><Default Extension="xml"')
    ct_path.write_text(ct, encoding="utf-8")
    core = work / "docProps" / "core.xml"
    c = core.read_text(encoding="utf-8")
    c = re.sub(r"<dc:title>.*?</dc:title>", f"<dc:title>{esc(title)}</dc:title>", c)
    c = re.sub(r"<dc:creator>.*?</dc:creator>", "<dc:creator>" + esc("; ".join(a for a, _ in AUTHORS)) + "</dc:creator>", c)
    c = re.sub(r"<cp:lastModifiedBy>.*?</cp:lastModifiedBy>", "<cp:lastModifiedBy></cp:lastModifiedBy>", c)
    core.write_text(c, encoding="utf-8")

    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(ct_path, "[Content_Types].xml")
        for f in sorted(work.rglob("*")):
            if f.is_file() and f != ct_path:
                z.write(f, f.relative_to(work).as_posix())
    shutil.rmtree(work)
    print(f"wrote {OUT.name}: {len(refs)} references, {len(media.items)} figures")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
