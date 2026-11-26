"""Build report/PDC_Project_Report.pdf from report/REPORT.md (headings, tables,
lists, inline bold/italic/code) with the results graphs embedded."""
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.fonts import addMapping
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (HRFlowable, Image, KeepTogether, ListFlowable,
                                ListItem, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
MD = (ROOT / "report" / "REPORT.md").read_text(encoding="utf-8")
OUT = ROOT / "report" / "PDC_Project_Report.pdf"

F = r"C:\Windows\Fonts"
for n, f in (("Calibri", "calibri.ttf"), ("Calibri-Bold", "calibrib.ttf"),
             ("Calibri-Italic", "calibrii.ttf"), ("Calibri-BoldItalic", "calibriz.ttf"),
             ("Consolas", "consola.ttf")):
    pdfmetrics.registerFont(TTFont(n, rf"{F}\{f}"))
addMapping("Calibri", 0, 0, "Calibri"); addMapping("Calibri", 1, 0, "Calibri-Bold")
addMapping("Calibri", 0, 1, "Calibri-Italic"); addMapping("Calibri", 1, 1, "Calibri-BoldItalic")

NAVY, GREY, RULE, SHADE = (colors.HexColor(c) for c in ("#1F3A5F", "#5A6270", "#C9D1DC", "#F3F5F8"))
S = {
    "body": ParagraphStyle("b", fontName="Calibri", fontSize=10.3, leading=13.6, spaceAfter=5),
    "h1": ParagraphStyle("1", fontName="Calibri-Bold", fontSize=15, leading=18, textColor=NAVY, spaceBefore=12, spaceAfter=4),
    "h2": ParagraphStyle("2", fontName="Calibri-Bold", fontSize=12.5, leading=15, textColor=NAVY, spaceBefore=10, spaceAfter=3),
    "cell": ParagraphStyle("c", fontName="Calibri", fontSize=9.3, leading=12),
    "chead": ParagraphStyle("ch", fontName="Calibri-Bold", fontSize=9.3, leading=12),
    "quote": ParagraphStyle("q", fontName="Calibri-Italic", fontSize=9.6, leading=12.8, textColor=GREY, leftIndent=10, spaceAfter=5),
    "title": ParagraphStyle("t", fontName="Calibri-Bold", fontSize=19, leading=23, textColor=NAVY, spaceAfter=2),
    "sub": ParagraphStyle("s", fontName="Calibri", fontSize=11, leading=14, textColor=GREY, spaceAfter=10),
}

def inline(t):
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"`([^`]+)`", r'<font face="Consolas" size="8.8">\1</font>', t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<i>\1</i>", t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'\1 <font size="8" color="#5A6270">(\2)</font>', t)
    return t

def md_table(rows):
    data = [[Paragraph(inline(c), S["chead" if i == 0 else "cell"]) for c in r] for i, r in enumerate(rows)]
    w = (A4[0] - 36 * mm) / len(rows[0])
    t = Table(data, colWidths=[w] * len(rows[0]), repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), SHADE), ("LINEBELOW", (0, 0), (-1, 0), 0.8, RULE),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, RULE), ("LINEBELOW", (0, 0), (-1, -2), 0.25, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return [t, Spacer(1, 6)]

def graphs_block():
    imgs = []
    for png in ("speedup_24.png", "efficiency_24.png", "time_24.png", "crossover.png"):
        p = ROOT / "results" / png
        if p.exists():
            imgs.append(Image(str(p), width=81 * mm, height=81 * mm * 0.68))
    rows = [imgs[i:i + 2] for i in range(0, len(imgs), 2)]
    t = Table(rows)
    t.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("TOPPADDING", (0, 0), (-1, -1), 2)]))
    return [t, Spacer(1, 6)]

story, lines, i = [], MD.splitlines(), 0
title_done = False
while i < len(lines):
    ln = lines[i]
    if "<!--" in ln:
        i += 1; continue
    if ln.startswith("# ") and not title_done:          # title block
        story += [Paragraph(inline(ln[2:]), S["title"])]
        title_done = True
    elif ln.startswith("### "):
        story.append(Paragraph(inline(ln[4:]), S["h2"]))
    elif ln.startswith("## "):
        story.append(Paragraph(inline(ln[3:]), S["h1"]))
        story.append(HRFlowable(width="100%", thickness=0.6, color=RULE, spaceAfter=5))
    elif ln.startswith("|"):
        rows = []
        while i < len(lines) and lines[i].startswith("|"):
            cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                rows.append(cells)
            i += 1
        story += md_table(rows)
        continue
    elif ln.startswith("> Graphs"):
        story += graphs_block()
    elif ln.startswith(">"):
        story.append(Paragraph(inline(ln.lstrip("> ")), S["quote"]))
    elif ln.startswith(("- ", "* ")):
        items = []
        while i < len(lines) and lines[i].startswith(("- ", "* ")):
            buf = lines[i][2:]
            while i + 1 < len(lines) and lines[i + 1].startswith("  ") and not lines[i + 1].lstrip().startswith(("-", "*", "|")):
                i += 1; buf += " " + lines[i].strip()
            items.append(ListItem(Paragraph(inline(buf), S["body"]), leftIndent=12))
            i += 1
        story.append(ListFlowable(items, bulletType="bullet", start="•", leftIndent=12, bulletFontName="Calibri"))
        story.append(Spacer(1, 4))
        continue
    elif ln.strip() in ("---", "***"):
        pass
    elif ln.strip():
        buf = ln
        while i + 1 < len(lines) and lines[i + 1].strip() and not lines[i + 1].startswith(("#", "|", "- ", "* ", ">", "---")) and "<!--" not in lines[i + 1]:
            i += 1; buf += " " + lines[i].strip()
        if not title_done or (title_done and len(story) == 1 and buf.startswith("**PDC")):
            pass
        story.append(Paragraph(inline(buf), S["sub"] if len(story) <= 2 else S["body"]))
    i += 1

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                        topMargin=15 * mm, bottomMargin=15 * mm,
                        title="Hybrid MPI + OpenMP Parallel FFT for Large Speech Signals — PDC Project Report",
                        author="M Tashkeel Pasha, M Kashan, Hamadullah")
doc.build(story)
from pypdf import PdfReader
print(f"wrote {OUT} ({len(PdfReader(str(OUT)).pages)} pages)")
