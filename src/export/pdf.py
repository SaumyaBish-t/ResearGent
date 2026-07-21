"""
PDF exporter for literature reviews.
Uses ReportLab to build a beautifully formatted academic-style PDF.
"""

from __future__ import annotations

import re
from datetime import date
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    HRFlowable,
    Preformatted,
    KeepTogether,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch


def markdown_to_pdf_bytes(title: str, markdown: str) -> bytes:
    """
    Parse the literature review markdown and compile it into a styled PDF.
    Returns the PDF as raw bytes.
    """
    buffer = BytesIO()
    
    # Page setup - Standard academic 1-inch margins
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    
    styles = getSampleStyleSheet()
    
    # Custom academic styles (using serif Times-Roman for a classic scholarly feel)
    title_style = ParagraphStyle(
        "AcademicTitle",
        parent=styles["Normal"],
        fontName="Times-Bold",
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#0f172a"),  # Deep slate
        alignment=1,  # Centered
        spaceAfter=15,
    )
    
    meta_style = ParagraphStyle(
        "AcademicMeta",
        parent=styles["Normal"],
        fontName="Times-Italic",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748b"),  # Cool grey
        alignment=1,  # Centered
        spaceAfter=20,
    )
    
    h1_style = ParagraphStyle(
        "AcademicH1",
        parent=styles["Normal"],
        fontName="Times-Bold",
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=18,
        spaceAfter=10,
        keepWithNext=True,
    )
    
    h2_style = ParagraphStyle(
        "AcademicH2",
        parent=styles["Normal"],
        fontName="Times-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#334155"),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True,
    )
    
    body_style = ParagraphStyle(
        "AcademicBody",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=10.5,
        leading=15,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=10,
        firstLineIndent=0.25 * inch,  # Classic academic paragraph indent
    )
    
    # Overrides firstLineIndent for the first paragraph of a section
    body_first_style = ParagraphStyle(
        "AcademicBodyFirst",
        parent=body_style,
        firstLineIndent=0,
    )
    
    code_style = ParagraphStyle(
        "BibTeXCode",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0f172a"),
        backColor=colors.HexColor("#f8fafc"),
        borderColor=colors.HexColor("#e2e8f0"),
        borderWidth=0.5,
        borderPadding=6,
        spaceAfter=12,
    )

    story = []
    
    # Split by double-newlines or lines to parse blocks
    blocks = re.split(r'\n\n+', markdown.strip())
    
    in_code_block = False
    code_lines = []
    is_first_para_of_section = True
    
    for block in blocks:
        block = block.strip()
        if not block:
            continue
            
        # Code block handling (e.g. BibTeX references)
        if block.startswith("```"):
            if in_code_block:
                # End of code block
                in_code_block = False
                code_text = "\n".join(code_lines)
                story.append(Preformatted(code_text, code_style))
                code_lines = []
            else:
                # Start of code block
                in_code_block = True
                # Extract any lines after the ``` language tag
                lines = block.split("\n")[1:]
                # Check if it ends with ``` in the same block
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                    in_code_block = False
                    code_text = "\n".join(lines)
                    story.append(Preformatted(code_text, code_style))
                else:
                    code_lines.extend(lines)
            continue
            
        if in_code_block:
            code_lines.append(block)
            continue
            
        # Heading 1 (e.g. # Document Title)
        if block.startswith("# "):
            title_text = block[2:].strip()
            story.append(Paragraph(_clean_inline_markdown(title_text), title_style))
            is_first_para_of_section = True
            continue
            
        # Heading 2 (e.g. ## Section Name)
        if block.startswith("## "):
            sec_text = block[3:].strip()
            story.append(Paragraph(_clean_inline_markdown(sec_text), h1_style))
            is_first_para_of_section = True
            continue
            
        # Heading 3 (e.g. ### Subsection Name)
        if block.startswith("### "):
            sub_text = block[4:].strip()
            story.append(Paragraph(_clean_inline_markdown(sub_text), h2_style))
            is_first_para_of_section = True
            continue
            
        # Metadata / italic lines (e.g. _Generated by..._)
        if block.startswith("_") and block.endswith("_"):
            meta_text = block[1:-1].strip()
            story.append(Paragraph(_clean_inline_markdown(meta_text), meta_style))
            continue
            
        # Horizontal Rule (e.g. ---)
        if block == "---":
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=15, spaceBefore=10))
            is_first_para_of_section = True
            continue
            
        # Standard paragraph
        cleaned_text = _clean_inline_markdown(block)
        current_style = body_first_style if is_first_para_of_section else body_style
        story.append(Paragraph(cleaned_text, current_style))
        is_first_para_of_section = False

    # Build the document
    # Simple footer page numbering
    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont("Times-Roman", 9)
        canvas.setFillColor(colors.HexColor("#64748b"))
        page_num = canvas.getPageNumber()
        canvas.drawRightString(
            letter[0] - 0.75 * inch,
            0.4 * inch,
            f"Page {page_num}"
        )
        canvas.drawString(
            0.75 * inch,
            0.4 * inch,
            "ResearGent Literature Review"
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data


def _clean_inline_markdown(text: str) -> str:
    """
    Safely escapes XML-sensitive characters while converting markdown-style bold,
    italic, links, and code snippets into ReportLab's XML tags.
    """
    # Normalize unicode hyphens, dashes, and quotes to standard ASCII to avoid ReportLab font black squares
    text = text.replace("\u2011", "-")  # Non-breaking hyphen
    text = text.replace("\u2010", "-")  # Hyphen
    text = text.replace("\u2012", "-")  # Figure dash
    text = text.replace("\u2013", "-")  # En-dash
    text = text.replace("\u2014", " -- ")  # Em-dash
    text = text.replace("\u2015", "-")  # Horizontal bar
    text = text.replace("\u2018", "'").replace("\u2019", "'")  # Smart single quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"')  # Smart double quotes

    # 1. Escape basic XML entities first
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    
    # 2. Convert Bold (**text** or __text__)
    text = re.sub(r'\*\*(.*?)\*\*|__(.*?)__', r'<b>\1\2</b>', text)
    
    # 3. Convert Italic (*text* or _text_)
    # Guarding against underscores in URLs or other contexts
    text = re.sub(r'\*(.*?)\*|_(.*?)_', r'<i>\1\2</i>', text)
    
    # 4. Highlight citations like [S1], [S2]
    text = re.sub(r'\[(S\d+)\]', r'<b>[\1]</b>', text)
    
    # 5. Convert inline backticks (`code`)
    text = re.sub(r'`(.*?)`', r'<font name="Courier" color="#0f172a"><b>\1</b></font>', text)
    
    # 6. Convert markdown links [text](url) to ReportLab links
    # ReportLab supports <a href="url">text</a>
    text = re.sub(
        r'\[(.*?)\]\((.*?)\)',
        r'<a href="\2"><font color="#0284c7"><u>\1</u></font></a>',
        text
    )
    
    return text
