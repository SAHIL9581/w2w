# backend/reporting/pdf_generator.py
import io
from pathlib import Path
from typing import List
import re
from html import escape  # Import for escaping XML characters

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle, HRFlowable, KeepTogether)
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT

OUTPUT_DIR = Path("/app/output")

def create_well_report_pdf(well_names: List[str], ai_summary: str) -> io.BytesIO:
    """
    Generates a professional PDF report with enhanced readability,
    larger fonts, and proper formatting for technical reports.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        rightMargin=0.75*inch, leftMargin=0.75*inch,
        topMargin=0.75*inch, bottomMargin=0.75*inch
    )
    styles = getSampleStyleSheet()
    
    # Professional style definitions
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=20,
        spaceAfter=20,
        spaceBefore=15,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#1f4e79'),
        fontName='Times-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'SubTitle',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=15,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#4472c4'),
        fontName='Times-Roman'
    )
    
    well_header_style = ParagraphStyle(
        'WellHeader',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=15,
        spaceBefore=20,
        alignment=TA_CENTER,
        textColor=colors.white,
        fontName='Times-Bold',
        borderWidth=0,
        borderPadding=12,
        backColor=colors.HexColor('#c5504b')
    )
    
    image_title_style = ParagraphStyle(
        'ImageTitle',
        parent=styles['Normal'],
        fontSize=12,
        alignment=TA_CENTER,
        spaceAfter=8,
        spaceBefore=6,
        textColor=colors.HexColor('#2f5597'),
        fontName='Times-Bold'
    )
    
    # Main summary text style
    summary_text_style = ParagraphStyle(
        'SummaryText',
        parent=styles['BodyText'],
        alignment=TA_JUSTIFY,
        spaceAfter=12,
        fontSize=11,
        leading=16,
        fontName='Times-Roman',
        leftIndent=0,
        rightIndent=0
    )

    def safe_paragraph_text(text):
        """Safely escape XML characters while preserving intended formatting"""
        if not text:
            return ""
        
        # First escape XML special characters
        safe_text = escape(text, quote=False)
        
        # Then restore intentional HTML tags that we want to keep
        # This allows us to keep <b>, <br/> etc. while escaping problematic characters
        return safe_text

    def format_important_terms(text):
        """Bold important technical terms - AFTER escaping"""
        important_patterns = [
            r'\b(sandstone|sandstones|shale|shales|reservoir quality|porosity|permeability|lithology|lithologies|gamma ray|density)\b',
            r'\b(cluster|clusters|t-SNE|elbow method|machine learning|K-means|optimal cluster count)\b',
            r'\b(best reservoir|superior reservoir|highest porosity|lowest clay content|good reservoir quality)\b',
            r'\b(production potential|investment|economic|commercial viability|drilling targets)\b',
            r'\b(recommended|priority|immediate action|next steps|core sampling|pressure testing)\b'
        ]
        
        formatted_text = text
        for pattern in important_patterns:
            formatted_text = re.sub(pattern, r'<b>\1</b>', formatted_text, flags=re.IGNORECASE)
        
        return formatted_text

    story = []
    
    # Title section
    story.append(HRFlowable(width="100%", thickness=3, color=colors.HexColor('#1f4e79')))
    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph("Well Log Analysis Report", title_style))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Machine Learning-Based Reservoir Characterization", 
                          ParagraphStyle('Subtitle2', parent=subtitle_style, fontSize=12, 
                                       textColor=colors.HexColor('#666666'))))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(f"Wells Analyzed: {', '.join(well_names)}", subtitle_style))
    story.append(Spacer(1, 0.25 * inch))
    story.append(HRFlowable(width="100%", thickness=3, color=colors.HexColor('#1f4e79')))
    story.append(Spacer(1, 0.3 * inch))

    # Well analysis sections
    for i, well_name in enumerate(well_names):
        safe_folder_name = well_name.replace("/", "_").replace("\\", "_")
        well_dir = OUTPUT_DIR / safe_folder_name
        if not well_dir.is_dir(): 
            continue
        
        well_section = []
        
        # Well header - SAFE TEXT
        safe_well_name = safe_paragraph_text(well_name)
        header_table = Table(
            [[Paragraph(f"Analysis for Well: {safe_well_name}", well_header_style)]],
            colWidths=[7*inch],
            style=TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#c5504b')),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'), 
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('TOPPADDING', (0,0), (-1,-1), 12), 
                ('BOTTOMPADDING', (0,0), (-1,-1), 12),
            ])
        )
        well_section.append(header_table)
        well_section.append(Spacer(1, 0.2 * inch))
        
        # Images
        plot_files = [
            ("Well Log with Classification", "well_log_cluster.png"),
            ("Elbow Method", "elbow.png"),
            ("t-SNE Visualization", "tsne.png")
        ]
        
        available_images = [(title, well_dir / filename) for title, filename in plot_files 
                           if (well_dir / filename).is_file()]
        
        for j in range(0, len(available_images), 2):
            if j + 1 < len(available_images):
                # Two images side by side
                title1, path1 = available_images[j]
                title2, path2 = available_images[j + 1]
                img1 = Image(path1, width=3.4*inch, height=2.5*inch, kind='proportional')
                img2 = Image(path2, width=3.4*inch, height=2.5*inch, kind='proportional')
                
                # SAFE TITLES
                safe_title1 = safe_paragraph_text(title1)
                safe_title2 = safe_paragraph_text(title2)
                
                image_table = Table(
                    [[Paragraph(safe_title1, image_title_style), Paragraph(safe_title2, image_title_style)], 
                     [img1, img2]],
                    colWidths=[3.5*inch, 3.5*inch],
                    style=TableStyle([
                        ('ALIGN', (0,0), (-1,-1), 'CENTER'), 
                        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                        ('TOPPADDING', (0,0), (-1,-1), 8), 
                        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
                        ('LEFTPADDING', (0,0), (-1,-1), 5), 
                        ('RIGHTPADDING', (0,0), (-1,-1), 5),
                        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#d9d9d9')),
                        ('LINEBELOW', (0,0), (-1,0), 1, colors.HexColor('#4472c4')),
                        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f8f9fa')),
                    ])
                )
            else:
                # Single image
                title1, path1 = available_images[j]
                img1 = Image(path1, width=4*inch, height=3*inch, kind='proportional')
                
                # SAFE TITLE
                safe_title1 = safe_paragraph_text(title1)
                
                image_table = Table(
                    [[Paragraph(safe_title1, image_title_style)], [img1]],
                    colWidths=[4*inch],
                    style=TableStyle([
                        ('ALIGN', (0,0), (-1,-1), 'CENTER'), 
                        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                        ('TOPPADDING', (0,0), (-1,-1), 8), 
                        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
                        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#d9d9d9')),
                        ('LINEBELOW', (0,0), (-1,0), 1, colors.HexColor('#4472c4')),
                        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f8f9fa')),
                    ])
                )
            
            well_section.append(image_table)
            well_section.append(Spacer(1, 0.2 * inch))
        
        story.append(KeepTogether(well_section))
        
        # Separator between wells
        if i < len(well_names) - 1:
            story.append(Spacer(1, 0.15 * inch))
            story.append(HRFlowable(width="80%", thickness=2, color=colors.HexColor('#c5504b')))
            story.append(Spacer(1, 0.2 * inch))
            
    # Executive Summary section
    story.append(PageBreak())
    story.append(HRFlowable(width="100%", thickness=3, color=colors.HexColor('#1f4e79')))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Executive Summary & Petrophysical Assessment", title_style))
    story.append(Spacer(1, 0.2 * inch))
    story.append(HRFlowable(width="100%", thickness=3, color=colors.HexColor('#1f4e79')))
    story.append(Spacer(1, 0.25 * inch))
    
    # Process and format the AI summary - FIXED: Proper XML escaping
    try:
        # Clean up markdown first
        cleaned_summary = ai_summary.replace('**', '').replace('##', '').replace('#', '').replace('*', '')
        
        # Split into paragraphs and process each safely
        paragraphs = cleaned_summary.split('\n\n')
        for paragraph_text in paragraphs:
            if paragraph_text.strip():
                # First escape the text safely
                safe_text = safe_paragraph_text(paragraph_text.strip())
                
                # Check if it's a header (contains keywords or is in caps)
                if (paragraph_text.strip().isupper() or 
                    any(keyword in paragraph_text.lower() for keyword in ['summary', 'analysis', 'assessment', 'comparison', 'recommendations', 'implications', 'methodology'])):
                    # Make it bold header
                    header_para = Paragraph(f"<b>{safe_text}</b>", 
                                          ParagraphStyle('HeaderPara', parent=summary_text_style, fontSize=13, textColor=colors.HexColor('#c5504b')))
                    story.append(header_para)
                    story.append(Spacer(1, 0.1 * inch))
                else:
                    # Apply technical term formatting AFTER escaping
                    formatted_text = format_important_terms(safe_text)
                    para = Paragraph(formatted_text, summary_text_style)
                    story.append(para)
                    story.append(Spacer(1, 0.12 * inch))
    
    except Exception as e:
        # Fallback: Simple plain text if formatting fails
        fallback_text = safe_paragraph_text(ai_summary.replace('**', '').replace('##', '').replace('#', '').replace('*', ''))
        para = Paragraph(fallback_text, summary_text_style)
        story.append(para)

    doc.build(story)
    buffer.seek(0)
    return buffer
