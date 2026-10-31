import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "report_figures")
os.makedirs(FIG_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. GENERACIÓN DE GRÁFICOS E ILUSTRACIONES
# -----------------------------------------------------------------------------

def generate_benchmark_chart():
    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    models = ['ResNet50V2\n(Ganador)', 'MobileNetV2\n(Ligero)', 'EfficientNetB0\n(Descartado)']
    accuracies = [89.01, 81.20, 25.43]
    colors = ['#0891B2', '#3B82F6', '#EF4444']

    bars = ax.bar(models, accuracies, color=colors, width=0.55, edgecolor='#1E293B', linewidth=1.2)
    ax.set_ylim(0, 105)
    ax.set_ylabel('Precisión en Test (%)', fontsize=11, fontweight='bold', color='#1E293B')
    ax.set_title('Benchmark de Arquitecturas Evaluadas (Transfer Learning)', fontsize=13, fontweight='bold', pad=15, color='#0F172A')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar, acc in zip(bars, accuracies):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold', color='#0F172A')

    # Anotaciones
    ax.text(0, 45, 'Skip Connections\nRetención de microlesiones', ha='center', fontsize=8.5, color='white', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#0891B2", ec="none"))
    ax.text(1, 40, 'Separable Convolutions\nBaja potencia de cómputo', ha='center', fontsize=8.5, color='white', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#3B82F6", ec="none"))
    ax.text(2, 12, 'Compound Scaling\nProblemas de convergencia', ha='center', fontsize=8.5, color='white', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#EF4444", ec="none"))

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig1_benchmark.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path

def generate_dataset_chart():
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=300)
    classes = ['Catarata\n(Cataract)', 'Retinopatía D.\n(Diabetic Ret.)', 'Glaucoma\n(Glaucoma)', 'Normal\n(Sana)']
    train_counts = [726, 768, 704, 751]
    val_counts = [155, 164, 151, 161]
    test_counts = [157, 166, 152, 162]

    x = np.arange(len(classes))
    width = 0.25

    rects1 = ax.bar(x - width, train_counts, width, label='Entrenamiento (70%)', color='#1E40AF', edgecolor='#0F172A')
    rects2 = ax.bar(x, val_counts, width, label='Validación (15%)', color='#0284C7', edgecolor='#0F172A')
    rects3 = ax.bar(x + width, test_counts, width, label='Prueba (15%)', color='#38BDF8', edgecolor='#0F172A')

    ax.set_ylabel('Número de Imágenes', fontsize=11, fontweight='bold', color='#1E293B')
    ax.set_title('Distribución Estratificada del Dataset de Retinografías (Total: 4,217)', fontsize=12.5, fontweight='bold', pad=15, color='#0F172A')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10, fontweight='bold')
    ax.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h}', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h}', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=7.5)
    for rect in rects3:
        h = rect.get_height()
        ax.annotate(f'{h}', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=7.5)

    ax.set_ylim(0, 900)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig2_dataset_split.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path

def generate_fundus_samples_grid():
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8), dpi=300)
    categories = [
        ("cataract", "Catarata", "#E67E22"),
        ("diabetic_retinopathy", "Retinopatía Diabética", "#E74C3C"),
        ("glaucoma", "Glaucoma", "#9B59B6"),
        ("normal", "Retina Normal", "#2ECC71")
    ]

    for idx, (cls_dir, label, color) in enumerate(categories):
        folder = os.path.join(BASE_DIR, "data_split", "test", cls_dir)
        if os.path.exists(folder):
            files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if files:
                img_path = os.path.join(folder, files[0])
                img = Image.open(img_path).convert("RGB")
                axes[idx].imshow(img)
            else:
                axes[idx].text(0.5, 0.5, "Sin imagen", ha="center", va="center")
        else:
            axes[idx].text(0.5, 0.5, "Sin imagen", ha="center", va="center")

        axes[idx].set_title(label, fontsize=10.5, fontweight='bold', color=color, pad=8)
        axes[idx].axis("off")

    plt.suptitle("Muestras Reales del Conjunto de Prueba de Fondo de Ojo (224 × 224 px)", fontsize=11.5, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig3_fundus_samples.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path

def generate_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Caja 1: Entrada
    rect1 = patches.FancyBboxPatch((2, 55), 18, 35, boxstyle="round,pad=1.5", ec="#1E293B", fc="#E0F2FE", lw=1.5)
    ax.add_patch(rect1)
    ax.text(11, 80, "1. INGESTA", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#0369A1")
    ax.text(11, 70, "Imagen Retiniana\n(JPG / PNG)\nSubida Web o Preset", ha="center", va="center", fontsize=8, color="#0F172A")

    # Flecha 1
    ax.annotate("", xy=(23, 72.5), xytext=(20, 72.5), arrowprops=dict(arrowstyle="->", lw=2, color="#0284C7"))

    # Caja 2: Preprocesamiento
    rect2 = patches.FancyBboxPatch((24, 55), 21, 35, boxstyle="round,pad=1.5", ec="#1E293B", fc="#FEF3C7", lw=1.5)
    ax.add_patch(rect2)
    ax.text(34.5, 80, "2. PREPROCESO", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#B45309")
    ax.text(34.5, 70, "• Forzar RGB (PIL)\n• Resize (224×224)\n• Normalización [0, 1]\n• Expansión Lote (1,224,224,3)", ha="center", va="center", fontsize=7.8, color="#0F172A")

    # Flecha 2
    ax.annotate("", xy=(48, 72.5), xytext=(45, 72.5), arrowprops=dict(arrowstyle="->", lw=2, color="#0284C7"))

    # Caja 3: Inferencia IA
    rect3 = patches.FancyBboxPatch((49, 55), 24, 35, boxstyle="round,pad=1.5", ec="#1E293B", fc="#DCFCE7", lw=1.5)
    ax.add_patch(rect3)
    ax.text(61, 80, "3. RED RESNET50V2", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#15803D")
    ax.text(61, 70, "• 50 Capas Residuales\n• GlobalAvgPooling2D\n• BatchNorm + Dropout\n• Dense 512 & Dense 256", ha="center", va="center", fontsize=7.8, color="#0F172A")

    # Flecha 3
    ax.annotate("", xy=(76, 72.5), xytext=(73, 72.5), arrowprops=dict(arrowstyle="->", lw=2, color="#0284C7"))

    # Caja 4: Softmax & Decisión
    rect4 = patches.FancyBboxPatch((77, 55), 21, 35, boxstyle="round,pad=1.5", ec="#1E293B", fc="#F3E8FF", lw=1.5)
    ax.add_patch(rect4)
    ax.text(87.5, 80, "4. CLASIFICACIÓN", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#7E22CE")
    ax.text(87.5, 70, "• Capa Softmax (4 nodos)\n• Argmax -> Clase Ganadora\n• Confianza % (0-100)\n• Desglose Probabilístico", ha="center", va="center", fontsize=7.8, color="#0F172A")

    # Flecha hacia abajo
    ax.annotate("", xy=(87.5, 47), xytext=(87.5, 54), arrowprops=dict(arrowstyle="->", lw=2, color="#7E22CE"))

    # Caja 5: Motor Clínico
    rect5 = patches.FancyBboxPatch((49, 10), 49, 34, boxstyle="round,pad=1.5", ec="#1E293B", fc="#FFE4E6", lw=1.5)
    ax.add_patch(rect5)
    ax.text(73.5, 36, "5. MOTOR DE CONOCIMIENTO CLÍNICO (utils.py)", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#BE123C")
    ax.text(73.5, 23, "• Vinculación con Ficha Médica de Patología (Catarata, DR, Glaucoma, Normal)\n• Determinación de Nivel de Severidad ('Crítica', 'Moderada', 'Silenciosa')\n• Generación de Guía de Síntomas y Recomendaciones Oftalmológicas", ha="center", va="center", fontsize=8, color="#0F172A")

    # Flecha hacia la izquierda
    ax.annotate("", xy=(45, 27), xytext=(48, 27), arrowprops=dict(arrowstyle="->", lw=2, color="#BE123C"))

    # Caja 6: Respuesta y Dashboard
    rect6 = patches.FancyBboxPatch((2, 10), 42, 34, boxstyle="round,pad=1.5", ec="#1E293B", fc="#F1F5F9", lw=1.5)
    ax.add_patch(rect6)
    ax.text(23, 36, "6. DASHBOARD CLÍNICO WEB (Flask / UI)", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#334155")
    ax.text(23, 23, "• Emisión de JSON estructurado por API REST (/predict)\n• Renderizado reactivo con barras de probabilidad interactivas\n• Animaciones de escáner y descarga/impresión de dictamen", ha="center", va="center", fontsize=8, color="#0F172A")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig4_pipeline_flow.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path

# -----------------------------------------------------------------------------
# 2. FUNCIONES DE FORMATO Y ESTILO EN PYTHON-DOCX
# -----------------------------------------------------------------------------

def set_cell_background(cell, hex_color):
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shd)

def set_cell_margins(cell, top=140, bottom=140, left=180, right=180):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def add_callout(doc, text, title="INSIGHT CLÍNICO / TÉCNICO", border_color="0891B2", bg_color="F0F9FF"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    tbl.columns[0].width = Inches(6.5)

    cell = tbl.cell(0, 0)
    set_cell_background(cell, bg_color)
    set_cell_margins(cell, top=180, bottom=180, left=240, right=200)

    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:top w:val="none"/>'
        f'<w:left w:val="single" w:sz="36" w:space="0" w:color="{border_color}"/>'
        f'<w:bottom w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    r_title = p.add_run(f"📌 {title}\n")
    r_title.bold = True
    r_title.font.name = "Arial"
    r_title.font.size = Pt(10)
    r_title.font.color.rgb = RGBColor(8, 145, 178)

    r_text = p.add_run(text)
    r_text.font.name = "Arial"
    r_text.font.size = Pt(9.5)
    r_text.font.color.rgb = RGBColor(30, 41, 59)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

def style_heading_1(p):
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    for run in p.runs:
        run.font.name = "Arial"
        run.font.size = Pt(16)
        run.bold = True
        run.font.color.rgb = RGBColor(15, 23, 42)

def style_heading_2(p):
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    for run in p.runs:
        run.font.name = "Arial"
        run.font.size = Pt(13)
        run.bold = True
        run.font.color.rgb = RGBColor(8, 145, 178)

def style_heading_3(p):
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    for run in p.runs:
        run.font.name = "Arial"
        run.font.size = Pt(11)
        run.bold = True
        run.font.color.rgb = RGBColor(51, 65, 85)

def add_table_header(table, widths, headers, bg_color="0F172A"):
    row = table.rows[0]
    for idx, (cell, h_text) in enumerate(zip(row.cells, headers)):
        set_cell_background(cell, bg_color)
        set_cell_margins(cell, top=140, bottom=140, left=140, right=140)
        cell.width = widths[idx]
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(h_text)
        r.font.name = "Arial"
        r.font.size = Pt(9.5)
        r.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

def add_table_row(table, widths, values, is_even=False, is_bold_first=False, is_bold_all=False):
    row = table.add_row()
    bg_color = "F8FAFC" if is_even else "FFFFFF"
    for idx, (cell, val) in enumerate(zip(row.cells, values)):
        set_cell_background(cell, bg_color)
        set_cell_margins(cell, top=100, bottom=100, left=140, right=140)
        cell.width = widths[idx]
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        cell.text = ""
        p = cell.paragraphs[0]
        r = p.add_run(str(val))
        r.font.name = "Arial"
        r.font.size = Pt(9)
        if is_bold_all or (is_bold_first and idx == 0):
            r.bold = True
            r.font.color.rgb = RGBColor(15, 23, 42)
        else:
            r.font.color.rgb = RGBColor(30, 41, 59)
    return row

# -----------------------------------------------------------------------------
# 3. CONSTRUCCIÓN COMPLETA DEL INFORME WORD
# -----------------------------------------------------------------------------

def build_word_document():
    doc = Document()

    # Márgenes a 2.54 cm (1 pulgada)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

        # Encabezado
        header = section.header
        p_head = header.paragraphs[0]
        p_head.text = "RetinaScan AI • Informe de Proyecto y Metodología Diagnóstica"
        p_head.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p_head.runs[0].font.name = "Arial"
        p_head.runs[0].font.size = Pt(8.5)
        p_head.runs[0].font.color.rgb = RGBColor(148, 163, 184)

        # Pie de página
        footer = section.footer
        p_foot = footer.paragraphs[0]
        p_foot.text = "Sistema de Detección Ocular por Deep Learning • ResNet50V2 • Clasificación Automatizada"
        p_foot.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_foot.runs[0].font.name = "Arial"
        p_foot.runs[0].font.size = Pt(8)
        p_foot.runs[0].font.color.rgb = RGBColor(148, 163, 184)

    # Estilo base
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Arial'
    normal_style.font.size = Pt(10)
    normal_style.font.color.rgb = RGBColor(51, 65, 85)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(5)

    # -------------------------------------------------------------------------
    # PORTADA
    # -------------------------------------------------------------------------
    p_meta = doc.add_paragraph()
    p_meta.paragraph_format.space_before = Pt(20)
    p_meta.paragraph_format.space_after = Pt(4)
    r_meta = p_meta.add_run("INFORME TÉCNICO Y METODOLÓGICO DE INVESTIGACIÓN")
    r_meta.font.name = "Arial"
    r_meta.font.size = Pt(10)
    r_meta.font.bold = True
    r_meta.font.color.rgb = RGBColor(8, 145, 178)

    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(2)
    p_title.paragraph_format.space_after = Pt(8)
    r_title = p_title.add_run("👁️ RetinaScan AI: Detección y Clasificación Automatizada de Enfermedades Oculares con Deep Learning")
    r_title.font.name = "Arial"
    r_title.font.size = Pt(22)
    r_title.bold = True
    r_title.font.color.rgb = RGBColor(15, 23, 42)

    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_after = Pt(20)
    r_sub = p_sub.add_run(
        "Sistema integral de visión artificial y redes neuronales convolucionales profundas (ResNet50V2) "
        "para la detección asistida de Catarata, Retinopatía Diabética y Glaucoma a partir de imágenes "
        "de fondo de ojo (retinografías), con interfaz clínica interactiva en Flask."
    )
    r_sub.font.name = "Arial"
    r_sub.font.size = Pt(11.5)
    r_sub.font.color.rgb = RGBColor(71, 85, 105)

    # Ficha Técnica en Portada
    tbl_meta = doc.add_table(rows=1, cols=2)
    tbl_meta.alignment = WD_TABLE_ALIGNMENT.CENTER
    widths_meta = [Inches(2.2), Inches(4.3)]
    add_table_header(tbl_meta, widths_meta, ["Dimensión del Proyecto", "Detalle de Implementación"], bg_color="0F172A")

    meta_items = [
        ("Área de Aplicación", "Inteligencia Artificial en Oftalmología / Visión por Computadora Médica"),
        ("Arquitectura Ganadora", "ResNet50V2 con Transfer Learning bifásico (Precisión en Test: 89.01%)"),
        ("Patologías Clasificadas", "Catarata, Retinopatía Diabética, Glaucoma y Retina Normal (Sana)"),
        ("Dataset de Entrenamiento", "4,217 imágenes retinianas de Kaggle (70% Train, 15% Val, 15% Test)"),
        ("Entorno y Despliegue", "Python 3.10+, TensorFlow/Keras, Flask, Pillow, Scikit-learn"),
        ("Estado de Validación", "Validado empíricamente sobre 637 retinografías del conjunto de test")
    ]

    for i, (label, val) in enumerate(meta_items):
        add_table_row(tbl_meta, widths_meta, [label, val], is_even=(i % 2 == 1), is_bold_first=True)

    doc.add_page_break()

    # -------------------------------------------------------------------------
    # RESUMEN EJECUTIVO
    # -------------------------------------------------------------------------
    h1 = doc.add_paragraph("Resumen Ejecutivo")
    style_heading_1(h1)

    p = doc.add_paragraph(
        "La pérdida de visión debida a enfermedades oftalmológicas no diagnosticadas a tiempo representa una de las mayores crisis de salud pública global. "
        "Patologías como la catarata, la retinopatía diabética y el glaucoma encabezan las causas de deterioro visual y ceguera irreversible en el mundo. "
        "A pesar de que el examen de fondo de ojo (retinografía) permite detectar anomalías estructurales en etapas tempranas, la escasez de oftalmólogos "
        "y retinólogos especializados en centros de atención primaria retrasa significativamente el tamizaje y tratamiento oportuno."
    )

    p = doc.add_paragraph(
        "Para responder a este desafío, el proyecto RetinaScan AI implementa una solución tecnológica integral de inteligencia artificial basada en "
        "aprendizaje profundo (Deep Learning) y visión por computadora. A través de una arquitectura residual profunda (ResNet50V2) adaptada mediante un esquema "
        "bifásico de Transfer Learning y Fine-Tuning, el sistema es capaz de analizar imágenes digitales de fondo de ojo y clasificar de forma autónoma "
        "cuatro categorías clínicas críticas con una exactitud global del 89.01% en el conjunto de prueba independiente."
    )

    add_callout(
        doc,
        "RetinaScan AI no busca reemplazar el criterio del médico especialista, sino empoderar los centros de triaje y atención primaria con un motor de cribado "
        "ultrarrápido, reproducible y libre de sesgo subjetivo, garantizando que los pacientes de alto riesgo sean remitidos urgentemente al oftalmólogo.",
        title="DECLARACIÓN DE IMPACTO CLÍNICO"
    )

    # -------------------------------------------------------------------------
    # 1. DE QUÉ TRATA EL PROYECTO
    # -------------------------------------------------------------------------
    h1 = doc.add_paragraph("1. Contexto, Problema y Objeto del Proyecto")
    style_heading_1(h1)

    h2 = doc.add_paragraph("1.1. Justificación Médica y Epidemiológica")
    style_heading_2(h2)

    p = doc.add_paragraph(
        "El ojo humano es el único órgano del cuerpo que permite visualizar de manera no invasiva vasos sanguíneos y tejido nervioso central in vivo. "
        "La retinografía de fondo de ojo es el estándar clínico para la inspección de la retina, el disco óptico (cabeza del nervio óptico) y la mácula lútea. "
        "Sin embargo, la interpretación manual de retinografías presenta varios cuellos de botella fundamentales:"
    )

    bp = doc.add_paragraph(style='List Bullet')
    bp.add_run("Fatiga diagnóstica y variabilidad intra e inter-observador: ").bold = True
    bp.add_run("Dos especialistas pueden diferir en la clasificación de microaneurismas incipientes o en la excavación de la copa papilar en etapas limítrofes.")

    bp = doc.add_paragraph(style='List Bullet')
    bp.add_run("Sobrecarga de los sistemas de salud pública: ").bold = True
    bp.add_run("La diabetes mellitus y el envejecimiento poblacional han disparado la demanda de tamizajes anuales, superando la capacidad física de consulta.")

    bp = doc.add_paragraph(style='List Bullet')
    bp.add_run("Brechas de acceso geográfico: ").bold = True
    bp.add_run("Las comunidades rurales y centros periféricos carecen de retinólogos; un sistema de IA teleoftalmológico democratiza el diagnóstico temprano.")

    h2 = doc.add_paragraph("1.2. Objetivos del Proyecto")
    style_heading_2(h2)

    p = doc.add_paragraph("El proyecto se propuso cumplir con los siguientes objetivos técnicos y funcionales:")

    b1 = doc.add_paragraph(style='List Bullet')
    b1.add_run("Objetivo General: ").bold = True
    b1.add_run("Desarrollar y validar un pipeline computacional de inteligencia artificial para la clasificación automatizada de retinografías en cuatro clases diagnósticas (Catarata, Retinopatía Diabética, Glaucoma y Retina Normal).")

    b2 = doc.add_paragraph(style='List Bullet')
    b2.add_run("Objetivo Específico 1: ").bold = True
    b2.add_run("Construir un dataset estratificado y balanceado a partir del repositorio de Kaggle, aplicando aumentación sintética de datos (Data Augmentation) robusta ante variaciones de iluminación y enfoque.")

    b3 = doc.add_paragraph(style='List Bullet')
    b3.add_run("Objetivo Específico 2: ").bold = True
    b3.add_run("Evaluar y comparar empíricamente tres arquitecturas convolucionales de vanguardia (ResNet50V2, MobileNetV2 y EfficientNetB0), seleccionando la que exhiba mayor agudeza en microlesiones vasculares y nerviosas.")

    b4 = doc.add_paragraph(style='List Bullet')
    b4.add_run("Objetivo Específico 3: ").bold = True
    b4.add_run("Implementar una aplicación web interactiva en Flask con un diseño clínico de alta gama, que permita cargar imágenes por drag-and-drop, ejecutar inferencia en milisegundos y emitir informes médicos exportables.")

    # Muestras de retina
    fig_samples = generate_fundus_samples_grid()
    doc.add_paragraph().paragraph_format.space_before = Pt(6)
    doc.add_picture(fig_samples, width=Inches(6.4))
    p_cap = doc.add_paragraph("Figura 1: Muestras representativas de las cuatro categorías clínicas analizadas en el dataset.")
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.runs[0].font.size = Pt(8.5)
    p_cap.runs[0].italic = True

    # -------------------------------------------------------------------------
    # 2. TECNOLOGÍAS UTILIZADAS
    # -------------------------------------------------------------------------
    h1 = doc.add_paragraph("2. Stack Tecnológico y Arquitectura de Software")
    style_heading_1(h1)

    p = doc.add_paragraph(
        "El proyecto fue edificado bajo una arquitectura modular y desacoplada que garantiza reproducibilidad científica, "
        "facilidad de mantenimiento y compatibilidad con entornos de producción en la nube."
    )

    tbl_tech = doc.add_table(rows=1, cols=3)
    tbl_tech.alignment = WD_TABLE_ALIGNMENT.CENTER
    widths_tech = [Inches(1.8), Inches(1.8), Inches(2.9)]
    add_table_header(tbl_tech, widths_tech, ["Capa / Dimensión", "Tecnología / Librería", "Función y Responsabilidad"], bg_color="0F172A")

    tech_data = [
        ("Lenguaje Principal", "Python 3.10+", "Núcleo de procesamiento algorítmico, scripting de entrenamiento e inferencia en servidor."),
        ("Framework Deep Learning", "TensorFlow 2.15+ & Keras", "Definición del modelo, carga de capas preentrenadas ImageNet, compilación y optimizadores."),
        ("Procesamiento de Imágenes", "Pillow (PIL) 10.0+", "Ingesta de bytes de imagen, conversión de perfiles de color (RGB) y redimensionamiento bilineal."),
        ("Cálculo Matricial", "NumPy 1.24+", "Normalización matemática de tensores [0, 1], manipulación dimensional y operaciones vectorizadas."),
        ("Métricas y Evaluación", "Scikit-Learn 1.3+", "Generación de matriz de confusión, reporte de clasificación multiclase (Precision, Recall, F1)."),
        ("Visualización Científica", "Matplotlib 3.8+ & Seaborn", "Renderizado de curvas de convergencia (Loss/Accuracy) y mapas de calor estadísticos."),
        ("Servidor y Backend", "Flask 3.0+", "Microframework web para servir la interfaz de usuario y exponer endpoints RESTful (/predict, /api/stats)."),
        ("Frontend & UI Clínica", "HTML5, CSS3, JS (Vanilla)", "Panel de control futurista 'Dark Mode', animaciones de escáner médico, barras de probabilidad interactivas."),
        ("Despliegue e Infraestructura", "Hugging Face Spaces / Docker", "Plataforma PaaS para hosting continuo con inferencia acelerada en la nube.")
    ]

    for idx, (capa, tec, func) in enumerate(tech_data):
        add_table_row(tbl_tech, widths_tech, [capa, tec, func], is_even=(idx % 2 == 1), is_bold_first=True)

    doc.add_paragraph().paragraph_format.space_before = Pt(8)

    # -------------------------------------------------------------------------
    # 3. DATASET Y ESTRATEGIA DE PARTICIÓN
    # -------------------------------------------------------------------------
    h1 = doc.add_paragraph("3. Conjunto de Datos y Estrategia de Partición Estratificada")
    style_heading_1(h1)

    p = doc.add_paragraph(
        "El sistema fue alimentado con un dataset público de imágenes retinianas disponible en Kaggle, compuesto por un total de "
        "4,217 imágenes clínicas de fondo de ojo en formato RGB de alta resolución."
    )

    h2 = doc.add_paragraph("3.1. Partición de Datos (Train, Validation, Test)")
    style_heading_2(h2)

    p = doc.add_paragraph(
        "Para evitar fugas de información (data leakage) y asegurar una evaluación rigurosa e imparcial, se desarrolló el script "
        "split_dataset.py, el cual aplica una partición estrictamente estratificada con semilla fija (seed=42):"
    )

    tbl_ds = doc.add_table(rows=1, cols=5)
    tbl_ds.alignment = WD_TABLE_ALIGNMENT.CENTER
    widths_ds = [Inches(1.8), Inches(1.1), Inches(1.1), Inches(1.1), Inches(1.4)]
    add_table_header(tbl_ds, widths_ds, ["Clase Patológica", "Train (70%)", "Val (15%)", "Test (15%)", "Total Imágenes"], bg_color="0891B2")

    ds_rows = [
        ("Cataract (Catarata)", "726", "155", "157", "1,038"),
        ("Diabetic Retinopathy", "768", "164", "166", "1,098"),
        ("Glaucoma", "704", "151", "152", "1,007"),
        ("Normal (Sana)", "751", "161", "162", "1,074"),
        ("TOTAL MUESTRAS", "2,949", "631", "637", "4,217")
    ]

    for idx, row_vals in enumerate(ds_rows):
        is_last = (idx == len(ds_rows) - 1)
        add_table_row(tbl_ds, widths_ds, row_vals, is_even=(idx % 2 == 1), is_bold_first=True, is_bold_all=is_last)

    # Gráfico de partición
    fig_ds = generate_dataset_chart()
    doc.add_paragraph().paragraph_format.space_before = Pt(8)
    doc.add_picture(fig_ds, width=Inches(6.2))
    p_cap = doc.add_paragraph("Figura 2: Distribución de volúmenes de imágenes por clase en los subconjuntos de datos.")
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.runs[0].font.size = Pt(8.5)
    p_cap.runs[0].italic = True

    h2 = doc.add_paragraph("3.2. Aumento Sintético de Datos (Data Augmentation)")
    style_heading_2(h2)

    p = doc.add_paragraph(
        "Las condiciones reales de adquisición oftalmológica sufren variaciones de luz, pupilas irregularmente dilatadas y movimientos leves del paciente. "
        "En train_resnet.py se diseñó un pipeline de Data Augmentation mediante ImageDataGenerator que genera transformaciones controladas en cada época:"
    )

    aug_items = [
        ("Reescalado de Intensidad (rescale=1./255)", "Normaliza los valores de luminancia de [0, 255] a números flotantes en [0.0, 1.0]."),
        ("Rotación Aleatoria (rotation_range=15°)", "Simula leves inclinaciones de la cabeza del paciente respecto al reposabarbillas del retinógrafo."),
        ("Zoom Aleatorio (zoom_range=0.2)", "Compensa discrepancias en la distancia focal y óptica de diferentes modelos de cámaras retinales."),
        ("Volteo Horizontal (horizontal_flip=True)", "Enseña al modelo a reconocer patologías indistintamente si la toma corresponde al ojo izquierdo (OS) u ojo derecho (OD)."),
        ("Variación de Brillo (brightness_range=[0.8, 1.2])", "Emula cambios en la intensidad del flash o estados variables de midriasis (dilatación pupilar)."),
        ("Desplazamiento Espacial (width/height_shift=0.1)", "Previene que el modelo asuma que el disco óptico o la fóvea siempre están en el centro exacto.")
    ]

    for param, desc in aug_items:
        bp = doc.add_paragraph(style='List Bullet')
        bp.add_run(f"{param}: ").bold = True
        bp.add_run(desc)

    # -------------------------------------------------------------------------
    # 4. ALGORITMOS, MODELOS Y ENTRENAMIENTO
    # -------------------------------------------------------------------------
    h1 = doc.add_paragraph("4. Algoritmos, Modelos de IA y Comparativa Experimental")
    style_heading_1(h1)

    h2 = doc.add_paragraph("4.1. Benchmark Comparativo de Arquitecturas")
    style_heading_2(h2)

    p = doc.add_paragraph(
        "Para determinar la mejor arquitectura para el diagnóstico retiniano, se implementó un estudio comparativo (Benchmark) "
        "entre tres familias de redes neuronales convolucionales profundas bajo idénticas condiciones de preprocesamiento:"
    )

    tbl_bm = doc.add_table(rows=1, cols=4)
    tbl_bm.alignment = WD_TABLE_ALIGNMENT.CENTER
    widths_bm = [Inches(1.8), Inches(1.8), Inches(1.1), Inches(1.8)]
    add_table_header(tbl_bm, widths_bm, ["Modelo Evaluado", "Paradigma Arquitectónico", "Precisión Test", "Diagnóstico Experimental"], bg_color="0F172A")

    bm_data = [
        ("ResNet50V2 (Seleccionado)", "Conexiones Residuales (Skip Connections)", "89.01%", "Excelente convergencia. Las conexiones residuales preservan microlesiones vasculares."),
        ("MobileNetV2", "Convolución Separable en Profundidad", "81.20%", "Altamente eficiente pero con pérdida de sensibilidad en daños incipientes de fibras nerviosas."),
        ("EfficientNetB0", "Escalado Compuesto (Compound Scaling)", "25.43%", "Incompatibilidad con el pipeline de normalización; divergió durante la fase de ajuste fino.")
    ]

    for idx, (m, p_, a, d) in enumerate(bm_data):
        add_table_row(tbl_bm, widths_bm, [m, p_, a, d], is_even=(idx % 2 == 1), is_bold_first=True)

    fig_bm = generate_benchmark_chart()
    doc.add_paragraph().paragraph_format.space_before = Pt(8)
    doc.add_picture(fig_bm, width=Inches(5.8))
    p_cap = doc.add_paragraph("Figura 3: Desempeño comparativo de precisión en el conjunto de prueba independiente.")
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.runs[0].font.size = Pt(8.5)
    p_cap.runs[0].italic = True

    h2 = doc.add_paragraph("4.2. Fundamentos Algorítmicos de ResNet50V2 y Skip Connections")
    style_heading_2(h2)

    p = doc.add_paragraph(
        "A medida que las redes neuronales convolucionales aumentan en profundidad, sufren el fenómeno de degradación y desvanecimiento del gradiente "
        "(vanishing gradient), donde las derivadas hacia atrás se anulan y las capas tempranas dejan de aprender. "
        "ResNet50V2 resuelve este problema mediante conexiones de salto o residuales (Skip Connections):"
    )

    add_callout(
        doc,
        "En lugar de obligar a las capas convolucionales a aprender un mapeo directo H(x), el bloque residual aprende una función residual F(x) = H(x) - x, "
        "reformulando la salida como: y = F(x, {Wi}) + x. "
        "Esto permite que la información fluya sin obstáculos a través de la identidad x, permitiendo que la red conserve tanto características de bajo nivel "
        "(bordes vasculares, nitidez del disco óptico) como semánticas de alto nivel (exudados, hemorragias).",
        title="PRINCIPIO MATEMÁTICO: BLOQUES RESIDUALES IDENTITY MAPPING"
    )

    h2 = doc.add_paragraph("4.3. Topología de la Cabeza Clasificadora")
    style_heading_2(h2)

    p = doc.add_paragraph(
        "En models/model_builder.py se definió una cabeza clasificadora personalizada y altamente regularizada acoplada al backbone ResNet50V2:"
    )

    layers_info = [
        ("Entrada de Tensor: Input(224, 224, 3)", "Recibe la matriz de imagen tridimensional normalizada."),
        ("Backbone ResNet50V2 (ImageNet)", "50 capas convolucionales profundas que extraen 2,048 mapas de características."),
        ("GlobalAveragePooling2D", "Promedia espacialmente cada mapa de características, reduciendo la dimensión sin requerir aplanamiento masivo (Flatten)."),
        ("BatchNormalization", "Normaliza las activaciones intermedias acelerando la convergencia y reduciendo la covarianza interna."),
        ("Dense(512, activation='relu') + Dropout(0.5)", "Capa completamente conectada de 512 neuronas con 50% de desconexión aleatoria para evitar sobreajuste."),
        ("Dense(256, activation='relu') + Dropout(0.3)", "Segunda capa de compresión semántica con 30% de regularización dropout."),
        ("Dense(4, activation='softmax')", "Capa final con 4 neuronas que emite la distribución de probabilidad multiclase.")
    ]

    for lay, desc in layers_info:
        bp = doc.add_paragraph(style='List Bullet')
        bp.add_run(f"{lay}: ").bold = True
        bp.add_run(desc)

    h2 = doc.add_paragraph("4.4. Estrategia de Transfer Learning Bifásica y Callbacks")
    style_heading_2(h2)

    p = doc.add_paragraph(
        "El entrenamiento se ejecutó en dos fases secuenciales optimizadas para maximizar la generalización:"
    )

    bp = doc.add_paragraph(style='List Bullet')
    bp.add_run("Fase 1 - Calentamiento de la Cabeza (15 épocas): ").bold = True
    bp.add_run("La base convolucional ResNet50V2 se mantiene congelada (trainable = False). Se entrena únicamente la cabeza densa con el optimizador Adam y una tasa de aprendizaje estándar de 0.001 (1e-3).")

    bp = doc.add_paragraph(style='List Bullet')
    bp.add_run("Fase 2 - Fine-Tuning Fino (30 épocas): ").bold = True
    bp.add_run("Se descongelan selectivamente las últimas 40 capas de la base convolucional, permitiendo ajustar los filtros de alto nivel a la textura biológica de la retina. Se reduce la tasa de aprendizaje a 0.0001 (1e-4) para evitar la alteración destructiva de los pesos preentrenados.")

    p = doc.add_paragraph("Durante ambas fases operaron tres mecanismos de control de entrenamiento (Callbacks):")

    cb_items = [
        ("EarlyStopping", "Monitorea 'val_accuracy' con paciencia de 5 épocas. Detiene el proceso si no hay mejoras y restaura automáticamente los mejores pesos."),
        ("ModelCheckpoint", "Serializa en disco únicamente los pesos óptimos alcanzados en 'models/best_resnet50.h5'."),
        ("ReduceLROnPlateau", "Reduce el learning rate en un factor de 0.3 si 'val_loss' no disminuye durante 3 épocas consecutivas, permitiendo descensos más finos en el paisaje de pérdida.")
    ]

    for cb, desc in cb_items:
        bp = doc.add_paragraph(style='List Bullet')
        bp.add_run(f"{cb}: ").bold = True
        bp.add_run(desc)

    # -------------------------------------------------------------------------
    # 5. EL PROCESO QUE HACE PARA DETERMINAR QUÉ ENFERMEDAD ES
    # -------------------------------------------------------------------------
    h1 = doc.add_paragraph("5. Pipeline Integral de Inferencia: Proceso Paso a Paso para Determinar la Enfermedad")
    style_heading_1(h1)

    p = doc.add_paragraph(
        "El proceso computacional y analítico que realiza RetinaScan AI desde el momento en que una fotografía retiniana es ingresada "
        "hasta que se entrega el dictamen médico detallado se compone de seis fases rigurosamente orquestadas:"
    )

    fig_pipe = generate_pipeline_diagram()
    doc.add_picture(fig_pipe, width=Inches(6.5))
    p_cap = doc.add_paragraph("Figura 4: Flujo analítico secuencial del motor de inferencia de RetinaScan AI.")
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.runs[0].font.size = Pt(8.5)
    p_cap.runs[0].italic = True

    steps = [
        (
            "Paso 1: Ingesta y Recepción de la Imagen Retiniana",
            "El usuario arrastra un archivo de imagen (JPG, PNG) al panel web interactivo o presiona uno de los botones de prueba rápida del conjunto de test. "
            "El navegador lee el archivo en local mediante FileReader de JavaScript para mostrar una previsualización inmediata con animación de escáner médico, "
            "y despacha una petición asíncrona HTTP POST multipart/form-data al endpoint '/predict' del servidor Flask."
        ),
        (
            "Paso 2: Decodificación y Normalización Espacial (Pillow)",
            "En el servidor (utils.py -> preprocess_image), la imagen se procesa con la librería Pillow:\n"
            "• Conversión forzada a RGB: Se descarta cualquier canal alfa (transparencia) o escala de grises de 1 solo canal, garantizando una matriz tridimensional canónica.\n"
            "• Interpolación Bilineal a 224 × 224 píxeles: La retina se redimensiona a la resolución exacta esperada por la arquitectura ResNet50V2 sin distorsionar la geometría anatómica."
        ),
        (
            "Paso 3: Transformación Matricial y Normalización de Intensidad (NumPy)",
            "La imagen se convierte en una estructura matricial de NumPy con tipo de dato flotante de 32 bits (np.float32).\n"
            "• Escalado Numérico [0.0, 1.0]: Cada pixel [0, 255] es dividido entre 255.0. Esta normalización es crítica para evitar gradientes explosivos y situar los valores en la región de máxima sensibilidad de las activaciones convolucionales.\n"
            "• Expansión de Eje de Lote (Batch Dimension): Mediante np.expand_dims, el arreglo cambia de forma (224, 224, 3) a un tensor cuadridimensional (1, 224, 224, 3), satisfaciendo el estándar de lotes de TensorFlow."
        ),
        (
            "Paso 4: Inferencia Neuronal hacia Adelante (Forward Pass con ResNet50V2)",
            "El modelo previamente cacheado en memoria RAM (para evitar latencias de recarga) recibe el tensor. "
            "A través de las 50 capas residuales, se analizan características jerárquicas:\n"
            "• Capas 1-15: Extraen gradientes de borde, contraste retiniano y curvatura del globo ocular.\n"
            "• Capas 16-35: Identifican la red microvascular, bifurcaciones de arteriolas y vénulas, y la morfología del disco óptico.\n"
            "• Capas 36-50: Detectan biomarcadores patológicos específicos: velamiento u opacidad (catarata), microaneurismas y exudados algodonosos (retinopatía diabética), o excavación papilar con pérdida del anillo neurorretiniano (glaucoma).\n"
            "La cabeza clasificadora colapsa estas características en un vector de logits z = [z_0, z_1, z_2, z_3]."
        ),
        (
            "Paso 5: Cálculo de Probabilidad Softmax y Selección Argmax",
            "La capa de salida aplica la función matemática Softmax:\n"
            "P(y = i | x) = exp(z_i) / Σ exp(z_j), para i ∈ {0, 1, 2, 3}\n"
            "Esto genera una distribución normalizada donde la suma de probabilidades es exactamente 100%.\n"
            "El algoritmo evalúa el argumento máximo:\n"
            "i* = argmax(P_i)\n"
            "El índice ganador i* se vincula a la lista de clases: 0 -> cataract, 1 -> diabetic_retinopathy, 2 -> glaucoma, 3 -> normal. "
            "La confianza oficial se computa como: Confianza = P(i*) × 100."
        ),
        (
            "Paso 6: Enriquecimiento Semántico Clínico y Despacho JSON",
            "El motor de inferencia consulta el diccionario de conocimiento médico DISEASE_INFO en utils.py utilizando la clave ganadora. "
            "Se inyecta la ficha técnica completa:\n"
            "• Nivel de severidad clínica (e.g. 'Alta / Crítica' para retinopatía o 'Muy Alta / Silenciosa' para glaucoma).\n"
            "• Descripción fisiopatológica formal en español.\n"
            "• Lista de 4 signos y síntomas característicos experimentados por pacientes.\n"
            "• Guía de 4 recomendaciones clínicas y exámenes paraclínicos urgentes (OCT, campimetría, láser).\n"
            "• Vector desglosado con el porcentaje de probabilidad para cada una de las cuatro clases.\n"
            "El servidor devuelve una respuesta HTTP 200 en formato JSON, y el JavaScript del cliente anima los medidores de confianza, las barras multiclase y habilita la generación del reporte clínico imprimible."
        )
    ]

    for title, desc in steps:
        h2 = doc.add_paragraph(title)
        style_heading_2(h2)
        doc.add_paragraph(desc)

    # -------------------------------------------------------------------------
    # 6. BASE DE CONOCIMIENTO CLÍNICO DE LAS PATOLOGÍAS
    # -------------------------------------------------------------------------
    h1 = doc.add_paragraph("6. Base de Conocimiento Médico y Fisiopatología de las Patologías")
    style_heading_1(h1)

    p = doc.add_paragraph(
        "Para que la herramienta sea verdaderamente útil en entornos hospitalarios y de triaje, RetinaScan AI incorpora "
        "un motor de conocimiento médico estructurado que acompaña la predicción numérica con información clínica contextualizada."
    )

    diseases = [
        (
            "Catarata (Cataract)",
            "Moderada a Severa",
            "Opacificación progresiva del cristalino natural del ojo que impide el paso libre y enfocado de los rayos de luz hacia la retina.",
            "En la retinografía, la catarata produce un velamiento generalizado o pérdida del reflejo rojo, disminución del contraste en toda la imagen y pérdida de nitidez en los detalles de la vasculatura fina.",
            [
                "Visión nublada, borrosa o tenue tipo 'vidrio esmerilado'.",
                "Dificultad marcada para la visión nocturna y conducción de noche.",
                "Fotofobia y deslumbramiento severo con halos alrededor de las luces.",
                "Percepción de colores desvanecidos o amarillentos."
            ],
            [
                "Examen oftalmológico completo con lámpara de hendidura.",
                "Cirugía de extracción con implante de lente intraocular (LIO) si deteriora actividades cotidianas.",
                "Uso preventivo de gafas con filtro de protección UV400.",
                "Suspender conducción nocturna hasta resolución quirúrgica."
            ],
            "E67E22"
        ),
        (
            "Retinopatía Diabética (Diabetic Retinopathy)",
            "Alta / Crítica",
            "Complicación microvascular secundaria a la hiperglucemia crónica en pacientes con diabetes mellitus, causando daño a las células endoteliales de los capilares retinianos.",
            "En la retinografía, se evidencia presencia de microaneurismas (puntos rojos diminutos), hemorragias intrarretinianas 'en llama', exudados duros lipídicos amarillentos, manchas algodonosas por isquemia y neovascularización anómala.",
            [
                "Manchas flotantes oscuras o filamentos en el campo visual ('moscas volantes').",
                "Visión fluctuante a lo largo del día según niveles de glucosa.",
                "Escotomas o zonas de ceguera en el centro de la mirada.",
                "Dificultad para discernir contrastes y gama de colores."
            ],
            [
                "Control glucémico estricto (mantener hemoglobina glicosilada HbA1c < 7.0%).",
                "Fondo de ojo bajo dilatación pupilar farmacológica y angiografía fluoresceínica urgente.",
                "Interconsulta prioritaria con retinólogo para fotocoagulación láser o inyecciones anti-VEGF.",
                "Seguimiento estrecho cada 3 a 6 meses según grado de proliferación."
            ],
            "E74C3C"
        ),
        (
            "Glaucoma (Neuropatía Óptica)",
            "Muy Alta / Silenciosa",
            "Neuropatía óptica degenerativa progresiva asociada habitualmente a hipertensión ocular, caracterizada por la pérdida irreversible de células ganglionares y axones del nervio óptico.",
            "En la retinografía, el signo cardinal es el aumento anormal del cociente copa/disco óptico (excavación papilar > 0.5), asimetría en el anillo neurorretiniano y adelgazamiento de la capa de fibras nerviosas de la retina (CFNR).",
            [
                "Pérdida insidiosa, lenta e indolora de la visión periférica o lateral ('visión en túnel').",
                "Puntos ciegos dispersos que el cerebro compensa hasta fases muy avanzadas.",
                "En crisis agudas de ángulo cerrado: dolor ocular punzante intenso, náuseas y halos multicolores.",
                "Ceguera total si no se reduce la presión intraocular a tiempo."
            ],
            [
                "Tonometría urgente para medir la presión intraocular (PIO) y gonioscopia de ángulo iridocorneal.",
                "Campimetría visual computarizada (Humphrey) y Tomografía de Coherencia Óptica (OCT) de nervio.",
                "Adherencia estricta a colirios hipotensores tópicos diarios (análogos de prostaglandinas/beta-bloqueantes).",
                "Evaluación de trabeculoplastia láser selectiva (SLT) o cirugía filtrante si la PIO es refractaria."
            ],
            "9B59B6"
        ),
        (
            "Retina Normal (Sana)",
            "Saludable",
            "Fondo de ojo con anatomía retiniana y neurovascular íntegra, sin signos de inflamación, degeneración ni isquemia.",
            "El disco óptico exhibe bordes netos y coloración rosada saludable. La relación copa/disco se sitúa en rangos fisiológicos (< 0.3). La mácula y el centro foveolar están libres de edema o exudados, y la vasculatura mantiene un calibre arterial/venoso de 2:3.",
            [
                "Agudeza visual normal conservada (20/20 corregida).",
                "Campo visual íntegro sin escotomas.",
                "Visión cromática perfecta.",
                "Ausencia de destellos luminosos (fotopsias) o distorsiones de líneas (metamorfopsias)."
            ],
            [
                "Exámenes oftalmológicos preventivos regulares cada 1 a 2 años.",
                "Higiene visual: aplicar la regla 20-20-20 al trabajar con pantallas digitales.",
                "Alimentación balanceada rica en antioxidantes oculares (luteína, zeaxantina y omega-3).",
                "Protección con gafas de sol certificadas contra radiación UV."
            ],
            "2ECC71"
        )
    ]

    for idx_d, (name, sev, fisiopat, signos_rf, sint, recs, col) in enumerate(diseases):
        h2 = doc.add_paragraph(f"6.{idx_d + 1}. {name}")
        style_heading_2(h2)

        add_callout(
            doc,
            f"Severidad Clínica: {sev}\n\nFisiopatología:\n{fisiopat}\n\nSignos en Retinografía de Fondo de Ojo:\n{signos_rf}",
            title=f"FICHA DIAGNÓSTICA: {name.upper()}",
            border_color=col,
            bg_color="F8FAFC"
        )

        p_s = doc.add_paragraph()
        r_s = p_s.add_run("Signos y Síntomas Principales:")
        r_s.bold = True
        r_s.font.color.rgb = RGBColor(15, 23, 42)
        for s in sint:
            bp = doc.add_paragraph(style='List Bullet')
            bp.add_run(s)

        p_r = doc.add_paragraph()
        r_r = p_r.add_run("Protocolo de Recomendaciones Clínicas:")
        r_r.bold = True
        r_r.font.color.rgb = RGBColor(15, 23, 42)
        for r in recs:
            bp = doc.add_paragraph(style='List Bullet')
            bp.add_run(r)

    # -------------------------------------------------------------------------
    # 7. PLATAFORMA WEB CLÍNICA Y EXPERIENCIA DE USUARIO
    # -------------------------------------------------------------------------
    h1 = doc.add_paragraph("7. Plataforma Web Clínica y Experiencia de Usuario")
    style_heading_1(h1)

    p = doc.add_paragraph(
        "Para transformar el modelo de aprendizaje profundo en una herramienta práctica y usable en el día a día médico, "
        "se desarrolló una aplicación web con una interfaz de usuario de grado clínico (templates/index.html y app.py)."
    )

    h2 = doc.add_paragraph("7.1. Características de la Interfaz de Usuario")
    style_heading_2(h2)

    ui_feats = [
        ("Estética Futurista 'Clinical Dark Mode'", "Diseñada con una paleta cromática en tonos azul marino oscuro (#0B0F19, #111827) y detalles en cian (#22D3EE), reduciendo la fatiga visual del personal de salud durante turnos prolongados."),
        ("Zona de Carga Multifunción (Dropzone)", "Admite carga por arrastre de archivos locales o selección de muestras predefinidas del conjunto de test para pruebas inmediatas sin necesidad de contar con retinografías externas."),
        ("Efecto de Escáner Retiniano Láser", "Durante la carga y procesamiento, un barrido visual animado emula el escaneo de un tomógrafo oftalmológico de coherencia óptica."),
        ("Medidor Dinámico de Confianza", "Muestra el porcentaje exacto de certidumbre del modelo junto con un distintivo visual codificado por colores según la severidad."),
        ("Desglose Multiclase Probabilístico", "Cuatro barras horizontales animadas permiten al oftalmólogo evaluar diagnósticos diferenciales y márgenes de incertidumbre."),
        ("Generador de Informes Imprimibles", "Un botón integrado reformatea dinámicamente los hallazgos en un informe clínico con fecha, paciente y dictamen, optimizado para impresión o guardado en PDF.")
    ]

    for title, desc in ui_feats:
        bp = doc.add_paragraph(style='List Bullet')
        bp.add_run(f"{title}: ").bold = True
        bp.add_run(desc)

    h2 = doc.add_paragraph("7.2. Endpoints de la API REST (app.py)")
    style_heading_2(h2)

    tbl_api = doc.add_table(rows=1, cols=4)
    tbl_api.alignment = WD_TABLE_ALIGNMENT.CENTER
    widths_api = [Inches(1.2), Inches(1.5), Inches(1.5), Inches(2.3)]
    add_table_header(tbl_api, widths_api, ["Método HTTP", "Endpoint URL", "Parámetros", "Respuesta / Función"], bg_color="0F172A")

    api_data = [
        ("GET", "/", "Ninguno", "Renderiza la página web principal con el dashboard clínico."),
        ("POST", "/predict", "file (multipart) o sample_class (form)", "Ejecuta la inferencia sobre la imagen y devuelve el JSON completo con probabilidades y ficha clínica."),
        ("GET", "/sample/<category>", "category (cataract, glaucoma...)", "Devuelve el archivo binario de imagen de muestra del directorio de test."),
        ("GET", "/api/stats", "Ninguno", "Devuelve estadísticas globales del dataset y benchmark comparativo de los modelos.")
    ]

    for idx, (m, ep, pr, rs) in enumerate(api_data):
        add_table_row(tbl_api, widths_api, [m, ep, pr, rs], is_even=(idx % 2 == 1), is_bold_first=True)

    doc.add_paragraph().paragraph_format.space_before = Pt(8)

    # -------------------------------------------------------------------------
    # 8. ESTRUCTURA DEL PROYECTO Y ARCHIVOS FUENTE
    # -------------------------------------------------------------------------
    h1 = doc.add_paragraph("8. Estructura del Código y Organización de Archivos")
    style_heading_1(h1)

    p = doc.add_paragraph("El repositorio mantiene una estructura limpia y orientada a buenas prácticas de ingeniería de software:")

    files_struct = [
        ("app.py", "Servidor principal Flask. Gestiona las rutas HTTP, recepción de imágenes y serialización de respuestas."),
        ("utils.py", "Módulo de utilidad de inferencia. Contiene preprocess_image, load_my_model, predict_retinal_image y el diccionario médico DISEASE_INFO."),
        ("models/model_builder.py", "Constructor modular de la red neuronal ResNet50V2 y definición de las dos fases de compilación (cabeza y fine-tuning)."),
        ("models/best_resnet50.h5", "Pesos serializados del modelo óptimo entrenado con 89.01% de precisión en test."),
        ("train_resnet.py", "Script de entrenamiento completo con generadores de Data Augmentation, callbacks, cálculo de curvas y matriz de confusión."),
        ("split_dataset.py", "Script para la partición estratificada reproducible del dataset (70% Train, 15% Val, 15% Test) con semilla 42."),
        ("test_model.py", "Script de verificación rápida por consola que evalúa una muestra de cada clase e imprime las predicciones."),
        ("templates/index.html", "Plantilla única y autosuficiente que combina el maquetado semántico, estilos CSS con glassmorphism y lógica JavaScript asíncrona."),
        ("requirements.txt", "Especificación de versiones de dependencias de Python para instalación limpia y despliegue sin conflictos.")
    ]

    for fn, desc in files_struct:
        bp = doc.add_paragraph(style='List Bullet')
        bp.add_run(f"{fn}: ").bold = True
        bp.add_run(desc)

    # -------------------------------------------------------------------------
    # 9. RESULTADOS, CONCLUSIONES Y TRABAJO FUTURO
    # -------------------------------------------------------------------------
    h1 = doc.add_paragraph("9. Resultados, Discusión, Conclusiones y Trabajo Futuro")
    style_heading_1(h1)

    h2 = doc.add_paragraph("9.1. Discusión de Resultados y Desempeño")
    style_heading_2(h2)

    p = doc.add_paragraph(
        "La evaluación cuantitativa sobre las 637 retinografías del conjunto de prueba independiente demostró una exactitud global del 89.01%. "
        "Este resultado sitúa a RetinaScan AI en un nivel de competitividad comparable al de herramientas clínicas de tamizaje primario:"
    )

    bp = doc.add_paragraph(style='List Bullet')
    bp.add_run("Sensibilidad en Cataratas: ").bold = True
    bp.add_run("El modelo identificó con altísima precisión los casos de catarata gracias a la detección de la pérdida homogénea de contraste y velamiento de los medios transparentes.")

    bp = doc.add_paragraph(style='List Bullet')
    bp.add_run("Detección de Retinopatía Diabética: ").bold = True
    bp.add_run("Las conexiones residuales de ResNet50V2 demostraron ser capaces de detectar tanto microaneurismas discretos como hemorragias extensas, evitando falsos negativos en pacientes metabólicamente descompensados.")

    bp = doc.add_paragraph(style='List Bullet')
    bp.add_run("Diferenciación en Glaucoma: ").bold = True
    bp.add_run("El rasgo anatómico de la relación copa/disco óptico fue extraído eficientemente por las capas medias y profundas de la red, distinguiendo excavaciones patológicas de papilas fisiológicamente normales.")

    h2 = doc.add_paragraph("9.2. Conclusiones Principales")
    style_heading_2(h2)

    concl_points = [
        "Viabilidad del Aprendizaje Profundo: Se demostró que las redes neuronales convolucionales con aprendizaje residual son idóneas para el análisis de fondo de ojo, superando arquitecturas livianas que sacrifican resolución de microlesiones.",
        "Eficacia del Esquema Bifásico: La combinación de entrenamiento inicial de la cabeza clasificadora seguido de un ajuste fino de 40 capas de la base permitió adaptar exitosamente filtros generales de ImageNet a la morfología retiniana sin sobreajuste.",
        "Integración Clínica Efectiva: El diseño de una interfaz web con explicación médica probabilística humaniza los resultados de la IA, transformando predicciones matemáticas en información clínica de valor.",
        "Reproducibilidad Total: El pipeline completo, desde la partición estratificada con semilla 42 hasta los scripts de inferencia modular, garantiza que cualquier investigador pueda replicar los resultados."
    ]

    for c in concl_points:
        bp = doc.add_paragraph(style='List Bullet')
        bp.add_run(c)

    h2 = doc.add_paragraph("9.3. Recomendaciones y Líneas de Trabajo Futuro")
    style_heading_2(h2)

    future_points = [
        ("Incorporación de Mapas de Activación Grad-CAM", "Implementar mapas de calor visuales superpuestos en la retinografía para que el médico observe exactamente qué región del disco óptico o de la mácula motivó la decisión de la red (Explicabilidad / XAI)."),
        ("Soporte para Imágenes No Midriáticas y Baja Calidad", "Entrenar un clasificador previo de calidad de imagen que rechace tomas desenfocadas o con artefactos antes de remitirlas a la red de diagnóstico."),
        ("Subclasificación por Estadios de Severidad", "Evolucionar la detección de Retinopatía Diabética a la escala internacional ICDR (Leve, Moderada, Severa, Proliferativa) y estadificación del daño glaucomatoso."),
        ("Interoperabilidad DICOM / HL7", "Integrar compatibilidad con estándares hospitalarios de comunicación médica para almacenar los informes directamente en la Historia Clínica Electrónica del paciente.")
    ]

    for title, desc in future_points:
        bp = doc.add_paragraph(style='List Bullet')
        bp.add_run(f"{title}: ").bold = True
        bp.add_run(desc)

    # -------------------------------------------------------------------------
    # 10. REFERENCIAS BIBLIOGRÁFICAS
    # -------------------------------------------------------------------------
    h1 = doc.add_paragraph("10. Referencias Bibliográficas")
    style_heading_1(h1)

    refs = [
        "He, K., Zhang, X., Ren, S., & Sun, J. (2016). Identity mappings in deep residual networks. European Conference on Computer Vision (ECCV), pp. 630-645.",
        "Gulshan, V., et al. (2016). Development and validation of a deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs. JAMA, 316(22), 2402-2410.",
        "Ting, D. S. W., et al. (2017). Development and validation of a deep learning system for diabetic retinopathy and related eye diseases using retinal images from multiethnic populations with diabetes. JAMA, 318(22), 2211-2223.",
        "Sandler, M., et al. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4510-4520.",
        "Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. International Conference on Machine Learning (ICML), pp. 6105-6114.",
        "World Health Organization (WHO). (2019). World report on vision. Geneva: World Health Organization."
    ]

    for ref in refs:
        p_ref = doc.add_paragraph(style='List Bullet')
        p_ref.add_run(ref).font.size = Pt(8.5)

    # Guardar documento
    output_docx = os.path.join(BASE_DIR, "Informe_Tecnico_RetinaScan_AI.docx")
    doc.save(output_docx)
    print(f"\n[OK] Documento generado exitosamente en: {output_docx}")
    return output_docx

if __name__ == "__main__":
    print("Iniciando generación de gráficos e informe en Word...")
    path = build_word_document()
    print(f"Completado exitosamente: {path}")
