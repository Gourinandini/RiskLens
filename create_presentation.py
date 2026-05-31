"""
Python script to generate a premium multi-page PDF presentation of the
RiskLens Credit Risk Intelligence Platform using Matplotlib.
Saves the output to ./documents/presentation.pdf.
"""
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
import textwrap

# Setup output directory
os.makedirs("documents", exist_ok=True)
pdf_path = "documents/presentation.pdf"

# Design tokens (matching RiskLens dark theme)
BG_COLOR = '#060912'
TEXT_COLOR = '#94A3B8'   # Muted gray body text for professional tone
TITLE_COLOR = '#F1F5F9'  # Bright white for titles
ACCENT_CYAN = '#00F2FE'
ACCENT_PURPLE = '#7F00FF'
ACCENT_GREEN = '#00E676'
ACCENT_RED = '#FF1744'
CARD_BG = '#0C101C'

def apply_slide_theme(fig, ax, title_text=""):
    """Setup standard slide formatting."""
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    
    if title_text:
        # Title text
        ax.text(5, 91, title_text, fontsize=18, color=TITLE_COLOR, 
                fontweight='bold', fontfamily='sans-serif')
        # Underline accent
        ax.plot([5, 18], [88, 88], color=ACCENT_CYAN, linewidth=2.5)
        # Footer
        ax.text(5, 3, "RiskLens Credit Risk Intelligence Platform", 
                fontsize=8, color='#475569', fontfamily='sans-serif')
        ax.text(95, 3, "Confidential", 
                fontsize=8, color='#475569', fontfamily='sans-serif', ha='right')

def draw_card(ax, x, y, width, height, title="", border_color=ACCENT_CYAN):
    """Draw a styled glassmorphic box for holding content."""
    rect = patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=1.5",
        facecolor=CARD_BG,
        edgecolor=border_color,
        linewidth=1.2,
        alpha=0.9
    )
    ax.add_patch(rect)
    
    if title:
        ax.text(x + 2, y + height - 2.5, title, fontsize=11, 
                color=TITLE_COLOR, fontweight='bold', fontfamily='sans-serif')

def draw_wrapped_text(ax, x, y, text, width_chars, fontsize=8, color=TEXT_COLOR, linespacing=1.45):
    """Draw text that wraps automatically within a character width, flowing downwards."""
    lines = []
    for paragraph in text.split('\n'):
        if paragraph.strip() == "":
            lines.append("")
        elif paragraph.startswith("•") or paragraph.startswith("✔") or paragraph.startswith("-"):
            bullet = paragraph[0] + " "
            content = paragraph[1:].strip()
            # Wrap remaining content
            wrapped = textwrap.wrap(content, width=width_chars - 3)
            if wrapped:
                lines.append(bullet + wrapped[0])
                for w in wrapped[1:]:
                    lines.append("   " + w)
            else:
                lines.append(bullet)
        else:
            lines.extend(textwrap.wrap(paragraph, width=width_chars))
            
    # Calculate coordinate decrement per line based on scale
    y_decrement = linespacing * (fontsize / 72.0) * (100.0 / 8.5)
    
    current_y = y
    for line in lines:
        ax.text(x, current_y, line, fontsize=fontsize, color=color, 
                fontfamily='sans-serif', va='top', ha='left')
        current_y -= y_decrement

with PdfPages(pdf_path) as pdf:
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SLIDE 1: Title Slide (Cover)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    fig, ax = plt.subplots(figsize=(11, 8.5))
    apply_slide_theme(fig, ax)
    
    # Large Cover Title
    ax.text(50, 60, "RiskLens", fontsize=48, color=ACCENT_CYAN, 
            fontweight='black', fontfamily='sans-serif', ha='center')
    ax.text(50, 48, "AI-Powered Credit Risk Intelligence Platform", 
            fontsize=18, color=TITLE_COLOR, fontweight='medium', 
            fontfamily='sans-serif', ha='center')
    
    # Divider line
    ax.plot([30, 70], [42, 42], color=ACCENT_PURPLE, linewidth=2.5)
    
    # Credits
    ax.text(50, 28, "Developed by: Gourinandini", fontsize=12, 
            color=TITLE_COLOR, fontfamily='sans-serif', ha='center')
    ax.text(50, 22, "Streamlit · LightGBM · SHAP · Groq LLaMA 3.3", fontsize=10, 
            color='#64748B', fontfamily='sans-serif', ha='center', fontstyle='italic')
    
    pdf.savefig(fig, facecolor=BG_COLOR)
    plt.close()
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SLIDE 2: Use Case & Business Value
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    fig, ax = plt.subplots(figsize=(11, 8.5))
    apply_slide_theme(fig, ax, "Use Case & Business Value")
    
    # Column 1: The Problem
    draw_card(ax, 5, 20, 42, 60, "The Credit Risk Challenge", ACCENT_RED)
    problem_text = (
        "• High Default Losses:\n"
        "  Unpaid loans drive heavy financial exposure.\n\n"
        "• Black-Box ML Models:\n"
        "  Accurate models lack explainability, failing regulatory audits (e.g., SR 11-7).\n\n"
        "• Complex Data Access:\n"
        "  Risk teams cannot explore portfolio metrics without technical SQL knowledge."
    )
    draw_wrapped_text(ax, 7, 72, problem_text, width_chars=48, fontsize=9)
    
    # Column 2: The Solution
    draw_card(ax, 51, 20, 42, 60, "The RiskLens Core Solution", ACCENT_GREEN)
    solution_text = (
        "✔ Accurate Scoring:\n"
        "  Real-time default risk prediction using optimized threshold tuning.\n\n"
        "✔ Human-Readable Rules:\n"
        "  Surrogate decision trees convert black-box predictions into transparent rules.\n\n"
        "✔ Global & Local Explainability:\n"
        "  SHAP integration provides instant feature attribution for compliance.\n\n"
        "✔ AI Copilot Agent:\n"
        "  Conversational interface translates natural language queries into instant SQL."
    )
    draw_wrapped_text(ax, 53, 72, solution_text, width_chars=48, fontsize=9)
            
    pdf.savefig(fig, facecolor=BG_COLOR)
    plt.close()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SLIDE 3: System Architecture
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    fig, ax = plt.subplots(figsize=(11, 8.5))
    apply_slide_theme(fig, ax, "Platform System Architecture")
    
    # Draw Architecture flow boxes
    # Stage 1
    draw_card(ax, 4, 58, 20, 22, "1. Ingest", ACCENT_PURPLE)
    draw_wrapped_text(ax, 6, 73, "• Kaggle Application CSV\n• 307k Applicants\n• 122 Features", width_chars=22, fontsize=8)
    
    # Arrow 1->2
    ax.annotate("", xy=(29, 70), xytext=(25, 70),
                arrowprops=dict(arrowstyle="->", color=ACCENT_CYAN, lw=2))
    
    # Stage 2
    draw_card(ax, 30, 58, 20, 22, "2. Preprocess", ACCENT_PURPLE)
    draw_wrapped_text(ax, 32, 73, "• Outlier correction\n• Feature engineering\n• Median imputation\n• One-Hot encoding", width_chars=22, fontsize=8)
    
    # Arrow 2->3
    ax.annotate("", xy=(55, 70), xytext=(51, 70),
                arrowprops=dict(arrowstyle="->", color=ACCENT_CYAN, lw=2))
    
    # Stage 3
    draw_card(ax, 56, 58, 20, 22, "3. ML Training", ACCENT_PURPLE)
    draw_wrapped_text(ax, 58, 73, "• LightGBM training\n• Early stopping (50)\n• scale_pos_weight", width_chars=22, fontsize=8)
    
    # Arrow 3->4
    ax.annotate("", xy=(81, 70), xytext=(77, 70),
                arrowprops=dict(arrowstyle="->", color=ACCENT_CYAN, lw=2))
                
    # Stage 4
    draw_card(ax, 82, 58, 14, 22, "4. Artifacts", ACCENT_GREEN)
    draw_wrapped_text(ax, 84, 73, "• Model file\n• Pipeline joblib\n• Metric JSONs\n• Decision rules", width_chars=16, fontsize=8)

    # Down Arrows from Stage 4 to Runtime
    ax.annotate("", xy=(40, 48), xytext=(40, 58),
                arrowprops=dict(arrowstyle="->", color=ACCENT_CYAN, lw=2))
    ax.annotate("", xy=(70, 48), xytext=(70, 58),
                arrowprops=dict(arrowstyle="->", color=ACCENT_CYAN, lw=2))
                
    # Runtime Stage 1: UI Dashboard
    draw_card(ax, 10, 18, 36, 26, "Runtime: Streamlit Front-End", ACCENT_CYAN)
    draw_wrapped_text(ax, 12, 37, "Interactive Business Dashboard:\n• Portfolio analytics (EDA)\n• Individual risk predictions & SHAP\n• Model metrics and threshold radar\n• Explainable rule finder & filters", width_chars=40, fontsize=8)
    
    # Runtime Stage 2: Copilot Agent
    draw_card(ax, 52, 18, 38, 26, "Runtime: RiskLens Copilot", ACCENT_CYAN)
    draw_wrapped_text(ax, 54, 37, "NL-to-SQL Analytics Agent:\n• Embedded SQLite database\n• LLaMA-based natural language parser\n• Built-in SQL validation checks\n• Context-aware conversation memory", width_chars=42, fontsize=8)

    pdf.savefig(fig, facecolor=BG_COLOR)
    plt.close()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SLIDE 4: Model Performance & Class Imbalance Strategy
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    fig, ax = plt.subplots(figsize=(11, 8.5))
    apply_slide_theme(fig, ax, "Model Performance & Imbalance Strategy")
    
    # Stats columns
    draw_card(ax, 5, 52, 42, 30, "Model Metrics Report", ACCENT_CYAN)
    metrics_text = (
        "• ROC-AUC Score: 0.7683 (Robust separation)\n"
        "• PR-AUC Score:  0.2591 (3.2x random baseline)\n"
        "• F1 Score:      0.3202 (Optimized for minority)\n"
        "• Calibrated Threshold: 0.6815\n"
        "• Validation Accuracy: 86.0%"
    )
    draw_wrapped_text(ax, 7, 74, metrics_text, width_chars=45, fontsize=8.5)
            
    # Class imbalance explanation
    draw_card(ax, 5, 12, 42, 34, "Class Imbalance Strategy", ACCENT_CYAN)
    imbalance_text = (
        "• Skewed Dataset:\n"
        "  ~92% repaid, ~8% defaulted. Standard accuracy is deceptive.\n\n"
        "• scale_pos_weight Tuning:\n"
        "  Penalizes missed defaults 11.3x more heavily than false positives.\n\n"
        "• Custom Classification Threshold:\n"
        "  Set at score >= 0.6815 to maximize minority-class F1-score."
    )
    draw_wrapped_text(ax, 7, 39, imbalance_text, width_chars=48, fontsize=8)
            
    # Pie chart on the right representing dataset targets
    draw_card(ax, 52, 12, 43, 70, "Applicant Default Imbalance (307k)", ACCENT_PURPLE)
    
    # Subplot placement for pie chart
    pie_ax = fig.add_axes([0.56, 0.18, 0.35, 0.45])
    pie_ax.patch.set_facecolor(CARD_BG)
    
    labels = ['Repaid (91.9%)', 'Defaulted (8.1%)']
    sizes = [91.9, 8.1]
    colors = [ACCENT_GREEN, ACCENT_RED]
    
    wedges, texts = pie_ax.pie(
        sizes, labels=labels, colors=colors, startangle=140,
        textprops=dict(color=TEXT_COLOR, fontsize=10, fontweight='bold'),
        wedgeprops=dict(width=0.4, edgecolor=BG_COLOR, linewidth=3)
    )
    pie_ax.axis('equal')

    pdf.savefig(fig, facecolor=BG_COLOR)
    plt.close()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SLIDE 5: Explainability & Decision Rules
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    fig, ax = plt.subplots(figsize=(11, 8.5))
    apply_slide_theme(fig, ax, "Model Transparency & Rules Engine")
    
    # Column 1: Explainability
    draw_card(ax, 5, 12, 42, 70, "SHAP Local & Global Explanations", ACCENT_CYAN)
    xai_text = (
        "• Local & Global SHAP Values:\n"
        "  Uses shap.TreeExplainer for exact risk contributions.\n\n"
        "• Individual Risk Shift:\n"
        "  Highlights features that increase risk (red) or reduce risk (green) relative to the base average.\n\n"
        "• Top Influential Features:\n"
        "  1. External Credit Scores (EXT_SOURCE_2/3)\n"
        "  2. Age (AGE_YEARS)\n"
        "  3. Leverage (CREDIT_INCOME_RATIO)"
    )
    draw_wrapped_text(ax, 7, 74, xai_text, width_chars=48, fontsize=8)
            
    # Column 2: Rules Engine
    draw_card(ax, 51, 12, 44, 70, "Surrogate Rule Derivation", ACCENT_GREEN)
    rules_text = (
        "• Model Approximation:\n"
        "  Shallow decision tree models the LightGBM decision boundary.\n\n"
        "• Key Metrics:\n"
        "  - 16 human-readable rules\n"
        "  - 85.73% surrogate fidelity (agreement rate)\n\n"
        "• Calibrated Risk Bands:\n"
        "  Low, Medium, High risk bands derived from leaf nodes.\n\n"
        "• Sample Decision Rule:\n"
        "  IF Credit Score 2 <= 0.4305 AND Credit Score 3 > 0.31\n"
        "  AND Employed > 4.5 years\n"
        "  --> RISK BAND: Medium (58.9% Confidence)"
    )
    draw_wrapped_text(ax, 53, 74, rules_text, width_chars=50, fontsize=8)

    pdf.savefig(fig, facecolor=BG_COLOR)
    plt.close()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SLIDE 6: RiskLens Copilot (NL-to-SQL Chatbot)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    fig, ax = plt.subplots(figsize=(11, 8.5))
    apply_slide_theme(fig, ax, "RiskLens Copilot — Conversational Analyst")
    
    # Left Column: Features
    draw_card(ax, 5, 12, 42, 70, "Copilot Agent Core Capabilities", ACCENT_CYAN)
    copilot_text = (
        "• Natural Language to SQL:\n"
        "  Converts user queries to SQLite code and runs them instantly.\n\n"
        "• Contextual Memory:\n"
        "  Remembers previous questions (memory depth k=5).\n\n"
        "• Guardrails & Security:\n"
        "  - Restricts access to read-only SELECT commands.\n"
        "  - Blocks destructive updates (DROP/DELETE).\n"
        "  - Caps response payloads to 100 rows."
    )
    draw_wrapped_text(ax, 7, 74, copilot_text, width_chars=48, fontsize=8)
            
    # Right Column: Query Patterns
    draw_card(ax, 51, 12, 44, 70, "Verified Query Patterns & Examples", ACCENT_CYAN)
    patterns_text = (
        "✔ Aggregations & Metrics:\n"
        "  - 'What is the average income of defaulted applicants?'\n"
        "  - SQL: SELECT AVG(AMT_INCOME_TOTAL) FROM app WHERE TARGET=1\n\n"
        "✔ Grouped Rates:\n"
        "  - 'Show default rate by education level'\n"
        "  - SQL: SELECT education, AVG(TARGET) FROM app GROUP BY education\n\n"
        "✔ Filters & Counts:\n"
        "  - 'Find count of female applicants with >2 children'\n"
        "  - SQL: SELECT COUNT(*) FROM app WHERE gender='F' AND children > 2"
    )
    draw_wrapped_text(ax, 53, 74, patterns_text, width_chars=50, fontsize=8)

    pdf.savefig(fig, facecolor=BG_COLOR)
    plt.close()

print(f"Presentation successfully compiled and saved to: {pdf_path}")
