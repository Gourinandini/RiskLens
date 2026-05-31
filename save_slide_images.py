"""
Python script to render and save RiskLens presentation slides as PNG images
for user review and chat preview.
Saves slide images to ./documents/slide1.png to slide7.png.
"""
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Design tokens
BG_COLOR = '#060912'
TEXT_COLOR = '#CBD5E1'
TITLE_COLOR = '#F1F5F9'
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
        ax.text(5, 92, title_text, fontsize=20, color=TITLE_COLOR, 
                fontweight='bold', fontfamily='sans-serif')
        ax.plot([5, 18], [89, 89], color=ACCENT_CYAN, linewidth=2)
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
        linewidth=1,
        alpha=0.85
    )
    ax.add_patch(rect)
    if title:
        ax.text(x + 2, y + height - 2, title, fontsize=11, 
                color=TITLE_COLOR, fontweight='bold', fontfamily='sans-serif')

os.makedirs("documents", exist_ok=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 1
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, ax = plt.subplots(figsize=(11, 8.5))
apply_slide_theme(fig, ax)
ax.text(50, 60, "RiskLens", fontsize=48, color=ACCENT_CYAN, fontweight='black', fontfamily='sans-serif', ha='center')
ax.text(50, 48, "AI-Powered Credit Risk Intelligence Platform", fontsize=18, color=TITLE_COLOR, fontweight='medium', fontfamily='sans-serif', ha='center')
ax.plot([30, 70], [42, 42], color=ACCENT_PURPLE, linewidth=2.5)
ax.text(50, 28, "Developed by: Gourinandini", fontsize=12, color=TEXT_COLOR, fontfamily='sans-serif', ha='center')
ax.text(50, 22, "Streamlit · LightGBM · SHAP · Groq LLaMA 3.3", fontsize=10, color='#64748B', fontfamily='sans-serif', ha='center', fontstyle='italic')
plt.savefig("documents/slide1.png", dpi=100, facecolor=BG_COLOR)
plt.close()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 2
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, ax = plt.subplots(figsize=(11, 8.5))
apply_slide_theme(fig, ax, "Use Case & Business Value")
draw_card(ax, 5, 20, 42, 60, "The Credit Risk Challenge", ACCENT_RED)
problem_text = (
    "• High Default Exposure:\n"
    "  Financial institutions face heavy losses from unpaid loans.\n\n"
    "• Opposing Objectives:\n"
    "  Underwriters need accurate predictions, but legacy models\n"
    "  lack granular resolution.\n\n"
    "• Black-Box Compliancy Issues:\n"
    "  Modern ML (e.g., LightGBM) yields high performance but\n"
    "  remains unexplainable, failing regulatory audits (like SR 11-7).\n\n"
    "• Data Inaccessibility:\n"
    "  Risk managers struggle to explore portfolio metrics without\n"
    "  writing complex SQL queries."
)
ax.text(7, 73, problem_text, fontsize=10, color=TEXT_COLOR, fontfamily='sans-serif', linespacing=1.6, va='top')

draw_card(ax, 51, 20, 42, 60, "The RiskLens Core Solution", ACCENT_GREEN)
solution_text = (
    "✔ Calibrated Risk Predictor:\n"
    "  Real-time default probability scoring using optimized thresholds.\n\n"
    "✔ Regulatory Auditability via Rules:\n"
    "  Post-hoc surrogate decision tree maps complex predictions into\n"
    "  human-readable if/then decision rules.\n\n"
    "✔ Explainability at Scale:\n"
    "  Global & local feature attributions using SHAP explainers.\n\n"
    "✔ RiskLens Copilot (NL-to-SQL):\n"
    "  Allows business teams to query SQLite portfolio records in\n"
    "  natural language without technical knowledge."
)
ax.text(53, 73, solution_text, fontsize=10, color=TEXT_COLOR, fontfamily='sans-serif', linespacing=1.6, va='top')
plt.savefig("documents/slide2.png", dpi=100, facecolor=BG_COLOR)
plt.close()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 3
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, ax = plt.subplots(figsize=(11, 8.5))
apply_slide_theme(fig, ax, "Platform System Architecture")
draw_card(ax, 4, 60, 20, 20, "1. Ingestion", ACCENT_PURPLE)
ax.text(6, 73, "Kaggle CSV Ingest\n• 307k applicants\n• 122 raw features", fontsize=9, color=TEXT_COLOR, va='top')
ax.annotate("", xy=(29, 70), xytext=(25, 70), arrowprops=dict(arrowstyle="->", color=ACCENT_CYAN, lw=2))

draw_card(ax, 30, 60, 20, 20, "2. Preprocessing", ACCENT_PURPLE)
ax.text(32, 73, "DataPreprocessor\n• Anomaly fix\n• Feature engineering\n• Median imputation\n• OHE Encoding", fontsize=8, color=TEXT_COLOR, va='top')
ax.annotate("", xy=(55, 70), xytext=(51, 70), arrowprops=dict(arrowstyle="->", color=ACCENT_CYAN, lw=2))

draw_card(ax, 56, 60, 20, 20, "3. ML Training", ACCENT_PURPLE)
ax.text(58, 73, "ModelTrainer\n• LightGBM Model\n• Early Stopping (50)\n• scale_pos_weight", fontsize=9, color=TEXT_COLOR, va='top')
ax.annotate("", xy=(81, 70), xytext=(77, 70), arrowprops=dict(arrowstyle="->", color=ACCENT_CYAN, lw=2))

draw_card(ax, 82, 60, 14, 20, "4. Artifacts", ACCENT_GREEN)
ax.text(84, 73, "• lgbm_model.pkl\n• preprocessor.joblib\n• metrics.json\n• rules.json", fontsize=8, color=TEXT_COLOR, va='top')

ax.annotate("", xy=(40, 48), xytext=(40, 58), arrowprops=dict(arrowstyle="->", color=ACCENT_CYAN, lw=2))
ax.annotate("", xy=(70, 48), xytext=(70, 58), arrowprops=dict(arrowstyle="->", color=ACCENT_CYAN, lw=2))

draw_card(ax, 10, 20, 36, 25, "Runtime: Streamlit Front-End", ACCENT_CYAN)
ax.text(12, 38, "5-Tab Business Interface:\n• Portfolio Analytics (EDA)\n• Local Risk Predictor + SHAP Bar Chart\n• Model Transparency & Metrics Radar\n• Business Rule Search & Filters", fontsize=8.5, color=TEXT_COLOR, va='top')

draw_card(ax, 52, 20, 38, 25, "Runtime: RiskLens Copilot", ACCENT_CYAN)
ax.text(54, 38, "NL-to-SQL Engine:\n• SQLite DB (100k records + predictions)\n• Groq LLaMA 3.3 API Connector\n• Validation checks (SELECT keyword filters)\n• LangChain BufferWindowMemory (k=5)", fontsize=8.5, color=TEXT_COLOR, va='top')
plt.savefig("documents/slide3.png", dpi=100, facecolor=BG_COLOR)
plt.close()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 4
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, ax = plt.subplots(figsize=(11, 8.5))
apply_slide_theme(fig, ax, "Model Performance & Imbalance Strategy")
draw_card(ax, 5, 52, 42, 30, "Model Metrics Report", ACCENT_CYAN)
metrics_text = (
    "• ROC-AUC Score:  0.7683\n"
    "• PR-AUC Score:   0.2591  (3.2x over random baseline)\n"
    "• F1 Score:       0.3202\n"
    "• Calibrated Threshold: 0.6815\n"
    "• Overall Validation Accuracy: 86.0%"
)
ax.text(7, 75, metrics_text, fontsize=11, color=TEXT_COLOR, fontfamily='monospace', fontweight='bold', linespacing=1.5, va='top')

draw_card(ax, 5, 12, 42, 34, "Class Imbalance Strategy", ACCENT_CYAN)
imbalance_text = (
    "• The Challenge:\n"
    "  Severe skewness (~92% repaid, ~8% defaulted). A naive\n"
    "  model achieves 92% accuracy by approving everyone.\n\n"
    "• scale_pos_weight Penalty:\n"
    "  Loss function penalizes missed defaults ~11.3x more heavily.\n\n"
    "• F1-Optimal Thresholding:\n"
    "  Precision-recall curve search anchors high-risk classification\n"
    "  at score >= 0.6815 to maximize minority-class F1."
)
ax.text(7, 39, imbalance_text, fontsize=9.5, color=TEXT_COLOR, fontfamily='sans-serif', linespacing=1.4, va='top')

draw_card(ax, 52, 12, 43, 70, "Applicant Default Imbalance (307k)", ACCENT_PURPLE)
pie_ax = fig.add_axes([0.56, 0.18, 0.35, 0.45])
pie_ax.patch.set_facecolor(CARD_BG)
labels = ['Repaid (91.9%)', 'Defaulted (8.1%)']
sizes = [91.9, 8.1]
colors = [ACCENT_GREEN, ACCENT_RED]
pie_ax.pie(
    sizes, labels=labels, colors=colors, startangle=140,
    textprops=dict(color=TEXT_COLOR, fontsize=10, fontweight='bold'),
    wedgeprops=dict(width=0.4, edgecolor=BG_COLOR, linewidth=3)
)
pie_ax.axis('equal')
plt.savefig("documents/slide4.png", dpi=100, facecolor=BG_COLOR)
plt.close()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 5
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, ax = plt.subplots(figsize=(11, 8.5))
apply_slide_theme(fig, ax, "Model Transparency & Rules Engine")
draw_card(ax, 5, 12, 42, 70, "SHAP Local & Global Explanations", ACCENT_CYAN)
xai_text = (
    "• True Transparency:\n"
    "  RiskLens uses shap.TreeExplainer to compute exact Shapley\n"
    "  attribution values for applicant features.\n\n"
    "• Local Risk Shift:\n"
    "  Attributions show how each feature pushes individual probability\n"
    "  higher (red, increases risk) or lower (green, reduces risk)\n"
    "  relative to the dataset expected base probability.\n\n"
    "• Top Global Features:\n"
    "  1. External Credit Score 2 (EXT_SOURCE_2)\n"
    "  2. External Credit Score 3 (EXT_SOURCE_3)\n"
    "  3. Avg External Score (EXT_SOURCE_MEAN)\n"
    "  4. Age (AGE_YEARS)\n"
    "  5. Leverage Ratio (CREDIT_INCOME_RATIO)"
)
ax.text(7, 75, xai_text, fontsize=9.5, color=TEXT_COLOR, fontfamily='sans-serif', linespacing=1.5, va='top')

draw_card(ax, 51, 12, 44, 70, "Surrogate Rule Derivation", ACCENT_GREEN)
rules_text = (
    "• Method:\n"
    "  Trains a shallow DecisionTree (max depth=4) on LightGBM\n"
    "  binarized targets to model the complex model's boundary.\n\n"
    "• Key Metrics:\n"
    "  - Extracted Rules: 16 human-readable paths\n"
    "  - Surrogate Fidelity: 85.73% agreement rate\n\n"
    "• Calibrated Band Rules:\n"
    "  Assigns leaves to Low, Medium, and High bands based on average\n"
    "  leaf default probability (optimal thresholds).\n\n"
    "• Sample Rule Output (Medium Risk):\n"
    "  IF External Credit Score 2 <= 0.4305\n"
    "  AND External Credit Score 3 > 0.31\n"
    "  AND Employment Duration > 4.5 years\n"
    "  --> RISK BAND: Medium  (Confidence: 58.9%, Support: 4.4%)"
)
ax.text(53, 75, rules_text, fontsize=9.5, color=TEXT_COLOR, fontfamily='sans-serif', linespacing=1.4, va='top')
plt.savefig("documents/slide5.png", dpi=100, facecolor=BG_COLOR)
plt.close()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 6
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, ax = plt.subplots(figsize=(11, 8.5))
apply_slide_theme(fig, ax, "RiskLens Copilot — Conversational Analyst")
draw_card(ax, 5, 12, 42, 70, "Copilot Agent Core Capabilities", ACCENT_CYAN)
copilot_text = (
    "• Zero-Code Data Access:\n"
    "  Translates natural language questions into executable SQL queries,\n"
    "  runs them against SQLite, and summarizes output.\n\n"
    "• Multi-Turn Memory Context:\n"
    "  Utilizes LangChain ConversationBufferWindowMemory (k=5) to\n"
    "  retain query sequence for subsequent follow-up tasks.\n\n"
    "• Schema-Anchored Safeguards:\n"
    "  Injects DDL and standard category values inside prompts to\n"
    "  block column hallucination.\n\n"
    "• Two-Layer Injection Defenses:\n"
    "  - Validates query structure (forces SELECT statements only).\n"
    "  - Blocks destructive commands (DROP, DELETE, UPDATE, ALTER, etc.).\n"
    "  - Caps output fetching to 100 records to prevent memory overflow."
)
ax.text(7, 75, copilot_text, fontsize=9.5, color=TEXT_COLOR, fontfamily='sans-serif', linespacing=1.5, va='top')

draw_card(ax, 51, 12, 44, 70, "Verified Query Patterns & Examples", ACCENT_CYAN)
patterns_text = (
    "✔ Query Pattern 1: Aggregates by Class\n"
    "  * 'What is the average income of applicants who defaulted?'\n"
    "  * SQL: SELECT AVG(AMT_INCOME_TOTAL) FROM applications WHERE TARGET=1\n\n"
    "✔ Query Pattern 2: Categorical Default Rates\n"
    "  * 'Show default rate by education type'\n"
    "  * SQL: SELECT NAME_EDUCATION_TYPE, AVG(TARGET)*100 FROM applications...\n\n"
    "✔ Query Pattern 3: Filter & Count Queries\n"
    "  * 'How many female applicants have more than 2 children?'\n"
    "  * SQL: SELECT COUNT(*) FROM applications WHERE CODE_GENDER='F'...\n\n"
    "✔ Query Pattern 4: Top-N Leaders\n"
    "  * 'What are the top 10 highest credit amounts?'\n"
    "  * SQL: SELECT AMT_CREDIT FROM applications ORDER BY AMT_CREDIT DESC...\n\n"
    "✔ Query Pattern 5: Multi-band comparisons\n"
    "  * 'Show average external credit score by risk band'\n"
    "  * SQL: SELECT RISK_BAND, AVG(EXT_SOURCE_MEAN) FROM applications..."
)
ax.text(53, 75, patterns_text, fontsize=8.5, color=TEXT_COLOR, fontfamily='sans-serif', linespacing=1.3, va='top')
plt.savefig("documents/slide6.png", dpi=100, facecolor=BG_COLOR)
plt.close()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 7
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, ax = plt.subplots(figsize=(11, 8.5))
apply_slide_theme(fig, ax, "Interactive Business UI Preview")
draw_card(ax, 4, 12, 22, 70, "Sidebar Panel", ACCENT_CYAN)
ax.text(6, 68, "RiskLens v2.0", fontsize=10, color=ACCENT_CYAN, fontweight='bold')
ax.text(6, 65, "Enterprise Edition", fontsize=7, color='#64748B')
ax.text(6, 52, "ROC-AUC: 0.7683", fontsize=8, color=TITLE_COLOR, fontweight='bold')
ax.text(6, 45, "PR-AUC:  0.2591", fontsize=8, color=TITLE_COLOR, fontweight='bold')
ax.text(6, 38, "F1 Score: 0.3202", fontsize=8, color=TITLE_COLOR, fontweight='bold')
ax.text(6, 25, "● ML Model: Loaded", fontsize=8, color=ACCENT_GREEN)
ax.text(6, 20, "● Database: Connected", fontsize=8, color=ACCENT_GREEN)
ax.text(6, 15, "● Copilot: Connected", fontsize=8, color=ACCENT_GREEN)

draw_card(ax, 29, 12, 66, 70, "Interactive Workspace Tabs", ACCENT_CYAN)
ax.text(31, 74, "📊 EDA Dashboard  |  🔍 Risk Predictor  |  🧠 Explainability  |  📋 Decision Rules  |  💬 RiskLens Copilot", 
        fontsize=8.5, color=ACCENT_CYAN, fontweight='bold')
        
draw_card(ax, 31, 52, 14, 15, "Applicants", ACCENT_CYAN)
ax.text(33, 56, "100,000", fontsize=12, color=TITLE_COLOR, fontweight='bold')

draw_card(ax, 47, 52, 14, 15, "Default Rate", ACCENT_RED)
ax.text(49, 56, "8.09%", fontsize=12, color=ACCENT_RED, fontweight='bold')

draw_card(ax, 63, 52, 14, 15, "Avg Credit", ACCENT_CYAN)
ax.text(65, 56, "₹5.99L", fontsize=12, color=TITLE_COLOR, fontweight='bold')

draw_card(ax, 79, 52, 14, 15, "Avg Income", ACCENT_GREEN)
ax.text(81, 56, "₹1.68L", fontsize=12, color=ACCENT_GREEN, fontweight='bold')

bar_ax = fig.add_axes([0.33, 0.18, 0.58, 0.28])
bar_ax.patch.set_facecolor(CARD_BG)
bar_ax.set_facecolor(CARD_BG)
bands = ['Low Risk', 'Medium Risk', 'High Risk']
shares = [40.0, 35.0, 25.0]
bar_colors = [ACCENT_GREEN, ACCENT_PURPLE, ACCENT_RED]
bars = bar_ax.bar(bands, shares, color=bar_colors, width=0.45)
bar_ax.set_ylabel('Portfolio Share (%)', color=TEXT_COLOR, fontsize=8)
bar_ax.tick_params(colors=TEXT_COLOR, labelsize=8)
bar_ax.set_title('Portfolio Distribution by Calibrated Risk Bands', color=TITLE_COLOR, fontsize=9, fontweight='bold')
bar_ax.grid(axis='y', color='#ffffff', alpha=0.05, linestyle='--')

for bar in bars:
    yval = bar.get_height()
    bar_ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.1f}%", 
                ha='center', va='bottom', color=TITLE_COLOR, fontsize=8, fontweight='bold')

plt.savefig("documents/slide7.png", dpi=100, facecolor=BG_COLOR)
plt.close()

print("Slides successfully rendered and saved to documents/")
