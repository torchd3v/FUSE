"""
Generate the FUSE demo paper PDF using reportlab.
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, HRFlowable, Flowable, Image
)
from reportlab.lib import colors
import os

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

OUTPUT = "fuse_paper.pdf"

# ── Page template ──────────────────────────────────────────────
def build_doc():
    doc = SimpleDocTemplate(
        OUTPUT,
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=1.0 * inch,
        rightMargin=1.0 * inch,
    )

    styles = getSampleStyleSheet()
    W = letter[0] - 2.0 * inch  # usable width

    # ── Custom styles ──
    styles.add(ParagraphStyle(
        "PaperTitle", parent=styles["Title"],
        fontSize=17, leading=21, spaceAfter=4, alignment=TA_CENTER,
        textColor=HexColor("#1a1a2e"),
    ))
    styles.add(ParagraphStyle(
        "Authors", parent=styles["Normal"],
        fontSize=11, leading=14, alignment=TA_CENTER,
        spaceAfter=2, textColor=HexColor("#333333"),
    ))
    styles.add(ParagraphStyle(
        "Affiliation", parent=styles["Normal"],
        fontSize=9, leading=12, alignment=TA_CENTER,
        spaceAfter=16, textColor=HexColor("#666666"), italics=True,
    ))
    styles.add(ParagraphStyle(
        "AbstractLabel", parent=styles["Normal"],
        fontSize=10, leading=13, fontName="Helvetica-Bold",
        alignment=TA_CENTER, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        "AbstractBody", parent=styles["Normal"],
        fontSize=9.5, leading=13, alignment=TA_JUSTIFY,
        leftIndent=24, rightIndent=24, spaceAfter=16,
        textColor=HexColor("#222222"),
    ))
    styles.add(ParagraphStyle(
        "SectionH", parent=styles["Heading1"],
        fontSize=13, leading=16, spaceBefore=18, spaceAfter=8,
        textColor=HexColor("#1a1a2e"), fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        "SubSectionH", parent=styles["Heading2"],
        fontSize=11, leading=14, spaceBefore=12, spaceAfter=6,
        textColor=HexColor("#2d2d44"), fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, leading=14, alignment=TA_JUSTIFY,
        spaceAfter=8, textColor=HexColor("#1a1a1a"),
    ))
    styles.add(ParagraphStyle(
        "BodyIndent", parent=styles["Normal"],
        fontSize=10, leading=14, alignment=TA_JUSTIFY,
        spaceAfter=8, leftIndent=18, textColor=HexColor("#1a1a1a"),
    ))
    styles.add(ParagraphStyle(
        "Caption", parent=styles["Normal"],
        fontSize=9, leading=12, alignment=TA_LEFT,
        spaceAfter=12, textColor=HexColor("#333333"),
        fontName="Helvetica-BoldOblique",
    ))
    styles.add(ParagraphStyle(
        "CodeBlock", parent=styles["Normal"],
        fontSize=8.5, leading=11, fontName="Courier",
        leftIndent=12, spaceAfter=8, textColor=HexColor("#2d2d2d"),
        backColor=HexColor("#f5f5f5"),
    ))
    styles.add(ParagraphStyle(
        "Footnote", parent=styles["Normal"],
        fontSize=8, leading=10, textColor=HexColor("#666666"),
    ))

    story = []
    S = story.append

    # ══════════════════════════════════════════════════════════
    #  TITLE BLOCK
    # ══════════════════════════════════════════════════════════
    S(Spacer(1, 12))
    S(Paragraph(
        "FUSE: Eliminating Activation Predictors for<br/>"
        "Sparse LLM Inference via Direct Gate Tracing",
        styles["PaperTitle"],
    ))
    S(Spacer(1, 6))
    S(Paragraph("Burak Egeli", styles["Authors"]))
    S(Paragraph("Independent Research", styles["Affiliation"]))

    # ── Abstract ──
    S(HRFlowable(width="60%", thickness=0.5, color=HexColor("#cccccc"),
                  spaceAfter=8, spaceBefore=0))
    S(Paragraph("Abstract", styles["AbstractLabel"]))
    S(Paragraph(
        "I present <b>FUSE</b> (Feed-forward Unit-Sparse Execution), a novel approach to sparse "
        "LLM inference that eliminates trained activation predictors entirely. In SwiGLU-based "
        "transformer models, the gate projection (W<sub>gate</sub>) already encodes which FFN neurons "
        "will fire for a given input. FUSE exploits this by keeping only W<sub>gate</sub> (33% of FFN "
        "parameters) in RAM, using it to trace active neurons, and then streaming only the "
        "corresponding rows of W<sub>up</sub> and W<sub>down</sub> from storage. I introduce a "
        "<b>per-layer adaptive calibration</b> algorithm that binary-searches for each layer's "
        "maximum tolerable sparsity above a cosine similarity floor. I validate FUSE across "
        "five models spanning three architecture families &mdash; LLaMA, Qwen2, and Qwen3.5 "
        "(hybrid Gated DeltaNet) &mdash; with zero code changes. On DeepSeek-R1-Distill-Qwen-7B, "
        "adaptive calibration achieves <b>53.3% average sparsity</b> (2.1x I/O reduction) at "
        "cos &ge; 0.95, with per-layer targets ranging from 32% to 91%. On ReLUfied TinyLlama "
        "1.1B, FUSE reaches <b>75.4% sparsity</b> (4.1x I/O reduction) at cos &ge; 0.98. On "
        "Qwen3.5-9B, FUSE achieves <b>44.3% sparsity</b> (1.8x I/O reduction), revealing a "
        "descending-slope layer sensitivity profile characteristic of the hybrid DeltaNet/attention "
        "architecture. Multi-step arithmetic reasoning (17 &times; 23 = 391) is preserved across "
        "all models. Unlike prior work (PowerInfer, Deja Vu, LLM in a Flash), FUSE requires "
        "no predictor training, no quantization, and no architecture modification &mdash; only a "
        "one-time calibration pass.",
        styles["AbstractBody"],
    ))
    S(HRFlowable(width="60%", thickness=0.5, color=HexColor("#cccccc"),
                  spaceAfter=16, spaceBefore=0))

    # ══════════════════════════════════════════════════════════
    #  1. INTRODUCTION
    # ══════════════════════════════════════════════════════════
    S(Paragraph("1&nbsp;&nbsp;&nbsp;Introduction", styles["SectionH"]))
    S(Paragraph(
        "Running large language models on consumer hardware is fundamentally an I/O problem. "
        "A 70B-parameter model in FP16 requires ~140 GB of weight data. During autoregressive "
        "generation, each token requires streaming the full model weights through the compute "
        "units, making memory bandwidth &mdash; not compute &mdash; the bottleneck. The FFN (feed-forward "
        "network) layers account for approximately two-thirds of total parameters and dominate "
        "this I/O cost.",
        styles["Body"],
    ))
    S(Paragraph(
        "Recent work has shown that FFN layers exhibit high activation sparsity: most neurons "
        "produce near-zero outputs for any given input. Systems like PowerInfer, Deja Vu, and "
        "LLM in a Flash exploit this by predicting which neurons will activate and loading "
        "only those weights from storage. However, all these approaches require <b>trained "
        "activation predictor networks</b> &mdash; small MLPs or hash tables that must be trained on "
        "calibration data for each model, adding complexity and potential prediction errors.",
        styles["Body"],
    ))
    S(Paragraph(
        "I observe that SwiGLU-based models already contain a built-in activation predictor: "
        "the <b>gate projection</b> W<sub>gate</sub>. In the standard SwiGLU FFN computation "
        "y = W<sub>down</sub>(SiLU(W<sub>gate</sub>x) &odot; W<sub>up</sub>x), the gate "
        "activation SiLU(W<sub>gate</sub>x) directly determines each neuron's contribution. "
        "Neurons where |SiLU(W<sub>gate</sub>x)| is small contribute negligibly to the output "
        "regardless of W<sub>up</sub> and W<sub>down</sub>.",
        styles["Body"],
    ))
    S(Paragraph(
        "FUSE turns this observation into a practical inference system with two phases per layer:",
        styles["Body"],
    ))
    S(Paragraph(
        "<b>Phase 1 (Trace):</b> Compute gate activations using W<sub>gate</sub> (always "
        "resident in RAM). Select the top-K neurons by magnitude. This is the <i>only</i> "
        "full matrix-vector multiply required.<br/>"
        "<b>Phase 2 (Sparse Compute):</b> Load only the selected neurons' rows from "
        "W<sub>up</sub> and W<sub>down</sub> (streamed from disk/SSD). Compute the sparse "
        "FFN output using only these neurons.",
        styles["BodyIndent"],
    ))
    S(Paragraph(
        "Since W<sub>gate</sub> is only one-third of the FFN parameters, and the remaining "
        "two-thirds are loaded sparsely, FUSE achieves significant I/O reduction with no "
        "predictor training and no loss of precision on the activated neurons.",
        styles["Body"],
    ))
    S(Paragraph(
        "The practical implication is that <b>FUSE enables running models far larger than "
        "available RAM</b>. W<sub>gate</sub> is ~33% of FFN and FFN is ~67% of total parameters, "
        "so W<sub>gate</sub> plus attention weights require roughly 40% of the full model size "
        "in RAM. The remaining 60% (W<sub>up</sub> and W<sub>down</sub>) is streamed from "
        "SSD on demand, with sparsity reducing the actual bytes read per token. A 70B model "
        "(140 GB in FP16) could thus run with ~35 GB of RAM and an NVMe drive, rather than "
        "requiring 140 GB of GPU memory. This is orthogonal to quantization &mdash; combining "
        "FUSE with 4-bit quantization would compound savings further (e.g., 50% sparsity "
        "&times; 4&times; quantization = 8&times; total I/O reduction).",
        styles["Body"],
    ))
    S(Paragraph(
        "A key contribution is my <b>per-layer adaptive calibration</b> algorithm. Rather than "
        "applying a uniform sparsity target across all layers (which wastes budget on tolerant "
        "layers while degrading sensitive ones), I run a one-time calibration pass that "
        "binary-searches for each layer's maximum sparsity above a quality floor. This "
        "consistently outperforms flat sparsity: on DeepSeek-R1 7B, adaptive calibration "
        "achieves 53.3% overall sparsity vs. the 40% flat-sparsity sweet spot &mdash; a 33% "
        "relative improvement at comparable quality.",
        styles["Body"],
    ))

    # ══════════════════════════════════════════════════════════
    #  2. RELATED WORK
    # ══════════════════════════════════════════════════════════
    S(Paragraph("2&nbsp;&nbsp;&nbsp;Related Work", styles["SectionH"]))
    S(Paragraph(
        "<b>Activation Sparsity in LLMs.</b> Liu et al. (2023) demonstrated that ReLU-based "
        "LLMs exhibit up to 90%+ activation sparsity. ProSparse and TurboSparse achieve 90-95% "
        "sparsity through ReLU fine-tuning. CATS (Contextual Activation Token Sparsity) and "
        "MoC (Mixture of Cuts) use the gate projection to select neurons for <i>compute</i> "
        "savings on GPU, but do not address I/O-bound disk-streaming scenarios.",
        styles["Body"],
    ))
    S(Paragraph(
        "<b>I/O-Optimized Sparse Inference.</b> PowerInfer (Xue et al., 2024) classifies "
        "neurons into hot (GPU-resident) and cold (CPU-resident) sets using offline profiling, "
        "with a trained predictor for runtime activation prediction. Deja Vu (Liu et al., 2023) "
        "trains small MLP predictors for each layer. LLM in a Flash (Alizadeh et al., 2024) "
        "uses precomputed activation statistics to determine loading patterns. All three require "
        "model-specific predictor training.",
        styles["Body"],
    ))
    S(Paragraph(
        "<b>FUSE's position.</b> I combine the key insight from CATS/MoC (the gate <i>is</i> "
        "the predictor) with the I/O optimization target of PowerInfer/Deja Vu/LLM in a Flash. "
        "This combination &mdash; gate tracing for I/O reduction &mdash; has not been explored in "
        "prior work. Additionally, my per-layer adaptive calibration addresses the layer "
        "heterogeneity that flat-sparsity approaches ignore.",
        styles["Body"],
    ))

    # ══════════════════════════════════════════════════════════
    #  3. METHOD
    # ══════════════════════════════════════════════════════════
    S(Paragraph("3&nbsp;&nbsp;&nbsp;Method", styles["SectionH"]))

    S(Paragraph("3.1&nbsp;&nbsp;&nbsp;Gate-as-Tracer Sparse Execution", styles["SubSectionH"]))
    S(Paragraph(
        "Consider a standard SwiGLU FFN layer with weight matrices W<sub>gate</sub>, "
        "W<sub>up</sub> &isin; R<super>d<sub>ffn</sub> x d<sub>model</sub></super> and "
        "W<sub>down</sub> &isin; R<super>d<sub>model</sub> x d<sub>ffn</sub></super>. "
        "The dense forward pass computes:",
        styles["Body"],
    ))
    S(Paragraph(
        "&nbsp;&nbsp;&nbsp;&nbsp;y = W<sub>down</sub>(SiLU(W<sub>gate</sub>x) &odot; W<sub>up</sub>x)",
        styles["CodeBlock"],
    ))
    S(Paragraph(
        "Let g = SiLU(W<sub>gate</sub>x) be the gate activation vector. FUSE selects the "
        "set S of neuron indices where |g<sub>i</sub>| is largest, keeping |S| = K = "
        "d<sub>ffn</sub>(1 - s) neurons for target sparsity s. The sparse output is:",
        styles["Body"],
    ))
    S(Paragraph(
        "&nbsp;&nbsp;&nbsp;&nbsp;y&#770; = W<sub>down</sub>[:, S] &middot; (g[S] &odot; (W<sub>up</sub>[S, :] &middot; x))",
        styles["CodeBlock"],
    ))
    S(Paragraph(
        "The only full matrix-vector product is W<sub>gate</sub>x. The W<sub>up</sub> and "
        "W<sub>down</sub> operations touch only |S| rows/columns, reducing I/O by a factor "
        "of 1/(1 - s).",
        styles["Body"],
    ))

    fig_path = os.path.join(FIGURES_DIR, "fuse_mechanism.png")
    if os.path.exists(fig_path):
        S(Spacer(1, 4))
        S(Image(fig_path, width=W * 0.85, height=W * 0.85, kind="proportional"))
        S(Paragraph(
            "<b>Figure 1.</b> FUSE two-phase pipeline. Phase 1 (always in RAM): compute gate "
            "activations, select top-K fired neurons. Phase 2 (from storage): load only fired "
            "rows of W<sub>up</sub> and columns of W<sub>down</sub>, compute sparse output.",
            styles["Caption"],
        ))

    S(Paragraph("3.2&nbsp;&nbsp;&nbsp;Per-Layer Adaptive Calibration", styles["SubSectionH"]))
    S(Paragraph(
        "Different layers exhibit different sensitivity to sparsity. Early layers may tolerate "
        "high pruning while middle layers are sensitive (or vice versa, depending on the model). "
        "My calibration algorithm finds the optimal per-layer sparsity schedule:",
        styles["Body"],
    ))
    S(Paragraph(
        "<b>Step 1.</b> Run a diverse set of calibration sentences (5 sentences covering "
        "factual, conversational, mathematical, code, and creative domains) through the model. "
        "Capture the hidden state input to each FFN layer across all tokens.<br/><br/>"
        "<b>Step 2.</b> For each layer, precompute the dense FFN output for all calibration "
        "tokens (batched matrix multiply). Cache gate activations, up-projections, and dense "
        "outputs.<br/><br/>"
        "<b>Step 3.</b> Coarse sweep: evaluate worst-case cosine similarity between dense and "
        "sparse outputs at sparsity levels {0.1, 0.2, ..., 0.9}.<br/><br/>"
        "<b>Step 4.</b> Binary search: between the highest safe and lowest unsafe coarse "
        "levels, perform 8 binary search iterations to find the maximum sparsity where "
        "worst-case cos(y, y&#770;) &ge; quality floor (typically 0.95 or 0.98).<br/><br/>"
        "<b>Step 5.</b> Save the per-layer sparsity schedule as a JSON file for use during "
        "inference.",
        styles["BodyIndent"],
    ))
    S(Paragraph(
        "The calibration is vectorized: dense outputs are computed once per layer via batched "
        "matmuls, and each binary search probe only recomputes the cheap sparse selection. "
        "Total calibration time is ~6 minutes for a 7B model on Apple M-series GPU.",
        styles["Body"],
    ))

    S(Paragraph("3.3&nbsp;&nbsp;&nbsp;Neuron-Indexed Storage Format", styles["SubSectionH"]))
    S(Paragraph(
        "For production disk streaming, I propose a neuron-indexed storage layout. Instead of "
        "storing W<sub>up</sub> and W<sub>down</sub> as separate contiguous matrices, I "
        "bundle each neuron's data: for neuron i, W<sub>up</sub>[i, :] and W<sub>down</sub>[:, i] "
        "are stored contiguously. Activating neuron i requires one sequential read of "
        "2 &times; d<sub>model</sub> values instead of two random reads into separate matrices. "
        "A lightweight index maps neuron IDs to file offsets.",
        styles["Body"],
    ))

    # ══════════════════════════════════════════════════════════
    #  4. EXPERIMENTS
    # ══════════════════════════════════════════════════════════
    S(Paragraph("4&nbsp;&nbsp;&nbsp;Experiments", styles["SectionH"]))

    S(Paragraph("4.1&nbsp;&nbsp;&nbsp;Setup", styles["SubSectionH"]))
    S(Paragraph(
        "I evaluate FUSE on five configurations across three architecture families: "
        "(1) TinyLlama 1.1B with native SwiGLU, (2) TinyLlama 1.1B with inference-time "
        "ReLUfication, (3) DeepSeek-R1-Distill-Qwen-7B with native SwiGLU, "
        "(4) Qwen3.5-4B and (5) Qwen3.5-9B &mdash; both featuring hybrid Gated DeltaNet "
        "attention (3:1 linear/full ratio) with SwiGLU FFN layers. "
        "All experiments use top-K neuron selection with per-layer adaptive calibration. "
        "Quality is measured as worst-case cosine similarity between dense and sparse layer "
        "outputs across calibration tokens. End-to-end generation is validated with a "
        "multi-step arithmetic reasoning task (17 &times; 23 = 391).",
        styles["Body"],
    ))

    S(Paragraph("4.2&nbsp;&nbsp;&nbsp;Per-Layer Calibration Results", styles["SubSectionH"]))

    # ── Figure: Sparsity comparison ──
    fig_path = os.path.join(FIGURES_DIR, "sparsity_comparison.png")
    if os.path.exists(fig_path):
        S(Image(fig_path, width=W, height=W * 0.38, kind="proportional"))
        S(Paragraph(
            "<b>Figure 2.</b> Average sparsity and I/O reduction across all five models. "
            "ReLUfied TinyLlama achieves the highest sparsity (75.4%, 4.06x I/O reduction); "
            "DeepSeek-R1 7B leads among SwiGLU models (53.3%, 2.14x).",
            styles["Caption"],
        ))

    # ── Table 1: Main results ──
    t1_data = [
        ["Model", "Act.", "Floor", "Avg\nSparsity", "I/O\nSpeedup", "Min\nLayer", "Max\nLayer", "Range"],
        ["TinyLlama 1.1B", "SwiGLU", "0.98", "35.0%", "1.54x", "29.0%\n(L12)", "42.7%\n(L0)", "13.6%"],
        ["TinyLlama 1.1B", "SwiGLU", "0.95", "48.0%", "1.92x", "40.5%\n(L13)", "57.2%\n(L0)", "16.7%"],
        ["TinyLlama 1.1B", "ReLU", "0.98", "75.4%", "4.06x", "60.2%\n(L21)", "82.4%\n(L9)", "22.2%"],
        ["DeepSeek-R1 7B", "SwiGLU", "0.95", "53.3%", "2.14x", "32.0%\n(L3)", "91.3%\n(L1)", "59.3%"],
        ["Qwen3.5-4B", "SwiGLU", "0.95", "41.2%", "1.70x", "26.2%\n(L26)", "60.7%\n(L0)", "34.4%"],
        ["Qwen3.5-9B", "SwiGLU", "0.95", "44.3%", "1.80x", "33.8%\n(L25)", "66.6%\n(L0)", "32.8%"],
    ]

    col_w = [W * f for f in [0.17, 0.08, 0.07, 0.10, 0.10, 0.13, 0.13, 0.09]]
    t1 = Table(t1_data, colWidths=col_w, repeatRows=1)
    t1.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#cccccc")),
        ("BACKGROUND", (0, 1), (-1, 1), HexColor("#f8f8ff")),
        ("BACKGROUND", (0, 2), (-1, 2), white),
        ("BACKGROUND", (0, 3), (-1, 3), HexColor("#f8f8ff")),
        ("BACKGROUND", (0, 4), (-1, 4), HexColor("#eef0ff")),
        ("FONTNAME", (0, 4), (-1, 4), "Helvetica-Bold"),
        ("BACKGROUND", (0, 5), (-1, 5), HexColor("#f0fff0")),
        ("BACKGROUND", (0, 6), (-1, 6), HexColor("#eefff0")),
        ("FONTNAME", (0, 6), (-1, 6), "Helvetica-Bold"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    S(Spacer(1, 4))
    S(t1)
    S(Paragraph(
        "<b>Table 1.</b> Per-layer adaptive calibration results across five models and three "
        "architecture families. Range = max layer sparsity - min layer sparsity. "
        "DeepSeek-R1 7B shows the widest range (59.3%); Qwen3.5 models share a distinctive "
        "descending-slope profile with late-layer bottlenecks.",
        styles["Caption"],
    ))

    S(Paragraph("4.3&nbsp;&nbsp;&nbsp;Layer Sensitivity Profiles", styles["SubSectionH"]))
    S(Paragraph(
        "The five model configurations exhibit qualitatively different layer sensitivity "
        "profiles, revealing how model architecture and size affect sparsity tolerance:",
        styles["Body"],
    ))
    S(Paragraph(
        "<b>TinyLlama SwiGLU (U-shape):</b> Early layers (0-5) and late layers (18-21) are "
        "the most tolerant (37-43%), while middle layers (9-14) form the quality bottleneck "
        "(29-30%). The sparsity range is narrow (13.6%), reflecting uniform utilization in a "
        "small model.",
        styles["Body"],
    ))
    S(Paragraph(
        "<b>TinyLlama ReLUfied (inverted):</b> Middle layers (8-13) become the <i>most</i> "
        "prunable (78-82%) because ReLU creates hard zeros in neurons that SiLU merely "
        "attenuated. The final layers (19-21) resist pruning (60-70%) as they require precise "
        "activations for next-token prediction. Overall sparsity jumps from 35% to 75.4%.",
        styles["Body"],
    ))
    S(Paragraph(
        "<b>DeepSeek-R1 7B (cliff pattern):</b> Layers 0-2 are extremely prunable (79-91%), "
        "then layer 3 drops sharply to 32% &mdash; suggesting it serves as a critical routing "
        "layer where distilled reasoning representations are established. The middle band "
        "(layers 6-20) stabilizes at 41-53%, with a gradual recovery in late layers (52-61%). "
        "The 59.3% range is 4.4x wider than TinyLlama.",
        styles["Body"],
    ))
    S(Paragraph(
        "<b>Qwen3.5-4B and 9B (descending slope):</b> Both Qwen3.5 models show a novel "
        "descending profile: early layers are the most tolerant (60-67% at L0), followed by "
        "a steady descent to a bottleneck at layers 25-26 (26-34%), then a sharp recovery at "
        "the final layer (54%). This pattern is an architectural fingerprint of Qwen3.5's "
        "3:1 hybrid design (3 DeltaNet + 1 full attention). The late bottleneck coincides "
        "with full-attention 'checkpoint' layers that reconcile the compressed recurrent "
        "state from DeltaNet layers with precise quadratic attention. The 9B model is "
        "uniformly more sparse-friendly than the 4B (44.3% vs 41.2%), confirming that "
        "larger models within the same family offer more redundancy.",
        styles["Body"],
    ))

    # ── Figure: Layer profiles ──
    fig_path = os.path.join(FIGURES_DIR, "layer_profiles.png")
    if os.path.exists(fig_path):
        S(Spacer(1, 6))
        S(Image(fig_path, width=W, height=W * 1.1, kind="proportional"))
        S(Paragraph(
            "<b>Figure 3.</b> Per-layer sparsity profiles for all five models. Each architecture "
            "produces a distinct fingerprint: U-shape (TinyLlama), inverted (ReLUfied), cliff "
            "(DeepSeek-R1), and descending slope (Qwen3.5). Min/max layers are annotated.",
            styles["Caption"],
        ))

    S(Paragraph("4.4&nbsp;&nbsp;&nbsp;Adaptive vs. Flat Sparsity", styles["SubSectionH"]))

    t2_data = [
        ["Model", "Flat\nSweet Spot", "Adaptive\nAvg", "Relative\nGain", "Quality\nFloor"],
        ["TinyLlama SwiGLU", "30%", "35.0%", "+16.7%", "cos \u2265 0.98"],
        ["TinyLlama ReLU", "60%", "75.4%", "+25.7%", "cos \u2265 0.98"],
        ["DeepSeek-R1 7B", "40%", "53.3%", "+33.3%", "cos \u2265 0.95"],
    ]
    col_w2 = [W * f for f in [0.22, 0.16, 0.16, 0.16, 0.16]]
    t2 = Table(t2_data, colWidths=col_w2, repeatRows=1)
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("LEADING", (0, 0), (-1, -1), 11),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#cccccc")),
        ("BACKGROUND", (0, 1), (-1, 1), HexColor("#f8f8ff")),
        ("BACKGROUND", (0, 2), (-1, 2), white),
        ("BACKGROUND", (0, 3), (-1, 3), HexColor("#eef0ff")),
        ("FONTNAME", (0, 3), (-1, 3), "Helvetica-Bold"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    S(Spacer(1, 4))
    S(t2)
    S(Paragraph(
        "<b>Table 2.</b> Adaptive vs. flat sparsity comparison. Flat sweet spot is the highest "
        "uniform sparsity achieving comparable quality. Relative gain increases with model "
        "size, reaching 33% on DeepSeek-R1 7B.",
        styles["Caption"],
    ))

    S(Paragraph("4.5&nbsp;&nbsp;&nbsp;End-to-End Generation: Reasoning Preservation", styles["SubSectionH"]))
    S(Paragraph(
        "To validate that FUSE preserves model capabilities beyond layer-level cosine "
        "similarity, I test DeepSeek-R1-Distill-Qwen-7B on a multi-step arithmetic task "
        "(\"What is 17 &times; 23? Think step by step.\") using the adaptive schedule at "
        "53.3% sparsity:",
        styles["Body"],
    ))

    t3_data = [
        ["Mode", "Output (truncated)"],
        ["Dense\n(baseline)",
         '17 * 20 is 340, and 17 * 3 is 51.\n'
         'So, 340 + 51 is 391.\nSo, 17 * 23 [= 391] \u2713'],
        ["FUSE\nDeepSeek\n53.3%",
         'Let me calculate 17 multiplied by 20 first,\n'
         'which is 340. Then, I need to add 17 multiplied\n'
         'by 3, which is 51. So, 340 plus 51 equals 391.\n'
         'Therefore, 17 * 23 is 391. \u2713\n'
         '(self-verification continues)'],
        ["FUSE\nQwen3.5-4B\n41.2%",
         '(10+7) x (20+3) = 200 + 30 + 140 + 21 = 391 \u2713\n'
         '(Full FOIL expansion method)'],
        ["FUSE\nQwen3.5-9B\n44.3%",
         'Break 23 into 20+3. 17*2=34, so 17*20=340.\n'
         '17*3=51. Then 340+51=391. \u2713\n'
         '(Textbook-quality step-by-step pedagogy)'],
    ]
    t3 = Table(t3_data, colWidths=[W * 0.14, W * 0.86], repeatRows=1)
    t3.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("FONTNAME", (1, 1), (1, -1), "Courier"),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#cccccc")),
        ("BACKGROUND", (0, 1), (-1, 1), HexColor("#f8f8ff")),
        ("BACKGROUND", (0, 2), (-1, 2), HexColor("#f0fff0")),
        ("BACKGROUND", (0, 3), (-1, 3), HexColor("#f0f8ff")),
        ("BACKGROUND", (0, 4), (-1, 4), HexColor("#f5f0ff")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    S(Spacer(1, 4))
    S(t3)
    S(Paragraph(
        "<b>Table 3.</b> Dense vs. FUSE sparse generation across three models. All arrive at "
        "the correct answer (391) but each uses a different solution strategy, demonstrating "
        "that FUSE preserves reasoning capability &mdash; not just surface-level token "
        "reproduction. DeepSeek self-verifies, Qwen3.5-4B uses FOIL expansion, Qwen3.5-9B "
        "provides textbook-quality pedagogy.",
        styles["Caption"],
    ))

    # ── Figure: Neurons fired ──
    fig_path = os.path.join(FIGURES_DIR, "neurons_fired.png")
    if os.path.exists(fig_path):
        S(Spacer(1, 6))
        S(Image(fig_path, width=W, height=W * 0.38, kind="proportional"))
        S(Paragraph(
            "<b>Figure 4.</b> Neurons fired vs skipped per layer across all models. "
            "Each skipped neuron represents W<sub>up</sub> and W<sub>down</sub> weight data "
            "that never leaves storage.",
            styles["Caption"],
        ))

    # ══════════════════════════════════════════════════════════
    #  5. PROJECTED IMPACT
    # ══════════════════════════════════════════════════════════
    S(Paragraph("5&nbsp;&nbsp;&nbsp;Projected Impact for Large Models", styles["SectionH"]))
    S(Paragraph(
        "FUSE's value proposition scales with model size, because larger models have more "
        "FFN redundancy and are more I/O-bound. I project savings for LLaMA-scale models "
        "using measured sparsity rates and standard storage bandwidths:",
        styles["Body"],
    ))

    t4_data = [
        ["Model", "FFN\nSize", "W_gate\nin RAM", "Sparsity\nScenario", "I/O per\nToken", "NVMe\nGen4", "NVMe\nGen5"],
        ["LLaMA-7B", "9.0 GB", "3.0 GB", "SwiGLU 50%", "3.0 GB", "2.3 tok/s", "4.7 tok/s"],
        ["LLaMA-7B", "9.0 GB", "3.0 GB", "ReLU 90%", "0.6 GB", "11.7 tok/s", "23.3 tok/s"],
        ["LLaMA-70B", "90.5 GB", "30.2 GB", "SwiGLU 50%", "30.2 GB", "0.2 tok/s", "0.5 tok/s"],
        ["LLaMA-70B", "90.5 GB", "30.2 GB", "ReLU 90%", "6.0 GB", "1.2 tok/s", "2.3 tok/s"],
    ]
    col_w4 = [W * f for f in [0.14, 0.10, 0.12, 0.14, 0.12, 0.14, 0.14]]
    t4 = Table(t4_data, colWidths=col_w4, repeatRows=1)
    t4.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#cccccc")),
        ("BACKGROUND", (0, 1), (-1, 2), HexColor("#f8f8ff")),
        ("BACKGROUND", (0, 3), (-1, 4), white),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    S(Spacer(1, 4))
    S(t4)
    S(Paragraph(
        "<b>Table 4.</b> Projected FUSE performance on LLaMA-scale models. W<sub>gate</sub> "
        "in RAM is 33% of FFN. I/O per token is the sparse read from W<sub>up</sub> + "
        "W<sub>down</sub>. NVMe rates assume sequential read from neuron-indexed storage. "
        "A ReLUfied LLaMA-70B at 90% sparsity could achieve >1 tok/s on a single NVMe Gen4 "
        "drive with 30 GB RAM for W<sub>gate</sub>.",
        styles["Caption"],
    ))

    # ── Figure: Memory savings ──
    fig_path = os.path.join(FIGURES_DIR, "memory_savings.png")
    if os.path.exists(fig_path):
        S(Spacer(1, 6))
        S(Image(fig_path, width=W, height=W * 0.38, kind="proportional"))
        S(Paragraph(
            "<b>Figure 5.</b> Left: full model size vs FUSE RAM requirement. A 70B model "
            "drops from 140 GB to ~35 GB. Right: largest model each device class can run "
            "with FUSE disk streaming, from IoT gateways (4 GB RAM &rarr; 4B models) to "
            "workstations (32 GB RAM &rarr; 70B models).",
            styles["Caption"],
        ))

    # ══════════════════════════════════════════════════════════
    #  6. DISCUSSION
    # ══════════════════════════════════════════════════════════
    S(Paragraph("6&nbsp;&nbsp;&nbsp;Discussion", styles["SectionH"]))

    S(Paragraph("6.1&nbsp;&nbsp;&nbsp;Interpretability Implications", styles["SubSectionH"]))
    S(Paragraph(
        "The per-layer sparsity profiles produced by FUSE calibration reveal interpretable "
        "structure. In DeepSeek-R1 7B, layer 3's anomalous sensitivity (32% maximum sparsity "
        "vs. 91% for layer 1) suggests it serves as a critical bottleneck for establishing "
        "task representations. The inverted profile under ReLUfication (middle layers become "
        "most prunable) indicates that SiLU's soft gating masks redundancy that ReLU exposes. "
        "These profiles could complement mechanistic interpretability research by identifying "
        "functionally critical layers without manual probing.",
        styles["Body"],
    ))

    S(Paragraph("6.2&nbsp;&nbsp;&nbsp;Limitations", styles["SubSectionH"]))
    S(Paragraph(
        "My current evaluation uses cosine similarity and single-task generation rather than "
        "comprehensive benchmarks (MMLU, GSM8K, HumanEval). While cosine similarity at the "
        "layer level is a strong proxy for output quality, benchmark scores would strengthen "
        "the claims. The token-by-token loop in the inference engine, while correct, does not "
        "achieve wall-clock speedups in the current RAM-resident implementation &mdash; actual "
        "speedups require the disk-streaming backend. Additionally, calibration on 5 sentences "
        "(~170 tokens) may not capture all activation patterns; scaling to larger calibration "
        "sets could refine the schedules further.",
        styles["Body"],
    ))

    S(Paragraph("6.3&nbsp;&nbsp;&nbsp;Complementarity with Other Techniques", styles["SubSectionH"]))
    S(Paragraph(
        "FUSE is orthogonal to quantization: the sparse neurons can be loaded in any precision "
        "(INT4, FP8, FP16). Combining FUSE with 4-bit quantization would compound the savings "
        "&mdash; e.g., 50% sparsity + 4x quantization = 8x total I/O reduction. FUSE also "
        "composes with KV-cache compression and speculative decoding, as it only modifies the "
        "FFN forward pass.",
        styles["Body"],
    ))

    # ══════════════════════════════════════════════════════════
    #  7. CONCLUSION
    # ══════════════════════════════════════════════════════════
    S(Paragraph("7&nbsp;&nbsp;&nbsp;Conclusion", styles["SectionH"]))
    S(Paragraph(
        "FUSE demonstrates that SwiGLU models contain their own activation predictor in the "
        "gate projection, eliminating the need for trained predictor networks in sparse LLM "
        "inference. My per-layer adaptive calibration algorithm exploits the heterogeneous "
        "sensitivity across layers, achieving 53.3% sparsity on DeepSeek-R1 7B (2.1x I/O "
        "reduction), 75.4% on ReLUfied TinyLlama (4.1x I/O reduction), and 44.3% on "
        "Qwen3.5-9B (1.8x I/O reduction) &mdash; all with preserved multi-step reasoning. "
        "Validation across five models spanning three architecture families (LLaMA, Qwen2, "
        "Qwen3.5 with hybrid Gated DeltaNet) with zero code changes demonstrates that FUSE "
        "generalizes across SwiGLU-based architectures. The distinct layer sensitivity profiles "
        "uncovered by calibration &mdash; U-shape (TinyLlama), cliff (DeepSeek-R1), descending "
        "slope (Qwen3.5) &mdash; provide interpretable architectural fingerprints that may "
        "complement mechanistic interpretability research. The approach requires no training, "
        "no quantization, and no architecture changes &mdash; only a one-time calibration pass "
        "that completes in minutes. Combined with neuron-indexed storage for sequential disk "
        "reads, FUSE provides a practical path to running models far larger than available RAM "
        "on consumer NVMe hardware.",
        styles["Body"],
    ))

    S(Spacer(1, 12))
    S(HRFlowable(width="100%", thickness=0.5, color=HexColor("#cccccc"),
                  spaceAfter=10))

    # ── References ──
    S(Paragraph("References", styles["SectionH"]))

    refs = [
        "Alizadeh, K. et al. (2024). LLM in a Flash: Efficient Large Language Model "
        "Inference with Limited Memory. <i>arXiv:2312.11514</i>.",

        "Liu, Z. et al. (2023). Deja Vu: Contextual Sparsity for Efficient LLMs at "
        "Inference Time. <i>ICML 2023</i>.",

        "Lee, J. et al. (2024). CATS: Contextually-Aware Thresholding for Sparsity in "
        "Large Language Models. <i>arXiv:2404.08763</i>.",

        "Mirzadeh, I. et al. (2024). ReLU Strikes Back: Exploiting Activation Sparsity "
        "in Large Language Models. <i>ICLR 2024</i>.",

        "Song, Y. et al. (2024). ProSparse: Introducing and Enhancing Intrinsic "
        "Activation Sparsity within Large Language Models. <i>arXiv:2402.13516</i>.",

        "Song, Y. et al. (2024). TurboSparse: Achieving LLM SOTA Performance with "
        "Minimal Activated Parameters. <i>arXiv:2406.05955</i>.",

        "Xue, Y. et al. (2024). PowerInfer: Fast Large Language Model Serving with a "
        "Consumer-grade GPU. <i>arXiv:2312.12456</i>.",

        "Zhang, M. &amp; Chen, B. (2024). TinyLlama: An Open-Source Small Language "
        "Model. <i>arXiv:2401.02385</i>.",
    ]
    for r in refs:
        S(Paragraph(r, ParagraphStyle(
            "Ref", parent=styles["Normal"], fontSize=8, leading=10.5,
            spaceAfter=4, leftIndent=18, firstLineIndent=-18,
            textColor=HexColor("#333333"),
        )))

    # ── Build ──
    doc.build(story)
    print(f"  Paper generated: {OUTPUT}")


if __name__ == "__main__":
    build_doc()
