there are 3 papers in paper-source/, read their latex source, understand them, and then create a report about status quo of yolo low bit quantization research. 

The focus of the report should be on the following aspects:
- status quo of low bit quantization result of yolo models using ptq, list the current best results, tables, and interpretation
- status quo of low bit quantization result of yolo models using qat, list the current best results, tables, and interpretation
- difficulties and challenges of low bit quantization of yolo models, using ptq and qat, list them as bullet points, and provide explanations, and provide references to the papers, with blockquote citing the relevant parts

## Report Template

Use the following template to write the report. Replace all bracketed placeholders like `[... ]`. Keep tables consistent (same metrics/columns) across PTQ vs QAT.

### 0) Metadata
- **Title**: Status quo of YOLO low-bit quantization (PTQ + QAT)
- **Date**: [YYYY-MM-DD]
- **Scope**: [Which YOLO family/versions covered: e.g., YOLOv5/v8/v10/v11]
- **Report directory**: `<report-md-dir>/` (this report markdown + `<report-md-dir>/figures/`)
- **Papers reviewed (from `paper-source/`)** (one block per paper):
  - **[Paper short name]**
    - IEEE citation: [A. Author, B. Author, and C. Author, "Paper Title," Venue, year, pp. xx–yy, doi: ...]
    - arXiv: `https://arxiv.org/abs/[arxiv-id]` (or `N/A` if not on arXiv)
    - TeX source (this workspace):
      - Main: `paper-source/[paper-dir]/tex/[main-tex].tex`
      - Bib: `paper-source/[paper-dir]/tex/[refs].bib` (if present)
  - **[Example: gupta2024-oscillations]**
    - IEEE citation: [fill]
    - arXiv: `https://arxiv.org/abs/[fill]`
    - TeX source (this workspace):
      - Main: `paper-source/gupta2024-oscillations/tex/od-qat.tex`
      - Bib: `paper-source/gupta2024-oscillations/tex/od-qat.bib`
  - **[Example: nagel2022-oscillations]**
    - IEEE citation: [fill]
    - arXiv: `https://arxiv.org/abs/[fill]`
    - TeX source (this workspace):
      - Main: `paper-source/nagel2022-oscillations/tex/main.tex`
      - Bib: `paper-source/nagel2022-oscillations/tex/dirty.bib`
  - **[Example: qyolo-2023]**
    - IEEE citation: [fill]
    - arXiv: `https://arxiv.org/abs/[fill]`
    - TeX source (this workspace):
      - Main: `paper-source/qyolo-2023/tex/paper.tex`
      - Bib: `paper-source/qyolo-2023/tex/references.bib`
- **Primary evaluation settings assumed in this report**:
  - Dataset(s): [COCO / VOC / custom]
  - Metric(s): [mAP@0.5:0.95, mAP@0.5, AP50, etc.]
  - Image size / preprocessing: [e.g., 640]
  - Baseline FP32 reference: [which model/checkpoint and its metrics]

### 1) Executive summary (1 page max)
- **Best reported PTQ result (headline)**: [W/A bits, model, dataset, metric]
- **Best reported QAT result (headline)**: [W/A bits, model, dataset, metric]
- **Key takeaways (3–7 bullets)**:
  - [Takeaway]
  - [Takeaway]
- **What’s still unclear / not comparable across papers (3–7 bullets)**:
  - [Protocol mismatch]
  - [Metric mismatch]

### 2) Definitions and comparability notes
- **Bit-width notation**: [e.g., W4A8, W4A16, per-channel/per-tensor, symmetric/asymmetric]
- **Quantization scope**: [weights only / weights+activations / first+last layer treatment]
- **Calibration protocol (PTQ)**: [data size, selection, augmentations, percentile/entropy, etc.]
- **Training protocol (QAT)**: [epochs, LR schedule, distillation, freezing, EMA, etc.]
- **Hardware / kernel availability assumptions**: [INT8 kernels? mixed precision? deployment target?]
- **Important comparability caveats**:
  - [E.g., different NMS / postprocess changes AP]
  - [E.g., model variants not matched]

### 3) PTQ status quo (results + interpretation)
#### 3.1 Snapshot table (best/representative PTQ results)
Fill in one row per “best result per paper per model family” (avoid dumping every ablation here; put them in Appendix).

| Paper | Model | Dataset | Metric | Image size | W bits | A bits | Quant scheme | Calibration | Reported accuracy | Δ vs FP32 | Notes |
|---|---|---|---|---:|---:|---:|---|---|---:|---:|---|
| [Paper A] | [yolo...] | [COCO] | [mAP@0.5:0.95] | [640] | [4] | [8/16] | [per-channel, sym] | [n images] | [xx.x] | [-x.x] | [notes] |

#### 3.2 Interpretation
- **Per-paper methods (PTQ)** (one block per reviewed paper):
  - **[Paper short name] — method overview**:
    - Problem targeted: [what PTQ failure it addresses]
    - Core idea: [1–3 sentences describing the method]
    - Where it changes YOLO: [backbone/neck/head, first/last layers, postprocess, etc.]
    - Quantization details: [W/A bits, scheme, granularity, observers/calibration]
    - Evidence: [point to the specific table/figure/section used for your PTQ table row]
  - **[Paper short name] — limitations**:
    - Stated limitations: [limitations explicitly mentioned by the authors]
    - Observed limitations: [limitations you infer from results/ablations; label as inference]
    - Key figures explaining limitations/difficulties (if any):
      - Place the extracted figure(s) under `<report-md-dir>/figures/` and insert them here.
      - Prefer `.png` or `.svg`. If the paper figure is `.pdf`, convert first.
      - Insert format:
        - `![{caption} (from [Paper short name], Fig. X)](figures/{paper_short}_figX.png)`
      - Conversion examples (pick what’s available on your system):
        - `pdftocairo -png -r 300 input.pdf figures/{paper_short}_figX` (then rename `...-1.png`)
        - `pdf2svg input.pdf figures/{paper_short}_figX.svg`
    - Comparability issues: [what prevents direct comparison vs other papers]
    - Deployment caveats: [kernels/hardware assumptions, latency not reported, etc.]
- **What works best across papers (PTQ)**:
  - [Synthesize the recurring ingredients that help, and in which conditions]
- **Common failure modes (PTQ)**:
  - [E.g., small-object AP drop, regression instability, activation outliers]
- **Apples-to-apples comparisons**:
  - [Explicitly state which rows are comparable and why]

#### 3.3 Evidence excerpts (PTQ)
Include blockquotes for the most important claims (results, protocols, limitations).
> [Quote supporting PTQ setup/claim, <=25 words]  
> — [Paper A], [section/subsection], [latex file name if helpful]

### 4) QAT status quo (results + interpretation)
#### 4.1 Snapshot table (best/representative QAT results)

| Paper | Model | Dataset | Metric | Image size | W bits | A bits | Quant scheme | QAT recipe highlights | Reported accuracy | Δ vs FP32 | Notes |
|---|---|---|---|---:|---:|---:|---|---|---:|---:|---|
| [Paper B] | [yolo...] | [COCO] | [mAP@0.5:0.95] | [640] | [4] | [8/16] | [per-channel, sym] | [distill + ...] | [xx.x] | [-x.x] | [notes] |

#### 4.2 Interpretation
- **Per-paper methods (QAT)** (one block per reviewed paper):
  - **[Paper short name] — method overview**:
    - Problem targeted: [what QAT failure it addresses]
    - Core idea: [1–3 sentences describing the method]
    - Where it changes YOLO: [backbone/neck/head, loss, postprocess, etc.]
    - QAT recipe: [fake-quant placement, observers, LR/epochs, distillation, freezing, EMA, etc.]
    - Evidence: [point to the specific table/figure/section used for your QAT table row]
  - **[Paper short name] — limitations**:
    - Stated limitations: [limitations explicitly mentioned by the authors]
    - Observed limitations: [limitations you infer from results/ablations; label as inference]
    - Key figures explaining limitations/difficulties (if any):
      - Place the extracted figure(s) under `<report-md-dir>/figures/` and insert them here.
      - Prefer `.png` or `.svg`. If the paper figure is `.pdf`, convert first.
      - Insert format:
        - `![{caption} (from [Paper short name], Fig. X)](figures/{paper_short}_figX.png)`
      - Conversion examples (pick what’s available on your system):
        - `pdftocairo -png -r 300 input.pdf figures/{paper_short}_figX` (then rename `...-1.png`)
        - `pdf2svg input.pdf figures/{paper_short}_figX.svg`
    - Comparability issues: [what prevents direct comparison vs other papers]
    - Training / compute caveats: [extra epochs, stability issues, data scale requirements]
- **What works best across papers (QAT)**:
  - [Synthesize the recurring ingredients that help, and in which conditions]
- **Key trade-offs (QAT)**:
  - [Compute/training time, stability, data needs]
- **Where QAT still struggles**:
  - [E.g., W2/W3, A4, detection head sensitivity]

#### 4.3 Evidence excerpts (QAT)
> [Quote supporting QAT setup/claim, <=25 words]  
> — [Paper B], [section/subsection], [latex file name if helpful]

### 5) Challenges & open problems (with cited evidence)
List challenges as bullet points, and include at least one supporting excerpt per major challenge.
- **[Challenge title]**: [Explain why it happens and how papers address it (or not).]
  - Evidence:
    > [Quote, <=25 words]  
    > — [Paper C], [section/subsection]
  - Key figures (if any):
    - If the paper includes a figure that illustrates this challenge, extract it, convert to `.png`/`.svg` if needed, save to `<report-md-dir>/figures/`, and insert it:
      - `![{caption} (from [Paper short name], Fig. X)](figures/{paper_short}_figX.png)`
- **[Challenge title]**: [...]

### 6) Practical recommendations (actionable)
- **For PTQ experiments in this repo**:
  - [What to try first, what to log, what to ablate]
- **For QAT experiments in this repo**:
  - [Suggested starting recipes and guardrails]
- **Minimum reporting checklist (for future runs)**:
  - [Exact model/version + dataset split]
  - [Metric + image size + postprocess]
  - [Quantization scheme and layer exceptions]
  - [Calibration/training details]
  - [Speed/latency and deployment constraints]

### 7) References
- [Paper A full title], [venue/year]. Source: `paper-source/...`
- [Paper B full title], [venue/year]. Source: `paper-source/...`
- [Paper C full title], [venue/year]. Source: `paper-source/...`
