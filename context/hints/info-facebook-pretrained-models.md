Here’s a **curated list of the major Meta / Facebook Research vision(-ish) models from 2021 onward** that are *explicitly positioned as general-purpose backbones / foundation encoders* (for 2D, video, or 3D).

I’ll group them by year and modality.

---

## 2021

### 1. DINO – self-supervised ViT backbone (ICCV 2021)

* **Paper:** *Emerging Properties in Self-Supervised Vision Transformers* (a.k.a. DINO).
* **What it is:** Self-distillation on ViT/DeiT that yields *very strong visual features* usable directly with k-NN / linear heads and good emergent segmentation. ([Medium][1])
* **Backbone role:** Pretrained ViT-S / ViT-B weights from `facebookresearch/dino` are widely used as generic image encoders in detection, segmentation, retrieval, etc.

---

### 2. MAE – Masked Autoencoders (CVPR 2022, arXiv 2021)

* **Paper:** *Masked Autoencoders Are Scalable Vision Learners* (He et al., FAIR). ([arXiv][2])
* **Code:** `facebookresearch/mae`. ([GitHub][3])
* **What it is:** ViT encoder + lightweight decoder trained to reconstruct heavily masked images.
* **Backbone role:** You typically **throw away the decoder** and use the MAE-pretrained ViT encoder as a generic backbone for classification, detection, and segmentation; it’s one of the canonical MIM pretrainers.

---

## 2022

### 3. MViT / MViTv2 – Multiscale Vision Transformers

* **Paper:** *Multiscale Vision Transformers* (ICCV 2021) and *MViTv2: Improved Multiscale Vision Transformers* (CVPR 2022). ([arXiv][4])
* **Code:** `facebookresearch/mvit`. The repo explicitly describes MViT as “a multiscale transformer which serves as a general vision backbone for different visual recognition tasks” (image, detection, instance seg, video). ([GitHub][5])
* **Backbone role:** Hierarchical transformer backbone plugged into Detectron2, PySlowFast, etc., for image and video tasks.

---

### 4. ConvNeXt – modern CNN backbone (CVPR 2022)

* **Paper:** *A ConvNet for the 2020s* (a.k.a. ConvNeXt).
* **Code:** `facebookresearch/ConvNeXt`. ([GitHub][6])
* **What it is:** A “modernized ResNet” CNN (large depthwise convs, LayerNorm, etc.) that matches or beats ViTs on ImageNet while staying fully convolutional. ([ResearchGate][7])
* **Backbone role:** Extremely common backbone in detection/seg frameworks (`timm`, Detectron2, MMSeg); many papers and repos treat “ConvNeXt-*” as drop-in replacements for ResNet backbones. ([Kaggle][8])

---

### 5. data2vec (vision) – modality-general SSL targets

* **Paper:** *data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language* (ICML 2022). ([arXiv][9])
* **Blog:** Meta: “a framework that uses the same learning method for either speech, NLP or computer vision.” ([AI Meta][10])
* **Backbone role:** Produces ViT-based **data2vec-vision** checkpoints (e.g. on HF under `facebook/data2vec-vision-*`) that serve as generic image backbones trained with contextualized target prediction instead of pixel/feature reconstruction. ([Hugging Face][11])

---

### 6. FLAVA – vision–language foundation model (CVPR 2022)

* **Paper / page:** *FLAVA: A Foundational Language And Vision Alignment Model* (arXiv 2021, CVPR 2022). ([arXiv][12])
* **Meta blog:** Described as a “single holistic universal model” targeting **vision, language, and multimodal tasks** at once. ([research.facebook.com][13])
* **Backbone role:** Contains **image encoder**, **text encoder**, and **multimodal encoder**; the image encoder can be used as a pretrained backbone for pure vision tasks (classification, retrieval, etc.) in addition to V+L.

---

### 7. MaskFeat – masked feature prediction (CVPR 2022)

* **Paper:** *Masked Feature Prediction for Self-Supervised Visual Pre-Training* (Wei et al., Facebook). ([CVF Open Access][14])
* **Code:** Implemented as a project in `facebookresearch/SlowFast` (MaskFeat + MViT). ([GitHub][15])
* **Backbone role:** Gives strong **MViT video backbones** via masked prediction of HOG-like features, widely cited in video SSL and video foundation model surveys. ([ResearchGate][16])

---

### 8. HRViT – high-resolution transformer backbone (segmentation)

* **Paper:** *Multi-Scale High-Resolution Vision Transformer for Semantic Segmentation* (CVPR 2022). ([arXiv][17])
* **Code:** `facebookresearch/HRViT`, described as “a new vision transformer backbone design for semantic segmentation” with multi-branch high-resolution architecture. ([GitHub][18])
* **Backbone role:** Used as a segmentation backbone analogous to HRNet but transformer-based; plugged into various semantic segmentation benchmarks (ADE20K, Cityscapes).

---

### 9. Mask2Former – universal segmentation transformer

* **Paper:** *Masked-attention Mask Transformer for Universal Image Segmentation* (CVPR 2022). ([GitHub][19])
* **Code:** `facebookresearch/Mask2Former`.
* **Backbone role:** Technically a **full segmentation head + decoder**, but the repo and paper treat Mask2Former as a **generic segmentation architecture that can sit on top of many backbones** (Swin, ConvNeXt, etc.), and is standard for semantic / instance / panoptic segmentation.

---

### 10. Omnivore – unified image / video / single-view 3D model

* **Paper:** *Omnivore: A Single Model for Many Visual Modalities* (CVPR 2022). ([arXiv][20])
* **Code:** `facebookresearch/omnivore`. Repo describes it as a model that can classify **images, videos, and single-view 3D** with one shared transformer trunk. ([GitHub][21])
* **Backbone role:** You can treat the shared trunk as a multimodal backbone for RGB, video, and depth-like inputs.

---

## 2023

### 11. ConvNeXt V2 – MAE-pretrained ConvNeXt

* **Paper:** *ConvNeXt V2: Co-Designing and Scaling ConvNets With Masked Autoencoders* (CVPR 2023). ([CVF Open Access][22])
* **Code:** `facebookresearch/ConvNeXt-V2`. ([GitHub][23])
* **Backbone role:** Strong ConvNeXt backbones pre-trained with a fully convolutional MAE variant; improved accuracy vs ConvNeXt v1, used for classification and dense prediction.

---

### 12. OmniMAE – single model MAE for images + videos

* **Paper:** *OmniMAE: Single Model Masked Pretraining on Images and Videos* (CVPR 2023). ([CVF Open Access][24])
* **Backbone role:** ViT backbone trained jointly on image and video MIM, producing a **unified image+video encoder** that can be fine-tuned for both domains; often cited as a **video foundation backbone**. ([arXiv][25])

---

### 13. DINOv2 – vision foundation model / backbone family

* **Paper / blog:** *DINOv2: Learning Robust Visual Features with Self-Supervision* (2023). ([arXiv][26])
* **Code:** `facebookresearch/dinov2`. ([GitHub][27])
* **Backbone role:**

  * Meta describes DINOv2 models as **“high-performance visual features that can be directly employed … on a variety of computer vision tasks”** and **“general, multipurpose backbones”**. ([GitHub][27])
  * Widely used as frozen encoders or fine-tuned backbones for classification, segmentation, depth, and more.

---

### 14. Segment Anything Model (SAM) – promptable segmentation

* **Paper:** *Segment Anything* (ICCV 2023). ([CVF Open Access][28])
* **Code:** `facebookresearch/segment-anything`, with checkpoints like `facebook/sam-vit-base`. ([GitHub][29])
* **Backbone role:** SAM itself is a **full segmentation foundation model**, but internally it uses a large ViT image encoder. People increasingly repurpose that encoder as a **strong vision backbone**, or use SAM as a teacher for lighter segmentation backbones. ([encord.com][30])

---

### 15. ImageBind – multimodal backbone (6 modalities)

* **Paper:** *ImageBind: One Embedding Space To Bind Them All* (2023). ([arXiv][31])
* **Code / page:** `facebookresearch/ImageBind`, Meta AI ImageBind site. ([GitHub][32])
* **What it is:** Joint embedding model for **images, text, audio, depth, thermal, and IMU**, using images as the central modality.
* **Backbone role:** The **image encoder** is effectively a powerful vision backbone that can be used alone, or in cross-modal setups (e.g., grounding, retrieval, generation).

---

## 2024

### 16. SAM 2 – video-capable promptable segmentation

* **Paper:** *Segment Anything Model 2 (SAM 2): Segment Anything in Images and Videos* (2024). ([arXiv][33])
* **Backbone role:** Extends SAM to **streaming video** with transformer + memory design. The underlying image/video encoder remains a strong visual backbone, now tuned for temporal data.

---

## 2025

### 17. VGGT – Visual Geometry Grounded Transformer (3D backbone)

* **Paper:** *VGGT: Visual Geometry Grounded Transformer* (CVPR 2025 Best Paper). ([arXiv][34])
* **Site / code:** vgg-t.github.io and `facebookresearch/vggt`. ([VGGT][35])
* **What it is:** A large feed-forward transformer that, from 1–100s of views, **directly predicts cameras, depth maps, point maps, and 3D tracks** in a single forward pass. ([arXiv][36])
* **Backbone role:** The paper explicitly shows that using **VGGT as a pretrained feature backbone significantly enhances downstream 3D tasks** like non-rigid point tracking and feed-forward novel view synthesis; it’s widely treated as a **3D vision foundation backbone**. ([arXiv][34])

---

### 18. DINOv3 – latest large-scale vision backbone family

* **Meta page / repo:** `ai.meta.com/dinov3` and `facebookresearch/dinov3`. ([AI Meta][37])
* **What it is:** Third-gen **self-supervised vision foundation model**, scaling to ~7B-parameter ViTs trained on ~1.7B images. Provides global and dense features for classification, segmentation, depth, tracking, etc. ([encord.com][38])
* **Backbone role:** Meta explicitly markets DINOv3 checkpoints (ViT + ConvNeXt variants) as **universal vision backbones**, with separate “heads/adapters” for downstream tasks. ([AI Meta][37])

---

### 19. SAM 3 – concept-level “segment anything”

* **Meta research page:** “Segment Anything Model 3 (SAM 3)” as a unified model that detects, segments, and tracks objects in image & video given **concept prompts** (text / semantics). ([AI Meta][39])
* **Backbone role:** Like SAM/SAM2, SAM 3’s image/video encoder is a powerful vision backbone; concept-prompting and tracking ability make it a candidate teacher / feature extractor for many downstream segmentation and tracking tasks.

---

## How to think about them as backbones (very short)

If you’re picking **pretrained encoders** from this list for *general* use:

* **Pure 2D vision (classification / det / seg):**
  **ConvNeXtV2**, **DINOv2/DINOv3**, **MAE-ViT**, **MViTv2** are the workhorses.

* **Segmentation-heavy:**
  **HRViT** as a dedicated segmentation backbone, with **Mask2Former** as the universal head; **SAM/SAM2/SAM3** as teachers or feature sources.

* **Video + images:**
  **OmniMAE**, **Omnivore**, **MaskFeat+MViT**.

* **Vision–language / multimodal:**
  **FLAVA**, **ImageBind** (and DINOv2/v3 for pure vision).

* **3D / geometry-centric:**
  **VGGT** as a 3D backbone, especially for multi-view geometry and tracking.

If you tell me your **target tasks + hardware regime (GPU vs edge vs your NPU)** I can narrow this to a short “top 3” list with concrete checkpoint names (e.g. specific DINOv2/DINOv3 configs or ConvNeXt variants) that are most sensible to build on.

[1]: https://sh-tsang.medium.com/review-dino-emerging-properties-in-self-supervised-vision-transformers-cfddbb4d3549?utm_source=chatgpt.com "Review — DINO: Emerging Properties in Self-Supervised ..."
[2]: https://arxiv.org/abs/2111.06377?utm_source=chatgpt.com "Masked Autoencoders Are Scalable Vision Learners"
[3]: https://github.com/facebookresearch/mae?utm_source=chatgpt.com "GitHub - facebookresearch/mae: PyTorch implementation ..."
[4]: https://arxiv.org/abs/2104.11227?utm_source=chatgpt.com "Multiscale Vision Transformers"
[5]: https://github.com/facebookresearch/mvit?utm_source=chatgpt.com "Code Release for MViTv2 on Image Recognition."
[6]: https://github.com/facebookresearch/ConvNeXt?utm_source=chatgpt.com "Code release for ConvNeXt model"
[7]: https://www.researchgate.net/publication/396787830_A_Comprehensive_Review_of_ConvNeXt_Architecture_in_Image_Classification_Performance_Applications_and_Prospects?utm_source=chatgpt.com "(PDF) A Comprehensive Review of ConvNeXt Architecture ..."
[8]: https://www.kaggle.com/code/kongkaifeng/convnext-cnn-with-databricks-mlflow-mlops/notebook?utm_source=chatgpt.com "ConvNeXt CNN with Databricks MLflow | MLOps"
[9]: https://arxiv.org/abs/2202.03555?utm_source=chatgpt.com "data2vec: A General Framework for Self-supervised ..."
[10]: https://ai.meta.com/research/data2vec-a-general-framework-for-self-supervised-learning-in-speech-vision-and-language/?utm_source=chatgpt.com "Data2vec: A General Framework for Self-supervised ..."
[11]: https://huggingface.co/facebook/data2vec-vision-large-ft1k?utm_source=chatgpt.com "facebook/data2vec-vision-large-ft1k"
[12]: https://arxiv.org/abs/2112.04482?utm_source=chatgpt.com "FLAVA: A Foundational Language And Vision Alignment Model"
[13]: https://research.facebook.com/publications/flava-a-foundational-language-and-vision-alignment-model/?utm_source=chatgpt.com "FLAVA: A Foundational Language And Vision Alignment Model"
[14]: https://openaccess.thecvf.com/content/CVPR2022/papers/Wei_Masked_Feature_Prediction_for_Self-Supervised_Visual_Pre-Training_CVPR_2022_paper.pdf?utm_source=chatgpt.com "Masked Feature Prediction for Self-Supervised Visual Pre- ..."
[15]: https://github.com/facebookresearch/SlowFast/tree/main/projects/maskfeat?utm_source=chatgpt.com "MaskFeat"
[16]: https://www.researchgate.net/publication/357114598_Masked_Feature_Prediction_for_Self-Supervised_Visual_Pre-Training?utm_source=chatgpt.com "Masked Feature Prediction for Self-Supervised Visual Pre- ..."
[17]: https://arxiv.org/pdf/2211.08675?utm_source=chatgpt.com "XRBench: An Extended Reality (XR) Machine Learning ..."
[18]: https://github.com/facebookresearch/HRViT?utm_source=chatgpt.com "HRViT (\"Multi-Scale High-Resolution Vision Transformer ..."
[19]: https://github.com/facebookresearch/Mask2Former?utm_source=chatgpt.com "facebookresearch/Mask2Former: Code release for \" ..."
[20]: https://arxiv.org/pdf/2201.08377?utm_source=chatgpt.com "arXiv:2201.08377v1 [cs.CV] 20 Jan 2022"
[21]: https://github.com/facebookresearch/omnivore?utm_source=chatgpt.com "Omnivore: A Single Model for Many Visual Modalities"
[22]: https://openaccess.thecvf.com/content/CVPR2023/papers/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.pdf?utm_source=chatgpt.com "ConvNeXt V2: Co-Designing and Scaling ConvNets With ..."
[23]: https://github.com/facebookresearch/ConvNeXt-V2?utm_source=chatgpt.com "Code release for ConvNeXt V2 model"
[24]: https://openaccess.thecvf.com/content/CVPR2023/papers/Girdhar_OmniMAE_Single_Model_Masked_Pretraining_on_Images_and_Videos_CVPR_2023_paper.pdf?utm_source=chatgpt.com "OmniMAE: Single Model Masked Pretraining on Images and ..."
[25]: https://arxiv.org/html/2405.03770v1?utm_source=chatgpt.com "Foundation Models for Video Understanding: A Survey"
[26]: https://arxiv.org/pdf/2304.07193?utm_source=chatgpt.com "DINOv2"
[27]: https://github.com/facebookresearch/dinov2?utm_source=chatgpt.com "PyTorch code and models for the DINOv2 self-supervised ..."
[28]: https://openaccess.thecvf.com/content/ICCV2023/papers/Kirillov_Segment_Anything_ICCV_2023_paper.pdf?utm_source=chatgpt.com "Segment Anything - CVF Open Access"
[29]: https://github.com/facebookresearch/segment-anything?utm_source=chatgpt.com "facebookresearch/segment-anything"
[30]: https://encord.com/blog/segment-anything-model-explained/?utm_source=chatgpt.com "Meta AI's Segment Anything Model (SAM) Explained"
[31]: https://arxiv.org/abs/2305.05665?utm_source=chatgpt.com "ImageBind: One Embedding Space To Bind Them All"
[32]: https://github.com/facebookresearch/ImageBind?utm_source=chatgpt.com "ImageBind One Embedding Space to Bind Them All"
[33]: https://arxiv.org/abs/2408.00714?utm_source=chatgpt.com "SAM 2: Segment Anything in Images and Videos"
[34]: https://arxiv.org/abs/2503.11651?utm_source=chatgpt.com "VGGT: Visual Geometry Grounded Transformer"
[35]: https://vgg-t.github.io/?utm_source=chatgpt.com "VGGT: Visual Geometry Grounded Transformer"
[36]: https://arxiv.org/html/2503.11651v1?utm_source=chatgpt.com "VGGT: Visual Geometry Grounded Transformer"
[37]: https://ai.meta.com/dinov3/?utm_source=chatgpt.com "DINOv3"
[38]: https://encord.com/blog/dinov3-explained-scaling-self-supervised-vision-tr/?utm_source=chatgpt.com "DINOv3 Explained: Scaling Self-Supervised Vision ..."
[39]: https://ai.meta.com/research/publications/segment-anything/?utm_source=chatgpt.com "Segment Anything | Research"
