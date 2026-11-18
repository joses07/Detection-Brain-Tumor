Bayesian-Inferred Brain Tumor Classification: Combining MC Dropout with Explainable AI for Uncertainty-Aware Decision Support

Official repository for the research paper: "Bayesian-Inferred Brain Tumor Classification: Combining MC Dropout with Explainable AI for Uncertainty-Aware Decision Support" by Cornelius Joses Javier Suyanto and Hartono (Department of Mathematics, Faculty of Science and Technology, Sanata Dharma University, Yogyakarta, Indonesia).

📌 Overview

Reliable brain tumor classification from Magnetic Resonance Imaging (MRI) requires more than just high accuracy; it demands trustworthy confidence estimates and transparent visual explanations. Conventional deterministic Convolutional Neural Networks (CNNs) often produce overconfident predictions on ambiguous or out-of-distribution scans.

This framework addresses these challenges by integrating:

Bayesian Inference via Monte Carlo (MC) Dropout: Implemented in the classification head to quantify predictive uncertainty (decomposed into aleatoric and epistemic components) across $T=50$ stochastic forward passes.

Multi-Criteria Decision Analysis (MCDA): A weighted ranking protocol evaluating six ImageNet-pretrained backbones across Clinical Performance ($w_1 = 0.50$), Calibration Quality ($w_2 = 0.30$), and Selective Prediction / Rejection Capability ($w_3 = 0.20$) over 30 random seeds.

Uncertainty-Aware Selective Prediction: Utilizing validation-selected epistemic uncertainty (Mutual Information) thresholds to defer ambiguous cases for human expert review.

Spatially-Aware Explainable AI (XAI): Computing Mean and Standard Deviation (Std) Grad-CAM heatmaps across MC samples to visualize both consistent model attention and spatial saliency variance.

⚠️ Clinical Disclaimer: Evidence in this study comes from a single public dataset without verifiable patient-level separation, external validation, or radiologist-annotated masks. This framework is intended as a research prototype for human-in-the-loop decision support, not a clinically validated diagnostic tool.

🏗️ Methodology & Pipeline

Dataset & Preprocessing:

Uses the Brain Tumor MRI Dataset (Kaggle) comprising 7,023 images across four classes: Glioma, Meningioma, Pituitary, and No Tumor.

Stratified 80:20 split on the training folder (4,570 train / 1,142 validation) and an independent held-out test folder (1,311 images).

Offline geometric and photometric augmentation applied strictly to the training set (expanding it to 18,280 images), resized to $224 \times 224$ pixels, and normalized using ImageNet statistics.

Hyperparameter Optimization (Optuna):

25 trials per backbone using a median pruner ($T=10$ MC passes on validation) tuning MC dropout rate, learning rate, weight decay, batch size, optimizer (AdamW / SGD), and epochs.

Multi-Seed Statistical Evaluation:

30 independent runs (fixed identical seeds across all 6 backbones) evaluated with $T=50$ MC forward passes.

Evaluated via Shapiro-Wilk normality tests, Friedman omnibus test, and Holm-adjusted pairwise Wilcoxon signed-rank tests.

MCDA Ranking & Selective Prediction:

Ranks architectures using Min-Max normalized metrics across 3 clinical pillars.

Applies epistemic uncertainty thresholds tuned on validation data to defer high-uncertainty scans.

📊 Key Experimental Results

1. Backbone Comparison & MCDA Ranking (30 Seeds, $T=50$)

While non-parametric tests showed no statistically significant accuracy difference among the top three backbones (ResNet18, DenseNet121, and MobileNetV2; Holm-adjusted $p \ge 0.70$), ResNet18 ranked 1st overall in the MCDA composite score:

Rank

Architecture

Test Accuracy

Macro ROC-AUC

Test ECE

Test NLL

Test AURC

MCDA $S_{\text{Total}}$

1

ResNet18

0.9916 ± 0.0019

0.9998 ± 0.0001

0.0063 ± 0.0016

0.0334 ± 0.0087

0.0003 ± 0.0002

0.9311

2

DenseNet121

0.9910 ± 0.0027

0.9997 ± 0.0002

0.0057 ± 0.0016

0.0325 ± 0.0072

0.0005 ± 0.0003

0.9058

3

MobileNetV2

0.9913 ± 0.0027

0.9995 ± 0.0002

0.0068 ± 0.0021

0.0400 ± 0.0120

0.0006 ± 0.0005

0.8125

4

GoogLeNet

0.9867 ± 0.0067

0.9994 ± 0.0006

0.0089 ± 0.0042

0.0527 ± 0.0282

0.0009 ± 0.0010

0.5718

5

VGG19

0.9834 ± 0.0074

0.9989 ± 0.0007

0.0118 ± 0.0052

0.0787 ± 0.0301

0.0016 ± 0.0012

0.1771

6

AlexNet

0.9796 ± 0.0073

0.9988 ± 0.0004

0.0103 ± 0.0034

0.0759 ± 0.0264

0.0016 ± 0.0008

0.0965

2. Deterministic vs. MC Dropout Inference (ResNet18)

MC Dropout preserves discriminative accuracy while achieving statistically significant calibration improvements (paired Wilcoxon $p < 0.001$):

ECE: $0.0072 \rightarrow \mathbf{0.0063}$

NLL: $0.0376 \rightarrow \mathbf{0.0334}$

Brier Score: $0.0145 \rightarrow \mathbf{0.0142}$

Convergence: Practical stability for accuracy, entropy, variance, and ECE is achieved within $T = 10\text{–}20$ forward passes.

3. Selective Prediction Performance (ResNet18)

Using validation-selected Mutual Information thresholds (95% target coverage):

Retained Test Coverage: $96.16\% \pm 0.62\%$ (only ~3.8% of scans deferred for expert review).

Selective Accuracy on Retained Scans: Improved from $99.16\%$ to $99.82\%$.

Error Capture Rate: Deferred scans captured $78.75\% \pm 10.94\%$ of all test misclassifications.
