# Adaptive NEON: Mitigating Model Collapse via Block-Wise Extrapolation

**Final Project - Deep Learning** **Department of Industrial Engineering and Management, Ben-Gurion University of the Negev**

**Authors:** Nir Soffer ([nirsof@post.bgu.ac.il](mailto:nirsof@post.bgu.ac.il))  
Amitay Granada ([granada@post.bgu.ac.il](mailto:granada@post.bgu.ac.il))

---

## 📌 Overview
This project addresses the phenomenon of **Model Collapse** in generative AI—a degenerative process where models trained on synthetic data progressively lose diversity and quality.

Building upon the **NEON** framework (Alemohammad et al., 2025), which proposes "Negative Extrapolation" to reverse this drift, we introduce a novel **Block-Wise Adaptive Extrapolation** approach. Instead of applying a single global correction scalar, we analyze the U-Net architecture as a hierarchical structure and apply distinct extrapolation weights to functional blocks (Encoder, Middle, Decoder).

Our experiments on a Toy Problem and CIFAR-10 (using EDM) demonstrate that this granular approach significantly outperforms global baselines in recovering the original data distribution.

## 🧠 The Problem: Model Collapse
As generative models (like Stable Diffusion or GPT) flood the web with synthetic data, future models are inevitably trained on this output. This creates a self-consuming loop where:
1.  **Tail Truncation:** Rare examples are ignored.
2.  **Mode Seeking:** The model converges to a homogenized mean.
3.  **Quality Degradation:** Artifacts are amplified.

## 💡 Our Solution: Block-Wise Adaptive NEON
The original NEON paper suggests correcting the weights using:
$$\theta_{new} = \theta_{base} - w \cdot (\theta_{collapsed} - \theta_{base})$$

We hypothesize that different layers collapse at different rates. We decompose the U-Net into three semantic blocks based on resolution:
* **Encoder:** Downsampling layers ($32 \times 32 \to 16 \times 16$).
* **Middle:** The semantic bottleneck ($8 \times 8$).
* **Decoder:** Upsampling layers (Texture and fine details).

We optimize a vector of weights $W = [w_{enc}, w_{mid}, w_{dec}]$ to achieve superior restoration.

---

## 📊 Key Results

### 1. Toy Problem (2D Gaussian)
We simulated model collapse on a simple MLP learning a 2D Gaussian distribution.
* **Global NEON Improvement:** ~12%
* **Block-Wise NEON Improvement:** **~73%** (using layer-specific weights).

### 2. Real-World Application (EDM on CIFAR-10)
Using Elucidated Diffusion Models (EDM), we performed a Grid Search over block weights.
* **Baseline NEON FID:** ~1.43
* **Adaptive NEON FID:** **< 1.43** (We observed that the Middle block often requires different magnitude correction than the Decoder to maximize diversity without harming precision).

---

## 📂 Repository Structure

```text
├── toy_problem.py        # Code for the 2D Gaussian experiment and visualization
├── edm_sampler.py        # Modified EDM sampler supporting block-wise extrapolation
├── grid_search.py        # Script for optimizing weights over the validation set
├── generate.py           # Main generation script
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
