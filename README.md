# 🤖 Transformer 101: The Annotated Transformer Explained

An interactive, visual, and code-level deep dive into the Transformer architecture based on the seminal paper [*Attention Is All You Need* (Vaswani et al.)](https://arxiv.org/abs/1706.03762) and Harvard NLP's *The Annotated Transformer*.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-blue?style=for-the-badge&logo=github)](https://murtuzasaifee.github.io/transformer_101/annotated_transformer_blueprint.html)

> 🌐 **Interactive Web Guide:** Explore the visual companion directly in your browser:  
> **[👉 View The Annotated Transformer Blueprint](https://murtuzasaifee.github.io/transformer_101/annotated_transformer_blueprint.html)**

---

## 🌟 Overview

This repository provides a comprehensive, beginner-to-advanced guide to understanding, visualizing, and implementing the Transformer model from scratch using PyTorch.

It breaks down every component—from input embeddings and positional encodings to multi-head attention, encoder-decoder stacks, layer normalization, residual connections, and training loops—into clear mathematical intuitions, tensor shape walkthroughs, and runnable code.

---

## 📂 Repository Structure

```text
transformer_101/
├── annotated_transformer_blueprint.html  # Interactive visual blueprint & guide
├── notebooks/
│   └── AnnotatedTransformer.ipynb        # Step-by-step PyTorch implementation with extended explanations
├── LICENSE                               # License details
└── README.md                             # Project documentation
```

---

## 🚀 Key Highlights

### 1. 🎨 Interactive Visual Blueprint ([Live Demo](https://murtuzasaifee.github.io/transformer_101/annotated_transformer_blueprint.html))
An interactive HTML companion that provides:
- **Architecture Breakdown**: Side-by-side explanations of the Encoder and Decoder stacks.
- **Multi-Head Attention Demystified**: Visualizing Query ($Q$), Key ($K$), and Value ($V$) projections, scaled dot-product attention, and causal masking.
- **Tensor Shape Flow**: Dimension tracking through every transformation layer (`[batch_size, seq_len, d_model]`).
- **Chapter Navigation**: Tabbed interface covering architecture intuition, sub-layers, training dynamics, and inference.

### 2. 📓 Annotated PyTorch Notebooks (`notebooks/`)
- Complete PyTorch implementation without external black-box libraries.
- Modular code structure: `Encoder`, `Decoder`, `MultiHeadedAttention`, `PositionalEncoding`, `Generator`, and `LayerNorm`.
- Synthetic task demonstrations (e.g., sequence copying task) illustrating loss curves, greedy decoding, and beam search intuition.

---

## 🛠️ Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed along with PyTorch and common scientific computing libraries:

```bash
pip install torch torchvision numpy matplotlib
```

### Running the Notebooks
Launch JupyterLab or Jupyter Notebook:

```bash
jupyter notebook notebooks/AnnotatedTransformer.ipynb
```

### Viewing the Interactive Blueprint

- **Online (GitHub Pages)**: [murtuzasaifee.github.io/transformer_101/annotated_transformer_blueprint.html](https://murtuzasaifee.github.io/transformer_101/annotated_transformer_blueprint.html)
- **Locally**: Open `annotated_transformer_blueprint.html` in any modern web browser:

```bash
# macOS
open annotated_transformer_blueprint.html

# Linux
xdg-open annotated_transformer_blueprint.html
```

---

## 🧠 Transformer Architecture at a Glance

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

| Component | Purpose | Key Concept |
| :--- | :--- | :--- |
| **Positional Encoding** | Injects order information | Sine and Cosine functions across frequencies |
| **Scaled Dot-Product Attention** | Calculates token-to-token relevance | Scaled by $\frac{1}{\sqrt{d_k}}$ to prevent gradient vanishing |
| **Multi-Head Attention** | Attends to information across subspaces | Linear projections split into $h$ heads |
| **Feed-Forward Network** | Position-wise non-linear mapping | Two linear layers with ReLU / GELU |
| **LayerNorm & Residuals** | Stabilizes training in deep networks | $\text{LayerNorm}(x + \text{SubLayer}(x))$ |

---

## 📜 References & Acknowledgements

- **Paper**: [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- **Harvard NLP**: [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- Built with ❤️ for machine learning enthusiasts and researchers.