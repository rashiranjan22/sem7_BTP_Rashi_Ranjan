## 📝 Constrained Cross-Lingual Parsing Pipeline

This document describes the $\mathbf{Constrained}$ $\mathbf{Cross-Lingual}$ $\mathbf{Parsing}$ $\mathbf{Pipeline}$ implemented to train a Bhojpuri dependency parser using synthetic data and a gold Hindi constraint.

---
![alt text](model_flowchart.png)

### 1. Data Loading and Feature Preparation

This initial phase establishes the core inputs for the model by loading necessary components and preparing the data features.

* **Load Encoder & Tokenizer:**
    * The **XLM-RoBERTa (XLM-R) Encoder** is loaded from Trankit's cache path.
    * The standard $\mathbf{xlm-roberta-base}$ **Tokenizer** is loaded for sequence processing.

* **Load Treebanks and Alignment:** Resources are loaded and synchronized by sentence ID.
    * **Alignment ($A$):** Provides the **word alignments** between Hindi (source) and Bhojpuri (target).
    * **Hindi Gold Data ($\mathbf{T}_{H}$):** Contains the **gold heads and labels** from the Hindi CoNLL-U. (Note: $\mathbf{T}_{H}$ uses discrete gold labels).
    * **Bhojpuri Synthetic Data ($\mathbf{Y}_{B}^{syn}$):** Contains the **noisy heads and labels** (synthetic) from the projected Bhojpuri CoNLL-U.

* **Encode Features ($\mathbf{H}_{B}$):**
    * The shared $\mathbf{XLM-R}$ **Encoder** generates **word-level contextual embeddings ($\mathbf{H}_{B}$)** for the Bhojpuri tokens.



---

### 2. The Custom Biaffine Parser ($\mathbf{P}_{Synth}$)

The core of the model is a custom implementation of the standard dependency parsing architecture.

* **Architecture:** The $\mathbf{BiaffineParser}$ class takes the Bhojpuri embeddings ($\mathbf{H}_{B}$) as input.
* **MLP Projection:** The input $\mathbf{H}_{B}$ is projected into specialized representations:
    * Heads and Dependents for $\mathbf{Arcs}$ ($\mathbf{H_h}$, $\mathbf{H}_{d}$).
    * Heads and Dependents for $\mathbf{Labels}$ ($\mathbf{L_h}, \mathbf{L}_{d}$).
* **Biaffine Attention:** A **True Biaffine** transformation (using $nn.Bilinear$) is applied to compute the **Arc Scores** and **Label Scores** ($\mathbf{P}_{B}^{pred}$).

---

### 3. The Constrained Training Loop

The model is trained by minimizing a combined loss function that balances synthetic supervision and cross-lingual constraint.

#### **Total Loss Calculation**
The training loop iterates over the synthetic Bhojpuri sentences, calculating a weighted total loss:
$$L_{Total} = L_{Syn} + \lambda \cdot L_{Al} \quad (\text{where } \lambda = 0.5)$$

#### **Supervised Loss ($L_{Syn}$)**
* **Source:** $\mathbf{Y}_{B}^{syn}$ (Bhojpuri synthetic labels/heads).
* **Mechanism:** Calculates a standard **Cross-Entropy Loss** between the predicted scores ($\mathbf{P}_{B}^{pred}$) and the **noisy synthetic labels** ($\mathbf{Y_B}^{syn}$). This trains $\mathbf{P_Synth}$ to capture the basic structure of the synthetic Bhojpuri data.

#### **Alignment Loss ($L_{Al}$ — The Constraint)**
* **Source:** $\mathbf{T}_{H}$ (Hindi gold heads/labels) and the $\mathbf{Alignment}$ map.
* **Mechanism:**
    1.  The loop iterates over the **aligned word pairs** $(bh, hi)$.
    2.  For each Bhojpuri word ($bh$), the gold Hindi head/label is **mapped** via the alignment to form a target.
    3.  A **Cross-Entropy Loss** is calculated between $\mathbf{P}_{B}^{pred}$ and this **mapped gold Hindi target**.
    4.  **Effect:** This loss pulls the $\mathbf{P_Synth}$'s predictions toward the reliable **gold Hindi structure**, serving to correct the noise in $\mathbf{Y}_{B}^{syn}$.

* **Optimization:** The $L_{Total}$ is backpropagated to update the weights of the $\mathbf{P}_{Synth}$ Biaffine layers.



---

### 4. Evaluation

The final phase assesses the performance of the trained parser.

* **Model:** The final trained $\mathbf{P}_{Synth}$ model.
* **Data:** A separate **Bhojpuri gold test set** (e.g., *bho\_bhtb-ud-test.conllu*).
* **Metrics:** Standard dependency parsing metrics are used:
    * **UAS (Unlabeled Attachment Score)**
    * **LAS (Labeled Attachment Score)**
