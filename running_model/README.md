# Bhojpuri Dependency Parser  
### XLM-R (Trankit) + Biaffine Parser + Hindi-Bhojpuri Alignment Transfer

This repository trains a **Bhojpuri dependency parser** using:

- **XLM-R (from Trankit)** as the encoder  
- **Dozat & Manning–style Biaffine Parser**  
- **Supervised loss on Synthetic Bhojpuri UD**  
- **Alignment loss from Hindi → Bhojpuri (GIZA++)**  
- Evaluation on **UD_Bhojpuri-BHTB Test Set**

---

## 1. Environment Setup

### Create Conda env
```bash
conda create --name my_env python=3.9
conda activate my_env
```
### Install dependencies
```
pip install -r requirements.txt
```
## 2. Install Trankit + Download XLM-R Checkpoint
```
cd running_model
git clone https://github.com/nlp-uoregon/trankit.git
cd trankit
pip install -e .
```
#### Run Trankit once to download XLM-R weights (in the running_model/trankit folder): 
```
python run_trankit.py
```
> This creates:
```
trankit/cache/xlm-roberta-base/models--xlm-roberta-base/snapshots/<HASH>/
```
#### Return to running_model folder:
```
cd ..
```


## 3. Directory Structure
```
running_model/
│
├── trankit/
│   └── cache/xlm-roberta-base/models--xlm-roberta-base/snapshots/<HASH>/
│
├── input/
│   ├── alignment.A3.final
│   ├── without_shift_hindi_final_merged.conllu
│   └── bhojpuri_transferred.conllu
│
├── bho_bhtb-ud-test.conllu
├── train_bhojpuri_biaffine.py
└── requirements.txt
```


## 4. Required Input Files
```
| File                                      | Description                                 |
| ----------------------------------------- | ------------------------------------------- |
| `alignment.A3.final`                      | Raw Hindi↔Bhojpuri alignments (from GIZA++) |
| `without_shift_hindi_final_merged.conllu` | Hindi gold UD                               |
| `bhojpuri_transferred.conllu`             | Synthetic Bhojpuri UD                       |
| `bho_bhtb-ud-test.conllu`                 | Gold evaluation test set                    |
```
## 5. Run the Training
> currently in running_model folder
```
python -u G_model_added_biaffine.py
```



# 🌟 Model Pipeline Overview

## 1. XLM-R Encoder (Contextual Embedding)

The pipeline starts with an **XLM-RoBERTa** encoder to generate rich, contextualized embeddings for each word.

* **Input:** Bhojpuri sentence tokens (e.g., `["एकर", "घर", "बहुत", "सुंदर", "बा", "।"]`).
* **Tokenization:** Tokens are split into subwords by the XLM-R tokenizer.
* **Encoding & Aggregation:** Subword IDs are passed through the XLM-R model. The resulting subword embeddings are then **averaged** to produce a final, fixed-size vector **$H_b$** for each word.

## 2. Biaffine Dependency Head Prediction

The word vectors $H_b$ are used to predict the dependency head and label for every token.

* **Arc Prediction (Head):** $H_b$ is projected into separate head ($H_h$) and dependent ($H_d$) representations using MLPs. A **Bilinear** function is applied to these to calculate **Arc Scores** ($N \times N$ matrix), indicating the likelihood of token $h$ being the head of token $d$.
* **Label Prediction (Relation):** Similarly, $H_b$ is projected into label representations ($L_h$, $L_d$). These are used in a separate Biaffine classifier to determine the most likely **dependency label** (relation) between the dependent and its predicted head.

## 3. Training & Loss Functions

The model is trained using a combination of supervised and weakly supervised losses:

* **Supervised Loss ($L_{syn}$):**
    * Calculated using **Cross-Entropy** on the synthetic Bhojpuri annotations (gold heads and labels).
    * $L_{syn} = CE(\text{arcs}) + CE(\text{labels})$.

* **Cross-lingual Alignment Loss ($L_{align}$):**
    * Calculated over Bhojpuri tokens that are **aligned** to Hindi tokens.
    * This provides a weak supervision signal by using the Hindi gold heads and labels as targets for the corresponding aligned Bhojpuri tokens.
    * This is crucial for leveraging the rich Hindi treebank data.

* **Total Loss:**
    * The final training objective is a weighted combination:
        $$L_{total} = L_{syn} + 0.5 \cdot L_{align}$$
    * The alignment loss is down-weighted to mitigate potential noise from the alignment process.

## 4. Evaluation Metrics

The final performance is measured on the gold UD Bhojpuri test set using standard dependency parsing metrics:

* **UAS (Unlabeled Attachment Score):** Percentage of tokens with the **correct predicted head**.
* **LAS (Labeled Attachment Score):** Percentage of tokens with the **correct predicted head AND correct label**.
