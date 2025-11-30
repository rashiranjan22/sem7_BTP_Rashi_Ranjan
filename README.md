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
|   ___ run_trankit.py
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


## 6. Alignment Cleaning Step
the function ```clean_and_parse_alignment``` creates a cleaned and renumbered alignment file ```running_model/input/alignment.cleaned```

## 7. Checkpoint Files Saved
All model checkpoints are saved automatically inside:

``` running_model/checkpoints/```
The script produces:
```
| File                | Meaning                                |
| ------------------- | -------------------------------------- |
| `parser_epoch_X.pt` | Model + optimizer state for each epoch |
| `parser_latest.pt`  | Latest model (state dict only)         |
| `parser_best.pt`    | Model with **lowest epoch loss**       |

```
Directory example:
```
checkpoints/
├── parser_epoch_1.pt
├── parser_epoch_2.pt
├── parser_epoch_3.pt
├── parser_best.pt
└── parser_latest.pt
```


## 8. Final Saved Model & Vocab

At the end of training, the script writes:
```

| File                          | Description                 |
| ----------------------------- | --------------------------- |
| `bhojpuri_biaffine_parser.pt` | Final model weights         |
| `label_vocab.json`            | Mapping: label → integer ID |
| `id2label.json`               | Reverse mapping: ID → label |

```
These are stored in:
```
running_model/
```

You will see:
```
✓ Parser saved
✓ Vocab saved
```
