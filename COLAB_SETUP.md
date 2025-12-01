# Bhojpuri Dependency Parser (colab notebook)

Run the ```G_colab_model_added_biaffine.ipynb``` on google colab. <br>

---
### 1.Clone Trankit & Run the Script
After installing dependencies, the notebook clones the Trankit repository:
```
git clone https://github.com/nlp-uoregon/trankit.git
cd trankit
```
Inside the ```content/trankit/``` directory, notebook runs the script:
```
python3.9 run_trankit.py
```
Once done, moves back to the main folder

### 2. Place Required Input Files
Before running the configuration section, create an ```input/``` directory:
```
/content/input/
   ├── alignment.A3.final
   ├── without_shift_hindi_final_merged.conllu
   ├── bhojpuri_transferred.conllu
```
The script then generates:
```
/content/input/alignment.cleaned
```
### 3. Bhojpuri Gold Test Will Auto-Download
during evaluation, this command runs:
```
wget -O bho_test.conllu https://raw.githubusercontent.com/UniversalDependencies/UD_Bhojpuri-BHTB/master/bho_bhtb-ud-test.conllu
```

So the file automatically appears in:
``` 
/content/bho_test.conllu
```
### 4. Checkpoints Location

All training checkpoints are saved in:
```
/content/checkpoints/
```

Specifically:
```
| File               | Meaning                              |
| ------------------ | ------------------------------------ |
| `parser_latest.pt` | Always overwritten — used for resume |
| `parser_best.pt`   | Best model (lowest training loss)    |
```
### 5. Final structure 
```
/content/
   ├── input/
   │     ├── alignment.A3.final
   │     ├── alignment.cleaned
   │     ├── without_shift_hindi_final_merged.conllu
   │     └── bhojpuri_transferred.conllu
   │
   ├── checkpoints/
   │     ├── parser_best.pt
   │     └── parser_latest.pt
   │
   ├── bhojpuri_biaffine_parser.pt
   ├── label_vocab.json
   ├── id2label.json
   ├── bho_test.conllu
   ├── trankit/
   └── (notebook)
```
