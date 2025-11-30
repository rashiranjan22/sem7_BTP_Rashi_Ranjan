


# for cleaned data loading
import re

def clean_and_parse_alignment(path, output_path, max_pairs=5000):
    print("======= Cleaning & Parsing Alignment File =======")

    alignments = {}
    current_id = None
    valid_blocks = {}

    skip_this_block = False
    temp_lines = []

    # Patterns
    pattern_pair = re.compile(r"#\s*Sentence\s*pair\s*(\d+)")
    bogus_pattern = re.compile(r"<<File")

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip()

            # Detect new sentence block
            m = pattern_pair.match(line)
            if m:
                # Save previous block
                if current_id is not None and not skip_this_block:
                    valid_blocks[current_id] = temp_lines

                current_id = int(m.group(1))
                temp_lines = [line]
                skip_this_block = False
                continue

            # Detect bogus lines
            if bogus_pattern.search(line):
                skip_this_block = True

            temp_lines.append(line)

    # Save last block
    if current_id is not None and not skip_this_block:
        valid_blocks[current_id] = temp_lines

    # print(f"Before cleaning: {len(valid_blocks)} valid blocks")

    # =====================
    # RENUMBER 1..N
    # =====================
    new_alignments = {}
    sorted_block_ids = sorted(valid_blocks.keys())
    # print("Smallest pair:", sorted_block_ids[0], "Largest:", sorted_block_ids[-1])

    # We stop renumbering at max_pairs (e.g., 5000)
    kept_ids = sorted_block_ids[:max_pairs]

    # ======== WRITE OUT CLEANED FILE ========
    with open(output_path, "w", encoding="utf-8") as out:
        new_id = 1
        for old_id in kept_ids:
            for ln in valid_blocks[old_id]:
                # Replace the wrong "# Sentence pair X" with new value
                if ln.startswith("# Sentence pair"):
                    ln = f"# Sentence pair {new_id}"
                out.write(ln + "\n")
            new_id += 1

    print("Clean alignment written to:", output_path)

    # Now parse cleaned alignment
    return parse_alignment_file(output_path)


def parse_alignment_file(path):
    # print("======= Parsing Alignment File =======")
    alignments = {}
    sent_id = None

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()

            if line.startswith("# Sentence pair"):
                sent_id = int(re.findall(r"\d+", line)[0])
                alignments[sent_id] = []
                continue

            if "({" in line:
                parts = re.findall(r"\(\{\s*(\d+)\s*\}\)", line)
                bh_index = 0
                for hi_idx in parts:
                    alignments[sent_id].append((bh_index, int(hi_idx)-1))
                    bh_index += 1

    print("Total aligned sentences:", len(alignments))
    return alignments




def load_hindi_conllu(path, limit=5000):
    # print("======= Loading Hindi Conllu =======")

    hindi = {}
    sent_id = None
    tokens = []
    heads = []
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line.startswith("# Sentence pair"):
                if sent_id is not None and len(hindi) < limit:
                    hindi[sent_id] = {
                        "tokens": tokens,
                        "heads": heads,
                        "labels": labels
                    }
                if len(hindi) >= limit:
                    break

                sent_id = int(re.findall(r"\d+", line)[0])
                tokens, heads, labels = [], [], []
                continue

            if line and not line.startswith("#"):
                cols = line.split("\t")
                if "-" in cols[0]: continue

                tokens.append(cols[1])
                heads.append(int(cols[6]))
                labels.append(cols[7])

    print("Loaded Hindi sentences:", len(hindi))
    return hindi


def load_bhojpuri_synth(path, limit=5000):
    # print("======= Loading Bhojpuri Synthetic Conllu =======")

    bhoj = {}
    sent_id = None
    tokens, heads, labels = [], [], []

    pattern_pair = re.compile(r"#\s*Sentence\s*pair\s*(\d+)")
    pattern_sentid = re.compile(r"#\s*sent_id\s*=\s*(\d+)")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # sentence start
            m1 = pattern_pair.match(line)
            m2 = pattern_sentid.match(line)

            if m1 or m2:
                if sent_id is not None and len(bhoj) < limit:
                    N = len(tokens)
                    fixed_heads = [(h if 0 <= h < N else 0) for h in heads]
                    bhoj[sent_id] = {
                        "tokens": tokens,
                        "heads": fixed_heads,
                        "labels": labels
                    }
                if len(bhoj) >= limit:
                    break

                sent_id = int(m1.group(1)) if m1 else int(m2.group(1))
                tokens, heads, labels = [], [], []
                continue

            if not line or line.startswith("#"):
                continue

            cols = line.split("\t")
            if "-" in cols[0]: continue

            while len(cols) < 10: cols.append("_")

            tokens.append(cols[1])
            try:
                head = int(cols[6])
            except:
                head = 0

            heads.append(head)

            lab = cols[7] if cols[7] != "_" else "dep"
            labels.append(lab)

    print("Loaded Bhojpuri sentences:", len(bhoj))
    return bhoj





    from transformers import AutoTokenizer, AutoModel
import torch


from transformers import AutoTokenizer, AutoModel

# Load encoder from Trankit snapshot


import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


model_path = os.path.join(
    BASE_DIR,
    "trankit",
    "cache",
    "xlm-roberta-base",
    "models--xlm-roberta-base",
    "snapshots",
    "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
)
encoder = AutoModel.from_pretrained(model_path)

# Load tokenizer from official xlm-roberta-base
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

print("✓ Encoder loaded")
print("✓ Tokenizer loaded")


# print("Model loaded:", type(encoder))
# print("Tokenizer loaded:", type(tokenizer))





# ================================
# CONFIGURATION: FILE PATHS
# ================================


ALIGNMENT_PATH = os.path.join(BASE_DIR, "input", "alignment.A3.final")
HINDI_CONLLU_PATH = os.path.join(BASE_DIR, "input", "without_shift_hindi_final_merged.conllu")
BHOJPURI_SYNTH_PATH = os.path.join(BASE_DIR, "input", "bhojpuri_transferred.conllu")


print("Using files:")
print("  Alignments:", ALIGNMENT_PATH)
print("  Hindi Treebank:", HINDI_CONLLU_PATH)
print("  Bhojpuri Synth:", BHOJPURI_SYNTH_PATH)

# ================================
# LOAD ALIGNMENTS, HINDI, BHOJPURI
# ================================
# alignments = parse_alignment_file(ALIGNMENT_PATH)
# print("\nAlignment Example:", list(alignments.items())[:1])

# hindi_data = load_hindi_conllu(HINDI_CONLLU_PATH)
# print("\nHindi Example:", list(hindi_data.items())[:1])

# bhojpuri_data = load_bhojpuri_synth(BHOJPURI_SYNTH_PATH)
# print("\nBhojpuri Example:", list(bhojpuri_data.items())[:1])


# Step 1 — Clean alignment file & parse it

CLEAN_ALIGN = os.path.join(BASE_DIR, "input", "alignment.cleaned")
alignments = clean_and_parse_alignment(ALIGNMENT_PATH, CLEAN_ALIGN, max_pairs=5000)
# print("\nAlignment Example:", list(alignments.items())[-1])

# Step 2 — Load Hindi & Bhojpuri (first 5000 only)
hindi_data = load_hindi_conllu(HINDI_CONLLU_PATH, limit=5000)
# print("\nHindi Example:", list(hindi_data.items())[-1])

bhojpuri_data = load_bhojpuri_synth(BHOJPURI_SYNTH_PATH, limit=5000)
# print("\nBhojpuri Example:", list(bhojpuri_data.items())[-1])


# DEBUG
print("Alignment:", len(alignments))
print("Hindi:", len(hindi_data))
print("Bhojpuri:", len(bhojpuri_data))



def aggregate_subwords_to_words(tokens, tokenizer, hb_subword):
    encoded = tokenizer(tokens, is_split_into_words=True, return_tensors="pt")
    word_ids = encoded.word_ids(0)

    word_embeddings = []
    current_word = []
    current_id = None

    for i, w_id in enumerate(word_ids):
        if w_id is None:
            # Ignore special tokens and unaligned tokens
            continue

        if current_id is None:
            # First valid word
            current_id = w_id

        if w_id != current_id:
            # Aggregate previous word
            if len(current_word) > 0:
                emb = torch.stack(current_word, dim=0).mean(dim=0)
                word_embeddings.append(emb)

            current_word = []
            current_id = w_id

        current_word.append(hb_subword[i])

    # Last word
    if len(current_word) > 0:
        emb = torch.stack(current_word, dim=0).mean(dim=0)
        word_embeddings.append(emb)

    return torch.stack(word_embeddings, dim=0)



def encode_Hb(tokens):
    # print("\n[Hb] Encoding sentence:")
    # print(" ", tokens[:20], "..." if len(tokens) > 20 else "")

    # Word-aware tokenization
    encoded = tokenizer(tokens, is_split_into_words=True, return_tensors="pt")
    # print("[Hb] Subword IDs:", encoded["input_ids"].shape)

    with torch.no_grad():
        out = encoder(**encoded)

    hb_sub = out.last_hidden_state.squeeze(0)    # [subwords, 768]

    # Aggregate to word-level
    Hb_word = aggregate_subwords_to_words(tokens, tokenizer, hb_sub)

    # print("[Hb] Word-level Hb shape:", Hb_word.shape)
    return Hb_word





import torch.nn as nn

class BiaffineParser(nn.Module):
    def __init__(self, hidden_size=768, arc_hidden=400, lbl_hidden=200, num_labels=28):
        super().__init__()

        print("======= Initializing Biaffine Parser =======")
        print("Hidden={}, Arc hidden={}, Label hidden={}, #Labels={}".format(
            hidden_size, arc_hidden, lbl_hidden, num_labels))

        # Shared MLP projections
        self.arc_dep = nn.Linear(hidden_size, arc_hidden)
        self.arc_head = nn.Linear(hidden_size, arc_hidden)

        self.lbl_dep = nn.Linear(hidden_size, lbl_hidden)
        self.lbl_head = nn.Linear(hidden_size, lbl_hidden)

        # TRUE biaffine for arcs
        self.arc_biaffine = nn.Bilinear(arc_hidden, arc_hidden, 1)

        # TRUE biaffine for labels
        self.lbl_biaffine = nn.Bilinear(lbl_hidden, lbl_hidden, num_labels)

    def forward(self, Hb):
        """
        Hb: (N, hidden)
        Returns:
            arc_scores: (N, N)
            lbl_scores: (N, N, num_labels)
        """
        N = Hb.size(0)

        Hh = self.arc_head(Hb)      # (N, A)
        Hd = self.arc_dep(Hb)       # (N, A)

        Lh = self.lbl_head(Hb)      # (N, L)
        Ld = self.lbl_dep(Hb)       # (N, L)

        # ---------- Vectorized biaffine ARC ----------
        # arc_scores[d,h] = Hd[d]ᵀ * W * Hh[h]
        # → (N, A) @ W → (N, A) → bmm → (N,N)
        W_arc = self.arc_biaffine.weight.squeeze(0)       # (A, A)
        b_arc = self.arc_biaffine.bias                    # (1)

        arc_scores = Hd @ W_arc @ Hh.t()                  # (N, N)
        arc_scores = arc_scores + b_arc

        # ---------- Vectorized biaffine LABEL ----------
        # For labels: (N,A) W (A,L) expansions
        W_lbl = self.lbl_biaffine.weight                  # (num_labels, L, L)
        b_lbl = self.lbl_biaffine.bias                    # (num_labels)

        # Compute Ld W Lhᵀ for all labels
        lbl_scores = torch.einsum(
            "di, lij, hj -> dhl",
            Ld, W_lbl, Lh
        )  # (N, N, num_labels)

        lbl_scores = lbl_scores + b_lbl

        return arc_scores, lbl_scores




def alignment_loss(arc_scores, lbl_scores, Th, aligns, label_vocab):
    # print("\n[Loss] Computing alignment loss...")

    loss_arc = 0.0
    loss_lbl = 0.0
    count = 0

    hindi_heads = Th["heads"]
    hindi_labels = Th["labels"]

    for (bh, hi) in aligns:

        # Convert to 0-based
        bh -= 1
        hi -= 1

        # Skip invalid
        if bh < 0 or hi < 0:
            continue

        if bh >= arc_scores.shape[0]:
            continue

        if hi >= len(hindi_heads):
            continue

        mapped_head = hindi_heads[hi]
        mapped_label = hindi_labels[hi]

        # Skip unmapped / invalid Hindi head
        if mapped_head < 0 or mapped_head >= arc_scores.shape[0]:
            continue

        if mapped_label not in label_vocab:
            continue

        # Arc supervision
        pred_arc = arc_scores[bh].unsqueeze(0)
        # --------------------added for optimisation
        # gold_arc = torch.tensor([mapped_head])
        gold_arc = torch.tensor([mapped_head], device=arc_scores.device)

        loss_arc += F.cross_entropy(pred_arc, gold_arc)

        # Label supervision
        # ------------added for optimisation
        # pred_lbl = lbl_scores[bh, mapped_head, :].unsqueeze(0)
        # gold_lbl = torch.tensor([label_vocab[mapped_label]])
        pred_lbl = lbl_scores[bh, mapped_head].unsqueeze(0)
        gold_lbl = torch.tensor([label_vocab[mapped_label]], device=arc_scores.device)
        #---------------------

        loss_lbl += F.cross_entropy(pred_lbl, gold_lbl)

        count += 1

    if count == 0:
        return torch.tensor(0.0)

    return (loss_arc + loss_lbl) / count




import torch.nn.functional as F

def supervised_loss(arc_scores, lbl_scores, heads, labels, label_vocab):
    # print("\n[Loss] Computing supervised loss...")

    N = len(heads)

    # ---------------------------
    # FIX INVALID HEADS FIRST
    # ---------------------------
    fixed_heads = []
    for h in heads:
        if h < 0 or h >= N:
            fixed_heads.append(0)
        else:
            fixed_heads.append(h)

    heads_tensor = torch.tensor(fixed_heads)

    # ---------------------------
    # ARC LOSS
    # ---------------------------
    loss_arc = F.cross_entropy(arc_scores, heads_tensor)
    # print("  Arc CE:", float(loss_arc))

    # ---------------------------
    # LABEL LOSS (use gold head per token)
    # ---------------------------
    label_preds = []

    for d in range(N):
        h = fixed_heads[d]

        # avoid crash for impossible heads
        if h < 0 or h >= lbl_scores.shape[1]:
            h = 0

        label_preds.append(lbl_scores[d, h, :].unsqueeze(0))

    label_preds = torch.cat(label_preds, dim=0)  # (N, num_labels)

    # Fix missing labels
    gold_lbl_ids = []
    for lab in labels:
        if lab not in label_vocab:
            gold_lbl_ids.append(label_vocab["dep"])
        else:
            gold_lbl_ids.append(label_vocab[lab])

    # ----------added for optimisation
    # gold_lbl_ids = torch.tensor(gold_lbl_ids)
    gold_lbl_ids = torch.tensor(gold_lbl_ids, device=arc_scores.device)
    #---------------


    loss_lbl = F.cross_entropy(label_preds, gold_lbl_ids)
    # print("  Label CE:", float(loss_lbl))

    return loss_arc + loss_lbl




# -------------------------------------------
# BUILD LABEL VOCAB FROM HINDI + BHOJPURI
# -------------------------------------------

def build_label_vocab(hindi_data, bhojpuri_data):
    label_set = set()

    # Collect labels from Hindi
    for sid, info in hindi_data.items():
        label_set.update(info["labels"])  # Hindi gold labels

    # Collect labels from Bhojpuri
    for sid, info in bhojpuri_data.items():
        label_set.update(info["labels"])  # Synthetic Bhojpuri labels

    # Sorting is optional but recommended
    label_list = sorted(list(label_set))

    # Create mapping
    label_vocab = {label: idx for idx, label in enumerate(label_list)}

    print("Total labels found:", len(label_vocab))
    return label_vocab

label_vocab = build_label_vocab(hindi_data, bhojpuri_data)
print("Label vocab example:", list(label_vocab.items())[:10])



import os
import random
import torch

NUM_EPOCHS = 5
CHECKPOINT_DIR = "checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

parser = BiaffineParser(num_labels=len(label_vocab))
optimizer = torch.optim.Adam(parser.parameters(), lr=2e-5)
id2label = {idx: label for label, idx in label_vocab.items()}

bhojpuri_sentence_ids = list(bhojpuri_data.keys())
TOTAL_STEPS = len(bhojpuri_sentence_ids)

best_loss = float("inf")

print(f"Starting training for {NUM_EPOCHS} epochs…")
print(f"Total Bhojpuri sentences = {TOTAL_STEPS}")

# -----------------------added for optimisation

print("Caching all Hb embeddings...")
Hb_cache = {}

for sid in bhojpuri_sentence_ids:
    Hb_cache[sid] = encode_Hb(bhojpuri_data[sid]["tokens"])

print("✓ Finished caching HB vectors.")

#---------------------------


for epoch in range(1, NUM_EPOCHS + 1):

    print("\n" + "="*60)
    print(f"==============  EPOCH {epoch}/{NUM_EPOCHS}  ==============")
    print("="*60)

    random.shuffle(bhojpuri_sentence_ids)
    epoch_loss = 0.0

    for idx, sid in enumerate(bhojpuri_sentence_ids, start=1):

        # ----------------------------
        # Progress bar (kept)
        # ----------------------------
        if idx % 20 == 0 or idx == 1:
            progress = (idx / TOTAL_STEPS) * 100
            print(f"[Epoch {epoch}] Progress: {progress:5.1f}%  ({idx}/{TOTAL_STEPS})")

        # ----------------------------
        # Encode → Forward → Loss
        # ----------------------------


        # Hb = encode_Hb(bhojpuri_data[sid]["tokens"])
        # ------------added for opitmisation
        Hb = Hb_cache[sid]
        #-------------------

        arc_s, lbl_s = parser(Hb)

        L_syn = supervised_loss(
            arc_s, lbl_s,
            bhojpuri_data[sid]["heads"],
            bhojpuri_data[sid]["labels"],
            label_vocab
        )

        L_al = alignment_loss(
            arc_s, lbl_s,
            hindi_data[sid],
            alignments[sid],
            label_vocab
        )

        loss = L_syn + 0.5 * L_al
        epoch_loss += float(loss)

        # ----------------------------
        # Print loss only every 50 steps
        # ----------------------------
        if idx % 50 == 0:
            print(f"  → Step {idx}: Loss = {float(loss):.4f}")

        # ----------------------------
        # Backprop
        # ----------------------------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"\n Epoch {epoch} Finished — Total Loss = {epoch_loss:.4f}")

    # =====================================================
    # SAVE CHECKPOINT FOR THIS EPOCH
    # =====================================================
    epoch_ckpt = f"{CHECKPOINT_DIR}/parser_epoch_{epoch}.pt"
    torch.save({
        "epoch": epoch,
        "parser_state": parser.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "label_vocab": label_vocab,
        "id2label": id2label
    }, epoch_ckpt)

    print(f" Saved checkpoint: {epoch_ckpt}")

    # =====================================================
    # SAVE BEST MODEL
    # =====================================================
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_ckpt = f"{CHECKPOINT_DIR}/parser_best.pt"

        torch.save({
            "epoch": epoch,
            "parser_state": parser.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "label_vocab": label_vocab,
            "id2label": id2label
        }, best_ckpt)

        print(f" ⭐ New BEST model saved: {best_ckpt}")

    # =====================================================
    # SAVE LATEST MODEL
    # =====================================================
    latest_ckpt = f"{CHECKPOINT_DIR}/parser_latest.pt"
    torch.save(parser.state_dict(), latest_ckpt)
    print(f" Latest model saved: {latest_ckpt}")




import torch
import json

# Save parser weights
torch.save(parser.state_dict(), "bhojpuri_biaffine_parser.pt")
print("✓ Parser saved")

# Save vocab
with open("label_vocab.json", "w", encoding="utf-8") as f:
    json.dump(label_vocab, f, ensure_ascii=False, indent=2)

with open("id2label.json", "w", encoding="utf-8") as f:
    json.dump(id2label, f, ensure_ascii=False, indent=2)

print("✓ Vocab saved")




def load_gold_conllu(path):
    print("Loading gold Bhojpuri test set...")
    data = {}
    sent_id = 0

    tokens = []
    heads = []
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line.startswith("#"):
                continue
            if not line:
                if tokens:
                    sent_id += 1
                    data[sent_id] = {"tokens": tokens, "heads": heads, "labels": labels}
                    tokens, heads, labels = [], [], []
                continue

            cols = line.split("\t")
            if "-" in cols[0]:
                continue

            tokens.append(cols[1])
            heads.append(int(cols[6]))
            labels.append(cols[7])

    print("Total gold sentences:", len(data))
    return data

gold_test = load_gold_conllu("bho_bhtb-ud-test.conllu")




parser = BiaffineParser(num_labels=len(label_vocab))
parser.load_state_dict(torch.load("bhojpuri_biaffine_parser.pt"))
parser.eval()

print("✓ Parser loaded for evaluation")


def parse_sentence(tokens):
    Hb = encode_Hb(tokens)
    arc_scores, lbl_scores = parser(Hb)

    predicted_heads = arc_scores.argmax(dim=1).tolist()

    predicted_labels = []
    for d, h in enumerate(predicted_heads):
        lbl_id = lbl_scores[d, h, :].argmax().item()
        predicted_labels.append(id2label[str(lbl_id)] if str(lbl_id) in id2label else id2label[lbl_id])

    return predicted_heads, predicted_labels




def evaluate(gold_data):
    total_tokens = 0
    correct_heads = 0
    correct_labels = 0

    for sid, sent in gold_data.items():
        tokens = sent["tokens"]
        gold_heads = sent["heads"]
        gold_labels = sent["labels"]

        pred_heads, pred_labels = parse_sentence(tokens)

        for gh, gl, ph, pl in zip(gold_heads, gold_labels, pred_heads, pred_labels):
            total_tokens += 1
            if gh == ph:
                correct_heads += 1
                if gl == pl:
                    correct_labels += 1

    uas = correct_heads / total_tokens
    las = correct_labels / total_tokens

    print("UAS:", uas)
    print("LAS:", las)

    return uas, las




uas, las = evaluate(gold_test)
print("FINAL RESULTS:")
print("UAS:", uas)
print("LAS:", las)



