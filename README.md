# HemoPI-MOD2

Prediction of Hemolytic Potency (HC50) of Chemically Modified Peptides

---

## 📚 Reference
**Rathore AS et al.** _HemoPI-MOD2: Prediction of Hemolytic Potency (HC50) of Chemically Modified Peptides._ **#Coming Soon#**

---
## 📌 Introduction

**HemoPI-MOD2** is developed to help researchers estimate the hemolytic
activity of peptides before synthesis. It predicts **HC50**, the concentration
causing 50% hemolysis, as a regression value rather than a bare
hemolytic/non-hemolytic call, so analogues can be ranked against each other.

Modified peptides are first-class input: N-terminal acetylation, C-terminal
amidation, head-to-tail cyclization, D-amino acids and non-natural residues are
all expressed in **MAP notation** and used by the models.

Three models are offered, each taking a different view of the same peptide, plus
a **Design** module that enumerates and scores every single-residue analogue of
a lead.

🔗 Visit the web server for more information: [HemoPI-MOD2 Web Server](https://webs.iiitd.edu.in/raghava/hemopimod2)

📖 Please cite relevant content for complete details, including the algorithm
behind the approach.

---

## 🧪 Quick Start

```bash
# 1. Clone or unpack the standalone
cd hemopimod2

# 2. Set up the environment (conda recommended)
conda env create -f environment.yml
conda activate HemoPI-MOD2

# 3. See the available options
python HemoPI_MOD2.py -h

# 4. Run prediction on the sample input
python HemoPI_MOD2.py -i example_sequence.fasta -o output.csv -j 1 -m 1

# 5. Design all single-residue analogues of a lead
python HemoPI_MOD2.py -i example_design.fasta -o design.csv -j 2 -m 1
```

The three model files are already in `models/`; nothing needs downloading.

---

## 🛠️ Installation Options

### 🔹 Standalone Installation

HemoPI-MOD2 is written in **Python 3** and requires the following
dependencies:

#### ✅ Required Libraries

```bash
python>=3.9
pip install scikit-learn==1.6.1
pip install xgboost==3.2.0
pip install "numpy>=1.23,<2.0"
pip install pandas
pip install joblib
```

Needed only for the Chemical (`-m 2`) and Structure (`-m 3`) models:

```bash
pip install rdkit
pip install mordred==1.2.0
pip install "networkx>=2.8,<3.0"
```

Or install everything at once:

```bash
pip install -r requirements.txt
```

### 🔹 Installation using environment.yml

1. Create a new Conda environment:
```bash
conda env create -f environment.yml
```
2. Activate the environment:
```bash
conda activate HemoPI-MOD2
```

---

## ⚠️ Important Note

- `scikit-learn` and `xgboost` are **pinned on purpose**. The three model files
  in `models/` were fitted with scikit-learn 1.6.1, and `chemical_model.pkl`
  holds a serialized XGBoost regressor. Other versions usually load, but are
  not guaranteed to reproduce the published values.
- The Sequence (`-m 1`) and Design (`-j 2`) modules run **without** RDKit and
  Mordred. Install those two only if you need the Chemical or Structure model.

---

## 🔬 Prediction

**HemoPI-MOD2** regresses **pHC50** and reports **HC50 in µM**, where
HC50 = 10^(−pHC50). A **higher HC50 means a less hemolytic peptide**. By
convention, HC50 above 100 µM is read as non-hemolytic. This is the cut-off
published with the method and is fixed.

🔹 **Model Options**
- **Sequence (MAP)**: amino acid composition + modification tags + D/L counts
- **Chemical (SMILES)**: Mordred 2D descriptors
- **Structure (PDB)**: Mordred 2D descriptors from the 3D structure

---

## 🚀 Usage

### 🔹 Minimum Usage

```bash
python HemoPI_MOD2.py -h
```

To run an example:

```bash
python HemoPI_MOD2.py -i example_sequence.fasta
```

### 🔹 Full Usage

```
usage: HemoPI_MOD2.py [-h]
                      -i INPUT
                      [-o OUTPUT]
                      [-j {1,2}]
                      [-m {1,2,3}]
                      [-d {1,2}]
                      [-wd WORKING DIRECTORY]
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `-i INPUT` | Input: peptide sequence in MAP/FASTA format or one sequence per line; SMILES for the Chemical model; a PDB file or a folder of PDB files for the Structure model |
| `-o OUTPUT` | Output file (default: `output.csv`) |
| `-j {1,2}` | Job type: 1-Prediction, 2-Design all possible mutants |
| `-m {1,2,3}` | Model selection: 1: Sequence (MAP), 2: Chemical (SMILES), 3: Structure (PDB) |
| `-d {1,2}` | Display: 1-hemolytic peptides only, 2-all peptides (default: 2) |
| `-wd WORKING` | Working directory for saving results |

Job type 1 (Prediction) runs with any of the three models. Job type 2
(Design) runs on the Sequence (MAP) model only.

`-d` filters only what is printed to the screen. The output CSV always holds
every input peptide.

---

## 📂 Input & Output Files

### ✅ Input File Format

HemoPI-MOD2 supports two formats for peptides:

**FASTA format** (example: `example_sequence.fasta`)

Modifications are written in **MAP notation**, as plain one-letter amino acid
codes with modifications in braces:

| Tag | Meaning |
|-----|---------|
| `{nt:Acet}` | N-terminal acetylation (written before the first residue) |
| `{ct:Amid}` | C-terminal amidation (written after the last residue) |
| `{cyc:N-C}` | head-to-tail backbone cyclization (written at the start) |
| `{d}` | the preceding residue is the D enantiomer |
| `{nnr:orn}` | a non-natural residue occupying one chain position |

```
>example
{nt:Acet}KWKLFKKIL{ct:Amid}
```

For the Structure model, `-i` takes either one `.pdb` file or a folder of
`.pdb` files; each structure's ID is taken from its file name.

### ✅ Output File

- Results are saved in **CSV format**.
- If no output file is specified, results are stored in `output.csv`.

**Prediction (`-j 1`)**

| Column | Meaning |
|--------|---------|
| `ID` | FASTA header, or the PDB file name for `-m 3` |
| `Sequence` | the input sequence, for an entry supplied as a peptide sequence |
| `SMILES` | the structure used by the Chemical model (`-m 2` only) |
| `pHC50` | predicted pHC50 |
| `Prediction` | **HC50 in µM** |
| `Hemolytic_Activity` | `Hemolytic` or `Non-hemolytic`, at the 100 µM cut-off |

Under the Chemical model, an entry supplied as a peptide sequence is reported
with both its sequence and its SMILES; an entry supplied as SMILES has no
sequence form, so only its SMILES is shown. A long SMILES is shortened on the
screen to keep the table readable, and written in full to the CSV.

**Design (`-j 2`)**

| Column | Meaning |
|--------|---------|
| `SeqID` | the lead peptide this variant came from |
| `Mutant_ID` | `Original_Seq1` for the wild type, else `Mutant_K1A` |
| `Sequence` | the full variant sequence, modifications included |
| `Position`, `Wild_Residue`, `Mutant_Residue` | the substitution made |
| `pHC50`, `Prediction` | predicted pHC50 and HC50 in µM |
| `Fold_Change_vs_WT` | HC50 relative to that lead's own wild type |
| `Direction` | `More hemolytic` / `Less hemolytic` / `No change` |
| `Hemolytic_Activity` | `Hemolytic` or `Non-hemolytic` |

---

## 🔍 Jobs & Features

### 🔹 Job Types

| Job | Description |
|-----|-------------|
| 1️⃣ **Prediction** | Predicts the HC50 of each input peptide. |
| 2️⃣ **Design** | Generates and scores **all possible single-residue mutants** of every input peptide. |

### 🔹 The Design module

Every position of each input peptide is substituted by each of the other 19
amino acids, and every variant is scored with the Sequence model.

Modifications are never altered. Terminal tags (`{nt:Acet}`, `{ct:Amid}`,
`{cyc:N-C}`), residue-level tags (`{d}`) and non-natural residues
(`{nnr:...}`) are carried into every variant exactly as supplied; only the
single-letter residue code at the scanned position changes. A non-natural
residue occupies a chain position but is not itself mutated.

More than one lead can be submitted at once, and `Fold_Change_vs_WT` is
measured against each lead's own wild type.

Limits: 50 residues per peptide (50 × 19 = 950 variants), and the Sequence
model only. A per-variant Chemical scan would need a SMILES and descriptor
pass for every mutant, and a Structure scan a folded PDB for every mutant;
neither is practical.

---

## 🧠 The Models

| Model | Input | Features | Estimator | Training samples |
|-------|-------|----------|-----------|------------------|
| Sequence | MAP sequence | 28: 20 AAC + 5 modification tags + 3 D/L | ExtraTrees (tuned) | 1924 |
| Chemical | SMILES | 100 Mordred 2D descriptors | XGBoost (tuned) | 1924 |
| Structure | PDB | 100 Mordred 2D descriptors | ExtraTrees | 1100 |

---

## 📑 Package Contents

| File | Description |
|------|-------------|
| **INSTALLATION.txt** | Installation instructions |
| **LICENSE.txt** | License information |
| **README.md** | This file |
| **HemoPI_MOD2.py** | Python program for prediction and design |
| **stitch.py** | MAP sequence → SMILES builder, used by the Chemical model |
| **models/** | The three model files |
| **data/** | Modification and non-natural residue database |
| **example_sequence.fasta** | Example file (FASTA format) |
| **example_smiles.txt** | Example file (SMILES, for `-m 2`) |
| **example_structure.pdb** | Example file (PDB, for `-m 3`) |
| **example_design.fasta** | Example lead peptide for `-j 2` |
| **environment.yml** | Conda environment specification |
| **requirements.txt** | Pip dependency list |

---

🚀 **Start predicting and designing hemolytic peptides with HemoPI-MOD2 today!**

🔗 Visit: [HemoPI-MOD2 Web Server](https://webs.iiitd.edu.in/raghava/hemopimod2)

(C) 2026 Dr. G.P.S. Raghava

