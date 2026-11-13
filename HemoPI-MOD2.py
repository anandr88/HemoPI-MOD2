#########################################################################
# HemoPI-MOD2 is developed for predicting the hemolytic activity (HC50)  #
# of natural and chemically modified peptides, and for designing less    #
# hemolytic analogues. It is developed by Prof G. P. S. Raghava's group. #
# Please cite : HemoPI-MOD2                                              #
#########################################################################

import os
import sys
import argparse
import pickle
import joblib
import re
import warnings

from typing import List, Tuple, Dict

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

DB_PATH = os.path.join(SCRIPT_DIR, "data", "merged_final_data.json")

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
STANDARD_AAS = AA_LIST

MODEL_PATHS = {
    "Sequence":  os.path.join(SCRIPT_DIR, "models", "sequence_model.pkl"),
    "Chemical":  os.path.join(SCRIPT_DIR, "models", "chemical_model.pkl"),
    "Structure": os.path.join(SCRIPT_DIR, "models", "structure_model.pkl"),
}

MODEL_CHOICE_MAP = {
    1: "Sequence",
    2: "Chemical",
    3: "Structure",
}

JOB_NAMES = {
    1: "Prediction",
    2: "Design",
}

# Maximum peptide length accepted for a design scan. A 50-mer full scan is
# 50 x 19 = 950 variants, which still runs in a few seconds.
MAX_RESIDUES = 50

# HC50 above this value (uM) is non-hemolytic. This is the cut-off reported in
# the HemoPI-MOD2 paper, so it is fixed and deliberately not configurable from
# the command line.
HC50_THRESHOLD = 100.0


# ===================== SEQUENCE FEATURES =====================

def clean_sequence_for_composition(seq: str) -> str:
    """Remove MAP modification tags and retain only standard amino acids."""
    seq = re.sub(r"\{[^{}]*\}", "", str(seq))
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", seq.upper())


def calc_aac(sequence: str) -> dict:
    """Amino acid composition (%) for all 20 standard amino acids."""
    seq = clean_sequence_for_composition(sequence)
    length = len(seq)

    if not length:
        return {f"AAC_{aa}": 0.0 for aa in AA_LIST}

    return {
        f"AAC_{aa}": seq.count(aa) / length * 100.0
        for aa in AA_LIST
    }


# ===================== D/L MAP PARSER =====================

def parse_d_map_sequences(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Extract D/L residue counts from MAP sequences.

    Supported D stereochemistry markers: {d}, {ptm:d}
    The marker applies to the immediately preceding residue.
    """
    n_rows = len(series)

    d_counts = np.zeros((n_rows, len(STANDARD_AAS)), dtype=int)
    l_counts = np.zeros((n_rows, len(STANDARD_AAS)), dtype=int)

    aa_to_idx = {aa: i for i, aa in enumerate(STANDARD_AAS)}

    for idx, seq in enumerate(series):
        tokens = []
        i, n = 0, len(seq)

        # Tokenize MAP sequence
        while i < n:
            if seq[i] == "{":
                j = seq.find("}", i)
                if j == -1:
                    break
                tokens.append(seq[i:j + 1])
                i = j + 1
            else:
                tokens.append(seq[i].upper())
                i += 1

        residues = []

        # Identify residues and stereochemistry
        for token in tokens:
            if token.startswith("{"):
                if token.lower() in ["{d}", "{ptm:d}"] and residues:
                    stereo, aa = residues[-1]
                    residues[-1] = ("D", aa)
            elif token in aa_to_idx:
                residues.append(("L", token))

        # Count D and L residues
        for stereo, aa in residues:
            col = aa_to_idx[aa]
            if stereo == "D":
                d_counts[idx, col] += 1
            else:
                l_counts[idx, col] += 1

    return d_counts, l_counts


# ===================== MAP CHIRALITY FEATURES =====================

def extract_map_chirality_features(seqs):
    """Calculate total_D_count, total_L_count and D_ratio.

    Names match the training columns exactly (summary_dl in the benchmarking
    script), because the model artifact selects its features by name.
    """
    map_series = pd.Series(seqs).fillna("").astype(str)

    d_counts, l_counts = parse_d_map_sequences(map_series)

    total_d = d_counts.sum(axis=1)
    total_l = l_counts.sum(axis=1)
    total = total_d + total_l

    return pd.DataFrame({
        "total_D_count": total_d,
        "total_L_count": total_l,
        "D_ratio": np.divide(
            total_d,
            total,
            out=np.zeros(len(total_d), dtype=float),
            where=total > 0
        )
    })


# ===================== MAP MODIFICATION TAG FEATURES =====================

def extract_map_tag_features(seqs):
    """Detect the specific MAP modification tags the Sequence model was
    trained on. Case-insensitive binary presence/absence features.

    Features (core_mods in the benchmarking script):
        ct:Amid : C-terminal amidation
        cyc:N-C : N-to-C backbone cyclization
        d       : D-stereochemistry marker present
        nnr:orn : ornithine, a non-natural residue
        nt:Acet : N-terminal acetylation

    The names must match the training columns character for character,
    because load_artifact()/make_feature_matrix() select features by name and
    silently median-fill anything they cannot find.

    These are SPECIFIC tags, not tag categories: {nt:Form} is an N-terminal
    modification but is not nt:Acet and must not set it. Detecting categories
    instead (has_nt, has_ct, ...) produces names the artifact does not know,
    which drops every modification feature at inference time.
    """
    map_series = pd.Series(seqs).fillna("").astype(str)

    features = pd.DataFrame(index=map_series.index)

    TAG_PATTERNS = {
        "ct:Amid": r"\{ct:amid\}",
        "cyc:N-C": r"\{cyc:n-c\}",
        "d":       r"\{(?:d|ptm:d)\}",
        "nnr:orn": r"\{nnr:orn\}",
        "nt:Acet": r"\{nt:acet\}",
    }

    for name, pattern in TAG_PATTERNS.items():
        features[name] = (
            map_series.str.contains(pattern, case=False, regex=True)
        ).astype(np.int8)

    return features


# ===================== COMPLETE SEQUENCE FEATURE EXTRACTION =====================

def extract_seq_features(records: List[Tuple[str, str]]) -> pd.DataFrame:
    """Extract Sequence model features.

        1. AAC             (20)
        2. total_D_count    (1)
        3. total_L_count    (1)
        4. D_ratio          (1)
        5. ct:Amid          (1)
        6. cyc:N-C          (1)
        7. d                (1)
        8. nnr:orn          (1)
        9. nt:Acet          (1)

    Total: 28 predictive features (matches Anchor_Base_+_Summary_DL).

    ID and Sequence are retained as metadata.
    """
    headers = [r[0] for r in records]
    seqs = [str(r[1]) if r[1] is not None else "" for r in records]

    df_comp = pd.DataFrame([calc_aac(seq) for seq in seqs])
    df_chirality = extract_map_chirality_features(seqs)
    df_tags = extract_map_tag_features(seqs)

    df_feats = pd.concat([df_comp, df_chirality, df_tags], axis=1)

    df_feats.insert(0, "ID", headers)
    df_feats.insert(1, "Sequence", seqs)

    return df_feats


# ===================== INPUT / DESCRIPTOR PROCESSING =====================

def load_input_sequences(input_path: str) -> List[Tuple[str, str]]:
    """Read peptides from a FASTA file, or one sequence per line."""
    if not os.path.exists(input_path):
        sys.exit(f"Error: Input file '{input_path}' not found.")

    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    records = []

    if any(line.startswith(">") for line in lines):
        header, seq_lines = None, []

        for line in lines:
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_lines)))
                header = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line)

        if header is not None:
            records.append((header, "".join(seq_lines)))
    else:
        records = [
            (f"Seq_{i}", line)
            for i, line in enumerate(lines, 1)
        ]

    return records


def is_valid_smiles(smiles_str: str) -> bool:
    from rdkit import Chem, rdBase

    if not isinstance(smiles_str, str) or not smiles_str.strip():
        return False

    item = smiles_str.strip()

    if "{" in item or "}" in item:
        return False

    if re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", item, re.I):
        return False

    rdBase.DisableLog("rdApp.error")
    try:
        mol = Chem.MolFromSmiles(item)
    finally:
        rdBase.EnableLog("rdApp.error")

    return mol is not None


def pdb_to_smiles(pdb_path: str) -> str:
    """Convert a PDB structure to a canonical SMILES string.

    Hydrogens are REMOVED. An all-atom PDB (which is what PEPstrMOD2 and most
    modelling tools write) read with removeHs=False gives RDKit separate
    hydrogen atoms joined by unspecified bonds, and MolToSmiles then emits
    fragments like "[HH]~OC(=O)..." that MolFromSmiles rejects with
    "Explicit valence for atom N, 5, is greater than permitted". The
    descriptor step would silently skip such a structure, leaving an empty
    feature matrix and a failed run.

    Removing hydrogens loses nothing here: the Structure model uses 2D
    Mordred descriptors (ignore_3D=True), which are computed from the
    molecular graph with implicit hydrogens.
    """
    from rdkit import Chem

    mol = Chem.MolFromPDBFile(pdb_path, removeHs=True)

    if mol is None:
        # Some files only parse with hydrogens retained; strip them after.
        mol = Chem.MolFromPDBFile(pdb_path, removeHs=False)
        if mol is not None:
            try:
                mol = Chem.RemoveHs(mol)
            except Exception:
                pass

    if mol is None:
        raise ValueError(f"Could not parse PDB structure from {pdb_path}")

    smiles = Chem.MolToSmiles(mol)

    # Fail loudly here rather than letting compute_mordred_2d skip the
    # structure and report only "feature matrix is empty".
    if Chem.MolFromSmiles(smiles) is None:
        raise ValueError(
            f"PDB {pdb_path} produced a SMILES that RDKit cannot read back "
            f"({smiles[:60]}...). The coordinates or connectivity may be "
            "malformed."
        )

    return smiles


def compute_mordred_2d(smiles_list: List[str], headers: List[str]) -> pd.DataFrame:
    from rdkit import Chem
    from mordred import Calculator, descriptors

    calc = Calculator(descriptors, ignore_3D=True)

    rows = []

    for header, smi in zip(headers, smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)

            if mol is None:
                print(f"Warning: Invalid SMILES for {header}; skipping.")
                continue

            row = calc(mol).asdict()
            row["ID"] = header
            row["SMILES"] = smi
            rows.append(row)

        except Exception as exc:
            print(f"Warning: Mordred calculation failed for {header}: {exc}")

    return pd.DataFrame(rows)


# ===================== PICKLE ARTIFACT COMPATIBILITY =====================

def first_present(mapping, names):
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def load_artifact(model_path):
    with open(model_path, "rb") as f:
        artifact = joblib.load(f)

    model = artifact
    scaler = cols = medians = None

    if isinstance(artifact, dict):
        model = first_present(
            artifact,
            ("model", "pipeline", "estimator", "regressor", "best_model")
        )
        scaler = first_present(artifact, ("scaler", "feature_scaler"))
        cols = first_present(
            artifact,
            ("feature_names", "feature_columns", "selected_features",
             "cols", "features")
        )
        medians = first_present(
            artifact,
            ("impute_medians", "medians", "feature_medians", "median_values")
        )

        if model is None:
            candidates = [
                (k, v) for k, v in artifact.items()
                if (hasattr(v, "predict") or hasattr(v, "steps"))
            ]
            if len(candidates) == 1:
                model = candidates[0][1]
            else:
                raise ValueError(
                    f"Could not identify the regressor estimator in "
                    f"{model_path}. Available keys: {list(artifact.keys())}"
                )

    if not hasattr(model, "predict"):
        raise ValueError(
            f"Loaded object from {model_path} has no predict() method."
        )

    return model, scaler, cols, medians, artifact


def metadata_columns(df):
    return [c for c in ("ID", "Sequence", "SMILES") if c in df.columns]


def make_feature_matrix(df_features, model, scaler, cols, medians):
    meta = metadata_columns(df_features)

    X = df_features.drop(columns=meta, errors="ignore").copy()
    X.columns = [str(c) for c in X.columns]

    if cols is not None:
        cols = [str(c) for c in list(cols)]
        X = X.reindex(columns=cols)

    X = X.apply(pd.to_numeric, errors="coerce")

    if medians is not None:
        if isinstance(medians, pd.Series):
            median_series = medians.copy()
        elif isinstance(medians, dict):
            median_series = pd.Series(medians)
        else:
            median_series = pd.Series(
                medians,
                index=(cols if cols is not None else X.columns)
            )

        median_series.index = median_series.index.map(str)
        median_series = median_series.reindex(X.columns)

        X = X.fillna(median_series)

    X = X.fillna(0.0)

    is_pipeline = hasattr(model, "steps")

    if scaler is not None and not is_pipeline:
        values = scaler.transform(X)
        X = pd.DataFrame(values, columns=X.columns, index=X.index)

    return X


# ===================== MAP TOKENISATION (DESIGN) =====================

def tokenize_map(seq: str) -> List[str]:
    """Split a MAP sequence into tags ('{...}') and single residue letters.

    '{nt:Acet}KW{d}K{ct:Amid}' ->
        ['{nt:Acet}', 'K', 'W', '{d}', 'K', '{ct:Amid}']
    """
    tokens: List[str] = []
    i, n = 0, len(seq)
    while i < n:
        if seq[i] == "{":
            j = seq.find("}", i)
            if j == -1:
                raise ValueError(
                    f"Unclosed modification tag starting at character "
                    f"{i + 1}. Every '{{' must have a matching '}}'."
                )
            tokens.append(seq[i:j + 1])
            i = j + 1
        elif seq[i].isspace():
            i += 1
        else:
            tokens.append(seq[i].upper())
            i += 1
    return tokens


def residue_sites(tokens: List[str]) -> List[Tuple[int, int, str]]:
    """Locate chain residues.

    Returns (token_index, chain_position, residue) for each residue in the
    chain. Chain position is 1-based and counts non-natural residues
    ({nnr:...}) as well, so numbering matches the real peptide. Only standard
    amino acids are returned as mutable sites; an {nnr:...} residue occupies a
    position but is not itself mutated.
    """
    sites: List[Tuple[int, int, str]] = []
    pos = 0
    for idx, tok in enumerate(tokens):
        if tok.startswith("{"):
            if tok.lower().startswith("{nnr:"):
                pos += 1  # occupies a chain position, not mutable
            continue
        if tok in AA_LIST:
            pos += 1
            sites.append((idx, pos, tok))
        else:
            raise ValueError(
                f"Unrecognised residue '{tok}' at character position "
                f"{idx + 1}. Use standard one-letter amino acid codes, or "
                "MAP tags in braces for modifications."
            )
    return sites


def all_mutants(records: List[Tuple[str, str]]) -> List[Dict]:
    """Generate every single-residue variant of every input peptide.

    For each peptide the wild type is emitted first, then each position
    substituted by each of the other 19 amino acids.

    Modifications are never altered. Terminal tags ({nt:Acet}, {ct:Amid},
    {cyc:N-C}), residue-level tags ({d}) and non-natural residues
    ({nnr:orn}) are carried into every variant exactly as supplied; only the
    single-letter residue code at the scanned position is substituted.
    """
    variants: List[Dict] = []

    for k, (seq_id, seq) in enumerate(records, 1):
        tokens = tokenize_map(seq)
        sites = residue_sites(tokens)

        if not sites:
            raise ValueError(
                f"{seq_id}: no standard amino acid residues found."
            )
        if len(sites) > MAX_RESIDUES:
            raise ValueError(
                f"{seq_id}: peptide has {len(sites)} residues; the design "
                f"scan is limited to {MAX_RESIDUES}."
            )

        variants.append({
            "SeqID": seq_id,
            "Mutant_ID": f"Original_Seq{k}",
            "Position": "-",
            "Wild_Residue": "-",
            "Mutant_Residue": "-",
            "Sequence": "".join(tokens),
            "lead": k,
            "is_wt": True,
        })

        for tok_idx, pos, wt in sites:
            for mut in AA_LIST:
                if mut == wt:
                    continue
                new_tokens = list(tokens)
                new_tokens[tok_idx] = mut  # only the residue letter changes
                variants.append({
                    "SeqID": seq_id,
                    "Mutant_ID": f"Mutant_{wt}{pos}{mut}",
                    "Position": pos,
                    "Wild_Residue": wt,
                    "Mutant_Residue": mut,
                    "Sequence": "".join(new_tokens),
                    "lead": k,
                    "is_wt": False,
                })

    return variants


# ===================== SHARED HELPERS =====================

def looks_like_peptide(seq: str) -> bool:
    """True when a string is a peptide sequence in MAP notation.

    Modification tags are stripped, then every remaining character must be one
    of the 20 standard one-letter amino acid codes. A SMILES string fails on
    its digits, brackets, bond symbols or lowercase aromatic atoms.

    This deliberately does NOT use RDKit. The Sequence model and the Design
    module must keep working in an installation without RDKit and Mordred,
    which are needed only by the Chemical and Structure models.
    """
    body = re.sub(r"\{[^{}]*\}", "", str(seq))
    body = re.sub(r"\s+", "", body).upper()

    if not body:
        return False

    return all(ch in AA_LIST for ch in body)


def require_peptide_input(records, module):
    """Stop when input for the Sequence model is not peptide sequences.

    Without this the composition extractor simply keeps whatever standard
    amino acid letters it can find and drops the rest, so a SMILES string
    such as 'CC(C)CC(NC(=O)...' yields a plausible-looking HC50 computed from
    the C, N and a few other letters in it. That is a wrong number, not an
    error, which is worse than refusing the run.
    """
    offenders = [
        header for header, seq in records
        if not looks_like_peptide(seq)
    ]

    if not offenders:
        return

    shown = ", ".join(offenders[:5])
    if len(offenders) > 5:
        shown += f", ... ({len(offenders) - 5} more)"

    sys.exit(
        f"Error: the {module} needs peptide sequences in MAP notation, but "
        f"{len(offenders)} of {len(records)} entries are not peptide "
        f"sequences:\n    {shown}\n"
        "  A SMILES string belongs to the Chemical model: run the same file "
        "with -m 2.\n"
        "  A peptide sequence uses the 20 one-letter amino acid codes, with "
        "modifications in braces, e.g. {nt:Acet}KWKLFKKIL{ct:Amid}"
    )


def hc50_label(hc50_values):
    """HC50 above the 100 uM cut-off is non-hemolytic."""
    return [
        "Non-hemolytic" if v > HC50_THRESHOLD else "Hemolytic"
        for v in hc50_values
    ]


def feature_guard(df_features, cols):
    """Refuse to predict when the model and the feature code disagree.

    make_feature_matrix() selects features by NAME and silently replaces
    anything it cannot find with a constant median. A name mismatch between
    the artifact and the extractor therefore does not raise: it quietly
    flattens every prediction. Better to stop than to write a table of
    plausible but wrong numbers.
    """
    if cols is None:
        return

    produced = set(df_features.columns) - set(metadata_columns(df_features))
    missing = [c for c in map(str, cols) if c not in produced]

    if missing:
        sys.exit(
            "Error: feature mismatch between the model and the feature code.\n"
            f"  The model expects {len(cols)} features but {len(missing)} of "
            f"them are never computed:\n    {missing}\n"
            "  This usually means the model file in models/ and this script "
            "are from different versions."
        )


def resolve_model(model_type):
    model_path = MODEL_PATHS[model_type]

    if not os.path.exists(model_path):
        sys.exit(
            f"Error: model file not found: {model_path}\n"
            "  The three model files belong in the models/ directory. See "
            "README.md."
        )

    try:
        return load_artifact(model_path)
    except Exception as exc:
        sys.exit(f"Error loading model artifact: {exc}")


def write_results(df, out_path, display_mode):
    df.to_csv(out_path, index=False)

    if display_mode == 1 and "Hemolytic_Activity" in df.columns:
        shown = df[df["Hemolytic_Activity"] == "Hemolytic"]
    else:
        shown = df

    # A peptide SMILES runs to several hundred characters and would push every
    # other column off the terminal, so it is shortened for the screen only.
    # The CSV written above always holds the full string.
    if "SMILES" in shown.columns:
        shown = shown.copy()
        shown["SMILES"] = [
            s if len(str(s)) <= 40 else str(s)[:37] + "..."
            for s in shown["SMILES"]
        ]

    with pd.option_context("display.width", 200,
                           "display.max_columns", 50,
                           "display.max_rows", 50):
        print()
        print(shown.to_string(index=False))

    print(f"\nResults saved to {out_path}")


# ===================== PREDICTION MODULE =====================

def run_prediction(args, model_type, out_path, display_mode):
    print(
        f"\n======= You are using the Prediction Module of HemoPI-MOD2. "
        f"Your results will be stored in file: {out_path} =======\n"
    )

    model, scaler, cols, medians, _ = resolve_model(model_type)

    # Set by the Chemical branch: the MAP sequence each entry was supplied as.
    map_given = {}

    if model_type == "Sequence":
        print("==== Extracting Sequence features "
              "(AAC + D/L counts + D ratio + MAP tag presence) ...")

        records = load_input_sequences(args.input)
        require_peptide_input(records, "Sequence model")

        df_features = extract_seq_features(records)

        print(f"     Predictive features: {df_features.shape[1] - 2}")

        feature_guard(df_features, cols)

    elif model_type == "Chemical":
        print("==== Processing Chemical input "
              "(MAP sequences or direct SMILES -> 2D Mordred) ...")

        try:
            import stitch
        except ImportError:
            stitch = None

        records = load_input_sequences(args.input)

        user_db = map_db = nnr_db = None
        headers, smiles_list = [], []

        # The MAP sequence each entry was supplied as, so the output can show
        # the input the way it was given. Empty for an entry supplied as
        # SMILES, which has no MAP form.
        map_given = {}

        for header, item in records:
            item_str = item.strip()

            # Each entry is recognised on its own. A SMILES string and a MAP
            # peptide sequence cannot be confused: is_valid_smiles() rejects
            # anything holding MAP braces or made only of one-letter amino
            # acid codes, and accepts only what RDKit can parse as a molecule.
            # A MAP sequence is converted to SMILES by stitch.py first, so
            # both reach the model as the same kind of input.
            if is_valid_smiles(item_str):
                smiles_list.append(item_str)
                headers.append(header)
                map_given[header] = ""
                continue

            if stitch is None:
                sys.exit(
                    "Error: stitch.py is required for processing peptide "
                    "sequences with the Chemical model."
                )

            if not os.path.exists(DB_PATH):
                sys.exit(f"Error: database file missing at {DB_PATH}")

            if user_db is None:
                user_db, map_db, nnr_db = stitch.load_databases(DB_PATH)

            try:
                smiles_list.append(
                    stitch.sequence_to_smiles(item_str, user_db, map_db, nnr_db)
                )
                headers.append(header)
                map_given[header] = item_str
            except Exception as exc:
                print(f"Warning: could not process sequence for "
                      f"{header}: {exc}")

        df_features = compute_mordred_2d(smiles_list, headers)

    else:  # Structure
        print("==== Processing PDB -> SMILES -> 2D Mordred descriptors ...")

        if os.path.isdir(args.input):
            pdb_files = [
                os.path.join(args.input, name)
                for name in sorted(os.listdir(args.input))
                if name.lower().endswith(".pdb")
            ]
            if not pdb_files:
                sys.exit(f"Error: no .pdb files found in {args.input}")
        else:
            pdb_files = [args.input]

        headers, smiles_list = [], []

        for pdb_file in pdb_files:
            header = os.path.splitext(os.path.basename(pdb_file))[0]
            try:
                smiles_list.append(pdb_to_smiles(pdb_file))
                headers.append(header)
            except Exception as exc:
                print(f"Skipping {pdb_file}: {exc}")

        df_features = compute_mordred_2d(smiles_list, headers)

    if df_features.empty:
        sys.exit("Error: feature matrix is empty; no predictions were made.")

    print(f"\n==== Running regression inference with the {model_type} "
          f"model, please wait ...")

    try:
        X = make_feature_matrix(df_features, model, scaler, cols, medians)
        predictions = np.asarray(model.predict(X), dtype=float)
    except Exception as exc:
        sys.exit(f"Error during prediction execution: {exc}")

    hc50 = np.power(10.0, -predictions)

    out = df_features[["ID"]].copy()

    if "Sequence" in df_features.columns:
        out["Sequence"] = df_features["Sequence"].values

    if model_type == "Chemical":
        # Report the input the way it was supplied. An entry given as a MAP
        # sequence shows that sequence next to the SMILES built from it; an
        # entry given as SMILES has no MAP form, so it shows the SMILES only.
        # A file holding nothing but SMILES therefore gets no Sequence column
        # at all.
        map_seqs = [map_given.get(i, "") for i in df_features["ID"]]
        if any(map_seqs):
            out["Sequence"] = map_seqs
        out["SMILES"] = df_features["SMILES"].values

    out["pHC50"] = np.round(predictions, 4)
    out["Prediction"] = np.round(hc50, 3)
    out["Hemolytic_Activity"] = hc50_label(hc50)

    write_results(out, out_path, display_mode)


# ===================== DESIGN MODULE =====================

def run_design(args, out_path, display_mode):
    print(
        f"\n======= You are using the Design Module of HemoPI-MOD2. "
        f"Your results will be stored in file: {out_path} =======\n"
    )

    # The design scan is a Sequence (MAP) model scan, always. Nothing here
    # reads args.model: main() has already refused any other model.
    model, scaler, cols, medians, _ = resolve_model("Sequence")

    records = load_input_sequences(args.input)
    require_peptide_input(records, "Design module")

    try:
        variants = all_mutants(records)
    except ValueError as exc:
        sys.exit(f"Error: {exc}")

    n_leads = len(records)
    n_mut = sum(0 if v["is_wt"] else 1 for v in variants)
    print(f"==== Leads: {n_lead_str(n_leads)}  ->  {n_mut} mutants "
          f"+ {n_leads} wild type")
    print("==== Predicting hemolytic activity using the Sequence (MAP) "
          "model: processing variants, please wait ...")

    df_features = extract_seq_features(
        [(v["Mutant_ID"], v["Sequence"]) for v in variants]
    )

    feature_guard(df_features, cols)

    try:
        X = make_feature_matrix(df_features, model, scaler, cols, medians)
        predictions = np.asarray(model.predict(X), dtype=float)
    except Exception as exc:
        sys.exit(f"Error during inference: {exc}")

    # Same target convention as the Prediction module: HC50 = 10^(-pHC50)
    hc50 = np.power(10.0, -predictions)

    out = pd.DataFrame({
        "SeqID":          [v["SeqID"] for v in variants],
        "Mutant_ID":      [v["Mutant_ID"] for v in variants],
        "Sequence":       [v["Sequence"] for v in variants],
        "Position":       [v["Position"] for v in variants],
        "Wild_Residue":   [v["Wild_Residue"] for v in variants],
        "Mutant_Residue": [v["Mutant_Residue"] for v in variants],
        "pHC50":          np.round(predictions, 4),
        "Prediction":     np.round(hc50, 3),
    })

    # Fold change is measured against each lead's own wild type.
    wt_hc50 = {
        v["lead"]: float(h)
        for v, h in zip(variants, hc50) if v["is_wt"]
    }
    leads = [v["lead"] for v in variants]
    ref = np.array([wt_hc50[l] for l in leads], dtype=float)

    out["Fold_Change_vs_WT"] = np.round(hc50 / ref, 3)

    # HC50 up = less hemolytic
    out["Direction"] = [
        "-" if v["is_wt"] else
        ("Less hemolytic" if fc > 1.0 else
         "More hemolytic" if fc < 1.0 else "No change")
        for v, fc in zip(variants, out["Fold_Change_vs_WT"])
    ]

    out["Hemolytic_Activity"] = hc50_label(out["Prediction"])

    for k, (seq_id, _) in enumerate(records, 1):
        print(f"     Wild-type HC50 for {seq_id}: {wt_hc50[k]:.3f} uM")

    write_results(out, out_path, display_mode)


def n_lead_str(n):
    return f"{n} peptide" if n == 1 else f"{n} peptides"


# ===================== MAIN =====================

def main():
    parser = argparse.ArgumentParser(
        description="Please provide following arguments for predicting and "
                    "designing hemolytic peptides with HemoPI-MOD2."
    )

    parser.add_argument(
        "-i", "--input", type=str, required=True,
        help="Input: peptide sequence in MAP/FASTA format or single sequence "
             "per line; SMILES for the Chemical model; a PDB file or a folder "
             "of PDB files for the Structure model"
    )

    parser.add_argument(
        "-o", "--output", type=str, default="output.csv",
        help="Output: file for saving results, by default output.csv"
    )
    parser.add_argument(
        "-j", "--job", type=int, choices=[1, 2], default=1,
        help="Job Type: 1: Prediction, 2: Design all possible mutants"
    )
    parser.add_argument(
        "-m", "--model", type=int, choices=[1, 2, 3], default=1,
        help="Model: 1: Sequence (MAP), 2: Chemical (SMILES), 3: Structure (PDB)"
    )
    parser.add_argument(
        "-d", "--display", type=int, choices=[1, 2], default=2,
        help="Display: 1: hemolytic peptides only, 2: all peptides, "
             "by default 2"
    )
    parser.add_argument(
        "-wd", "--working", type=str, default=os.getcwd(),
        help="Working directory for saving results (optional)."
    )

    args = parser.parse_args()

    print()
    print('##########################################################################################')
    print('#                                                                                        #')
    print('#  🩸 HemoPI-MOD2: A Tool to Predict and Design Hemolytic Peptides                       #')
    print('#                                                                                        #')
    print('#  This program predicts the hemolytic activity (HC50) of natural and chemically         #')
    print('#  modified peptides from their sequence, chemical structure (SMILES) or 3D              #')
    print('#  structure (PDB), and designs all possible single-residue analogues.                   #')
    print('#                                                                                        #')
    print("#  🧬 Developed by Prof. G. P. S. Raghava's group at IIIT-Delhi                          #")
    print('#                                                                                        #')
    print('##########################################################################################')

    # ---------------- Parameter initialization ----------------

    Sequence = args.input
    result_filename = args.output or "output.csv"
    Job = args.job
    Model = args.model
    dplay = int(args.display)
    wd = args.working

    model_type = MODEL_CHOICE_MAP[Model]

    # Job 1 (Prediction) runs on any of the three models. Job 2 (Design) runs
    # on the Sequence (MAP) model and nothing else: a Chemical scan would need
    # a SMILES and descriptor pass for every mutant, and a Structure scan a
    # folded PDB for every mutant, so neither is practical for a position scan.
    if Job == 2 and Model != 1:
        parser.error(
            "job type 2 (Design) runs on the Sequence (MAP) model only, so "
            "-m 1 is the only valid model with -j 2. A Chemical or Structure "
            "scan would need a new structure for every mutant."
        )

    if not os.path.exists(Sequence):
        sys.exit(f"Error: input '{Sequence}' not found.")

    if not os.path.isdir(wd):
        sys.exit(f"Error: working directory '{wd}' does not exist.")

    # An -o carrying its own directory is honoured as given; a bare file name
    # is written inside the working directory.
    if os.path.dirname(result_filename):
        out_path = os.path.abspath(result_filename)
        out_dir = os.path.dirname(out_path)
        if not os.path.isdir(out_dir):
            sys.exit(f"Error: output directory '{out_dir}' does not exist.")
    else:
        out_path = os.path.join(wd, result_filename)

    print("\nSummary of Parameters:")
    print(f"Input File: {Sequence} ; Model: {Model} ({model_type}) ; "
          f"Job: {Job} ({JOB_NAMES[Job]})")
    print(f"Output File: {out_path} ; Display: {dplay} ; "
          f"Hemolytic cut-off: {HC50_THRESHOLD:g} uM")

    if Job == 1:
        run_prediction(args, model_type, out_path, dplay)
    else:
        run_design(args, out_path, dplay)


if __name__ == "__main__":
    main()
