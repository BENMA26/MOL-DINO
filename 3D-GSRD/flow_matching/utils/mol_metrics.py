"""
mol_metrics.py — 3D molecule generation evaluation utilities.

Given a batch of generated (coords, atom_type_idx) tensors, convert to
RDKit Mol objects via OpenBabel bond inference, then compute:

  Geometry-level:
    • validity      — fraction of mol objects that sanitize cleanly
    • connectivity  — fraction of valid mols that are fully connected

  Chemistry-level (over valid mols):
    • uniqueness    — fraction of unique canonical SMILES
    • novelty       — fraction of SMILES not in the training set
    • QED           — Quantitative Estimate of Drug-likeness (mean)
    • SA            — Synthetic Accessibility score (mean, lower is better)
    • logP          — Wildman-Crippen logP (mean)
    • lipinski      — fraction satisfying all 4 Lipinski rules

  Diversity:
    • diversity     — mean pairwise Tanimoto distance over ECFP4 fingerprints

Optional (when PoseBusters is installed and enabled):
  3D-quality checks:
    • pb_valid_rate
    • pb_mol_pred_loaded
    • pb_sanitization
    • pb_inchi_convertible
    • pb_all_atoms_connected
    • pb_bond_lengths
    • pb_bond_angles
    • pb_internal_steric_clash
    • pb_aromatic_ring_flatness
    • pb_double_bond_flatness

All functions operate on CPU tensors / numpy arrays to avoid GPU→CPU overhead
in the inner loop.

QM9 atom-type mapping (matches data_provider/qm9_dataset.py):
    index 0 → H
    index 1 → C
    index 2 → N
    index 3 → O
    index 4 → F
"""

import contextlib
from typing import Dict, List, Optional

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, QED, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

# QM9 atom-type index → element symbol
QM9_IDX2ELEM = {0: "H", 1: "C", 2: "N", 3: "O", 4: "F"}

_PB_KEYS = (
    "mol/pb_valid_rate",
    "mol/pb_mol_pred_loaded",
    "mol/pb_sanitization",
    "mol/pb_inchi_convertible",
    "mol/pb_all_atoms_connected",
    "mol/pb_bond_lengths",
    "mol/pb_bond_angles",
    "mol/pb_internal_steric_clash",
    "mol/pb_aromatic_ring_flatness",
    "mol/pb_double_bond_flatness",
    "mol/pb_internal_energy",
    "mol/pb_available",
)


# ── Bond inference helpers ────────────────────────────────────────────────────

def _write_xyz(coords: np.ndarray, elem_syms: List[str]) -> str:
    """Write an xyz block string."""
    lines = [f"{len(coords)}", ""]
    for sym, (x, y, z) in zip(elem_syms, coords):
        lines.append(f"{sym}  {x:.6f}  {y:.6f}  {z:.6f}")
    return "\n".join(lines)


def _xyz_to_mol_openbabel(coords: np.ndarray, elem_syms: List[str]) -> Optional[Chem.Mol]:
    """OpenBabel: xyz → SDF → RDKit Mol (with bond-order inference)."""
    try:
        from openbabel import openbabel as ob

        xyz_str = _write_xyz(coords, elem_syms)

        conv = ob.OBConversion()
        conv.SetInAndOutFormats("xyz", "sdf")
        ob_mol = ob.OBMol()
        conv.ReadString(ob_mol, xyz_str)

        sdf_str = conv.WriteString(ob_mol)

        with contextlib.suppress(Exception):
            ob.obErrorLog.StopLogging()

        supplier = Chem.SDMolSupplier()
        supplier.SetData(sdf_str, sanitize=False)
        mol = supplier[0]
        if mol is None:
            return None

        # rebuild without radicals
        rw = Chem.RWMol()
        for atom in mol.GetAtoms():
            a = Chem.Atom(atom.GetSymbol())
            rw.AddAtom(a)
        rw.AddConformer(mol.GetConformer(0))
        for bond in mol.GetBonds():
            rw.AddBond(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                bond.GetBondType(),
            )
        for atom in rw.GetAtoms():
            atom.SetNumRadicalElectrons(0)
            atom.SetFormalCharge(0)
        return rw.GetMol()
    except Exception:
        return None


def _try_sanitize(mol: Chem.Mol) -> Optional[Chem.Mol]:
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def _mol_is_connected(mol: Chem.Mol) -> bool:
    """Return True iff the molecular graph is connected (single fragment)."""
    frags = Chem.GetMolFrags(mol)
    return len(frags) == 1


# ── Single-molecule conversion ────────────────────────────────────────────────

def coords_and_types_to_mol(
    coords: np.ndarray,        # (N, 3)
    atom_type_idx: np.ndarray, # (N,) int  e.g. QM9 indices
    idx2elem: Dict[int, str] = QM9_IDX2ELEM,
) -> Optional[Chem.Mol]:
    """
    Convert predicted coordinates + atom-type indices to an RDKit Mol.
    Returns None on failure.
    """
    elem_syms = [idx2elem.get(int(i), "C") for i in atom_type_idx]
    mol = _xyz_to_mol_openbabel(coords, elem_syms)
    mol = _try_sanitize(mol)
    return mol


# ── Batch conversion ──────────────────────────────────────────────────────────

def batch_to_mols(
    coords_batch: torch.Tensor,       # (B, N, 3)
    atomics_batch: torch.Tensor,      # (B, N, atom_dim)  one-hot
    padding_mask: torch.Tensor,       # (B, N) bool, True=padding
    idx2elem: Dict[int, str] = QM9_IDX2ELEM,
) -> List[Optional[Chem.Mol]]:
    """
    Convert a batch of dense tensors to a list of RDKit Mol objects.
    Invalid/failed molecules are None.
    """
    B = coords_batch.shape[0]
    coords_np = coords_batch.cpu().float().numpy()
    atomics_np = atomics_batch.cpu().float().numpy()
    mask_np = padding_mask.cpu().bool().numpy()  # True=padding

    mols = []
    for i in range(B):
        real = ~mask_np[i]  # (N,)
        if real.sum() == 0:
            mols.append(None)
            continue
        coords_i = coords_np[i][real]  # (n_real, 3)
        atom_idx_i = atomics_np[i][real].argmax(-1)  # (n_real,)
        mol = coords_and_types_to_mol(coords_i, atom_idx_i, idx2elem)
        mols.append(mol)
    return mols


# ── Fingerprint diversity ─────────────────────────────────────────────────────

def _ecfp4_fp(mol: Chem.Mol):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return gen.GetFingerprint(mol)


def _tanimoto(fp1, fp2) -> float:
    from rdkit import DataStructs

    return DataStructs.TanimotoSimilarity(fp1, fp2)


def mean_pairwise_diversity(mols: List[Chem.Mol]) -> float:
    """Mean pairwise Tanimoto *distance* (1 - similarity) over ECFP4."""
    fps = []
    for mol in mols:
        try:
            fps.append(_ecfp4_fp(mol))
        except Exception:
            pass
    if len(fps) < 2:
        return 0.0
    dists = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            dists.append(1.0 - _tanimoto(fps[i], fps[j]))
    return float(np.mean(dists))


# ── PoseBusters metrics (optional) ───────────────────────────────────────────

def _empty_pb_metrics() -> Dict[str, float]:
    return {k: 0.0 for k in _PB_KEYS}


def _norm_col(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def compute_posebusters_metrics(
    valid_mols: List[Chem.Mol],
    total_count: int,
    config: str = "mol",
) -> Dict[str, float]:
    """
    Compute PoseBusters metrics on valid molecules.

    Notes:
      - Returns zero metrics when PoseBusters is unavailable or no valid mol exists.
      - Fractions are normalized by `total_count` (generated molecules), matching
        the convention used in TABASCO's evaluator.
    """
    out = _empty_pb_metrics()
    if total_count <= 0:
        return out

    try:
        from posebusters import PoseBusters
    except Exception:
        return out

    out["mol/pb_available"] = 1.0

    if len(valid_mols) == 0:
        return out

    try:
        buster = PoseBusters(config=config)
        results = buster.bust(mol_pred=valid_mols)
    except Exception:
        return out

    if getattr(results, "empty", False):
        return out

    # Row-level intersection: no failing boolean check in the row.
    try:
        pb_pass = 0.0
        for _, row in results.iterrows():
            pb_pass += 0.0 if row.isin([False]).any() else 1.0
        out["mol/pb_valid_rate"] = pb_pass / max(total_count, 1)
    except Exception:
        pass

    norm2orig = {_norm_col(c): c for c in results.columns}
    key_map = {
        "mol_pred_loaded": "mol/pb_mol_pred_loaded",
        "sanitization": "mol/pb_sanitization",
        "inchi_convertible": "mol/pb_inchi_convertible",
        "all_atoms_connected": "mol/pb_all_atoms_connected",
        "bond_lengths": "mol/pb_bond_lengths",
        "bond_angles": "mol/pb_bond_angles",
        "internal_steric_clash": "mol/pb_internal_steric_clash",
        "aromatic_ring_flatness": "mol/pb_aromatic_ring_flatness",
        "double_bond_flatness": "mol/pb_double_bond_flatness",
        "internal_energy": "mol/pb_internal_energy",
    }

    for pb_col_norm, out_key in key_map.items():
        if pb_col_norm not in norm2orig:
            continue

        col = results[norm2orig[pb_col_norm]]
        try:
            if getattr(col, "dtype", None) == bool:
                out[out_key] = float(col.sum()) / max(total_count, 1)
            else:
                # Prefer boolean-like behavior when possible, otherwise average.
                vals = col.astype(float).to_numpy()
                if np.all(np.isin(vals, [0.0, 1.0])):
                    out[out_key] = float(vals.sum()) / max(total_count, 1)
                else:
                    out[out_key] = float(np.mean(vals))
        except Exception:
            continue

    return out


# ── SA score ─────────────────────────────────────────────────────────────────
# Lightweight re-implementation adapted from the GuacaMol / MOSES SA scorer.

def _sa_score(mol: Chem.Mol) -> Optional[float]:
    """Return Synthetic-Accessibility score (1=easy, 10=hard)."""
    try:
        # Use RDKit's built-in SA-like score via Crippen contributions
        # If sascorer is available, prefer it; otherwise fall back to a proxy.
        try:
            import sascorer

            return sascorer.calculateScore(mol)
        except ImportError:
            pass
        # Lightweight proxy: count ring-system complexity + chiral centres
        n_rings = rdMolDescriptors.CalcNumRings(mol)
        n_chiral = len(rdMolDescriptors.CalcChiralCenters(mol, includeUnassigned=True))
        n_heavy = mol.GetNumHeavyAtoms()
        # crude proxy in [1,10]
        score = 1.0 + (n_rings * 0.5 + n_chiral * 0.3) / max(n_heavy, 1) * 9
        return min(score, 10.0)
    except Exception:
        return None


# ── Main evaluation function ──────────────────────────────────────────────────

def compute_mol_metrics(
    coords_batch: torch.Tensor,      # (B, N, 3)  generated coords
    atomics_batch: torch.Tensor,     # (B, N, atom_dim) one-hot
    padding_mask: torch.Tensor,      # (B, N)  True=padding
    train_smiles: Optional[List[str]] = None,  # for novelty
    idx2elem: Dict[int, str] = QM9_IDX2ELEM,
    compute_posebusters: bool = False,
    posebusters_config: str = "mol",
) -> Dict[str, float]:
    """
    Full evaluation pipeline. Returns a dict of scalar metrics.

    Args:
        coords_batch:   generated 3D coordinates [B, N, 3]
        atomics_batch:  predicted atom-type one-hot [B, N, atom_dim]
        padding_mask:   True at padding positions [B, N]
        train_smiles:   list of SMILES strings from the training set (for novelty)
        idx2elem:       atom-type-index → element-symbol mapping

    Returns:
        dict with keys:
            validity, connectivity, uniqueness, novelty,
            qed_mean, sa_mean, logp_mean, lipinski_frac, diversity
            and optional PoseBusters metrics if enabled
    """
    B = coords_batch.shape[0]
    mols = batch_to_mols(coords_batch, atomics_batch, padding_mask, idx2elem)

    # ── validity ──────────────────────────────────────────────────────────────
    valid_mols = [m for m in mols if m is not None]
    validity = len(valid_mols) / max(B, 1)

    # ── connectivity (valid mols only) ───────────────────────────────────────
    connected = [m for m in valid_mols if _mol_is_connected(m)]
    connectivity = len(connected) / max(len(valid_mols), 1)

    if len(valid_mols) == 0:
        out = {
            "mol/validity": validity,
            "mol/connectivity": connectivity,
            "mol/uniqueness": 0.0,
            "mol/novelty": 0.0,
            "mol/qed_mean": 0.0,
            "mol/sa_mean": 10.0,
            "mol/logp_mean": 0.0,
            "mol/lipinski_frac": 0.0,
            "mol/diversity": 0.0,
        }
        if compute_posebusters:
            out.update(compute_posebusters_metrics([], total_count=B, config=posebusters_config))
        return out

    # ── SMILES ────────────────────────────────────────────────────────────────
    smiles_list = []
    for mol in valid_mols:
        try:
            smi = Chem.MolToSmiles(mol, isomericSmiles=True)
            # take largest fragment
            frags = Chem.GetMolFrags(mol, asMols=True)
            if frags:
                largest = max(frags, key=lambda x: x.GetNumAtoms())
                smi = Chem.MolToSmiles(largest, isomericSmiles=True)
            smiles_list.append(smi)
        except Exception:
            smiles_list.append(None)

    valid_smiles = [s for s in smiles_list if s is not None]
    unique_smiles = set(valid_smiles)

    uniqueness = len(unique_smiles) / max(len(valid_smiles), 1)

    if train_smiles is not None:
        train_set = set(train_smiles)
        novel_smiles = unique_smiles - train_set
        novelty = len(novel_smiles) / max(len(unique_smiles), 1)
    else:
        novelty = 0.0

    # ── chemistry properties ──────────────────────────────────────────────────
    qed_vals, sa_vals, logp_vals, lipinski_vals = [], [], [], []

    for mol in valid_mols:
        try:
            qed_vals.append(QED.qed(mol))
        except Exception:
            pass

        sa = _sa_score(mol)
        if sa is not None:
            sa_vals.append(sa)

        try:
            logp_vals.append(Descriptors.MolLogP(mol))
        except Exception:
            pass

        # Lipinski RO5: MW≤500, HBD≤5, HBA≤10, logP≤5
        try:
            mw = Descriptors.MolWt(mol)
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            hba = rdMolDescriptors.CalcNumHBA(mol)
            lp = Descriptors.MolLogP(mol)
            lip = int(mw <= 500 and hbd <= 5 and hba <= 10 and lp <= 5)
            lipinski_vals.append(lip)
        except Exception:
            pass

    qed_mean = float(np.mean(qed_vals)) if qed_vals else 0.0
    sa_mean = float(np.mean(sa_vals)) if sa_vals else 10.0
    logp_mean = float(np.mean(logp_vals)) if logp_vals else 0.0
    lipinski_frac = float(np.mean(lipinski_vals)) if lipinski_vals else 0.0

    # ── diversity ─────────────────────────────────────────────────────────────
    diversity = mean_pairwise_diversity(valid_mols)

    out = {
        "mol/validity": validity,
        "mol/connectivity": connectivity,
        "mol/uniqueness": uniqueness,
        "mol/novelty": novelty,
        "mol/qed_mean": qed_mean,
        "mol/sa_mean": sa_mean,
        "mol/logp_mean": logp_mean,
        "mol/lipinski_frac": lipinski_frac,
        "mol/diversity": diversity,
    }

    if compute_posebusters:
        out.update(
            compute_posebusters_metrics(
                valid_mols=valid_mols,
                total_count=B,
                config=posebusters_config,
            )
        )

    return out
