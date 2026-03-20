"""
eval_dino_gen.py  –  evaluate molecule generation with DINO-encoder + flow matching on QM9.

Pipeline:
  1. Load trained DinoAELitModule checkpoint (autoencoder)
  2. Load trained flow matching checkpoint (denoiser in latent space)
  3. Sample N molecules: noise → flow → decode → stability metrics
  4. Print: atom_stability, mol_stability, validity, uniqueness, novelty (UniGEM benchmark)

Usage:
    python eval_dino_gen.py \
        --ae_ckpt ./outputs/dino_ae_qm9_stage2/best.ckpt \
        --flow_ckpt ./outputs/dino_flow_qm9/best.ckpt \
        --ref_data_dir /path/to/qm9_train.pt \
        --n_samples 1000 \
        --num_steps 100 \
        --batch_size 64
"""

import argparse
import os
import sys

import torch
import numpy as np
from tqdm import tqdm
from rdkit import Chem

_here = os.path.dirname(os.path.abspath(__file__))
_tabasco_src = _here
if _tabasco_src not in sys.path:
    sys.path.insert(0, _tabasco_src)

from tabasco.data.components.lmdb_unconditional import UnconditionalLMDBDataset
from train_dino_ae import DinoAELitModule
from train_dino_flow import DinoFlowLitModule


# ---------------------------------------------------------------------------

def sample_molecules(ae_model, flow_model, ref_dataset, n_samples, num_steps, batch_size, device):
    """Generate molecules using the DINO AE + flow matching pipeline.

    Returns:
        List of dicts with keys 'positions' (N,3 numpy) and 'atom_types' (N, numpy indices)
    """
    ae_model.eval()
    flow_model.eval()

    molecules_list = []

    # collect reference padding masks from dataset for shape conditioning
    ref_loader = torch.utils.data.DataLoader(
        ref_dataset, batch_size=batch_size, shuffle=True
    )
    ref_iter = iter(ref_loader)

    generated = 0
    with torch.no_grad():
        while generated < n_samples:
            try:
                ref_batch = next(ref_iter)
            except StopIteration:
                ref_iter = iter(ref_loader)
                ref_batch = next(ref_iter)

            ref_batch = {k: v.to(device) for k, v in ref_batch.items() if isinstance(v, torch.Tensor)}
            padding_mask = ref_batch["padding_mask"]

            # encode reference to get z shape
            z_ref = ae_model.net.encode_z(
                ref_batch["coords"], ref_batch["atomics"], padding_mask, return_kl=False
            )

            # flow matching: noise → z_pred
            from tabasco.models.flow_model import FlowMatchingModel
            flow_net: FlowMatchingModel = flow_model.flow_model

            z_pred = flow_net.decode(
                z={"x": z_ref},
                padding_mask=padding_mask,
                num_steps=num_steps,
            )

            # decode z_pred → molecule
            pred_coords = z_pred["coords"]
            pred_atomics = z_pred["atomics"]

            B = pred_coords.shape[0]
            for b in range(B):
                mask = ~padding_mask[b]
                coords_b = pred_coords[b][mask].cpu().numpy()  # (N, 3)
                atomics_b = pred_atomics[b][mask].cpu().numpy()  # (N, A)
                atom_types = atomics_b.argmax(axis=-1)  # (N,) indices

                molecules_list.append({
                    'positions': coords_b,
                    'atom_types': atom_types,
                })

            generated += B
            print(f"Generated {min(generated, n_samples)}/{n_samples}", end="\r")

    return molecules_list[:n_samples]


def compute_metrics(molecules_list, ref_smiles):
    """
    Compute UniGEM-style metrics: atom_stability, mol_stability, validity, uniqueness, novelty.

    molecules_list: list of dicts with keys 'positions' (N,3) and 'atom_types' (N,)
    """
    # QM9 dataset info (5 atom types: H, C, N, O, F)
    dataset_info = {
        'name': 'qm9',
        'atom_decoder': ['H', 'C', 'N', 'O', 'F'],
        'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4},
    }

    # Import UniGEM stability checker
    unigem_path = os.path.join(os.path.dirname(__file__), '../../UniGEM')
    if unigem_path not in sys.path:
        sys.path.insert(0, unigem_path)

    try:
        from qm9.analyze import check_stability
    except ImportError:
        print("Warning: UniGEM not found, using simplified metrics")
        return compute_metrics_simple(molecules_list, ref_smiles)

    # Stability check
    n_stable_atoms = 0
    n_total_atoms = 0
    n_stable_mols = 0
    valid_smiles = []

    for mol_dict in molecules_list:
        positions = mol_dict['positions']  # (N, 3) numpy array
        atom_types = mol_dict['atom_types']  # (N,) numpy array of indices

        # Check stability using UniGEM's function
        mol_stable, nr_stable, total = check_stability(positions, atom_types, dataset_info)
        n_stable_atoms += nr_stable
        n_total_atoms += total
        n_stable_mols += int(mol_stable)

        # Convert to SMILES for validity/uniqueness/novelty
        try:
            # Build RDKit mol from positions and atom types
            mol = Chem.RWMol()
            for atom_idx in atom_types:
                atom_symbol = dataset_info['atom_decoder'][atom_idx]
                mol.AddAtom(Chem.Atom(atom_symbol))

            # Infer bonds from distances
            from rdkit.Chem import rdDetermineBonds
            conf = Chem.Conformer(len(atom_types))
            for i, pos in enumerate(positions):
                conf.SetAtomPosition(i, tuple(pos))
            mol.AddConformer(conf)

            rdDetermineBonds.DetermineBonds(mol, charge=0)
            mol = mol.GetMol()
            Chem.SanitizeMol(mol)
            smi = Chem.MolToSmiles(mol)
            if smi:
                valid_smiles.append(smi)
        except Exception:
            pass

    atom_stability = n_stable_atoms / max(n_total_atoms, 1)
    mol_stability = n_stable_mols / max(len(molecules_list), 1)
    validity = len(valid_smiles) / max(len(molecules_list), 1)

    unique_smiles = set(valid_smiles)
    uniqueness = len(unique_smiles) / max(len(valid_smiles), 1) if valid_smiles else 0

    novel = unique_smiles - set(ref_smiles)
    novelty = len(novel) / max(len(unique_smiles), 1) if unique_smiles else 0

    print(f"\n{'='*50}")
    print(f"QM9 Generation Metrics (UniGEM benchmark)")
    print(f"{'='*50}")
    print(f"Total generated    : {len(molecules_list)}")
    print(f"Atom stability     : {atom_stability:.4f}")
    print(f"Molecule stability : {mol_stability:.4f}")
    print(f"Validity           : {validity:.4f}")
    print(f"Uniqueness         : {uniqueness:.4f}")
    print(f"Novelty            : {novelty:.4f}")
    print(f"{'='*50}")

    return {
        "atom_stability": atom_stability,
        "mol_stability": mol_stability,
        "validity": validity,
        "uniqueness": uniqueness,
        "novelty": novelty,
    }


def compute_metrics_simple(molecules_list, ref_smiles):
    """Fallback if UniGEM not available."""
    valid_smiles = []

    dataset_info = {
        'atom_decoder': ['H', 'C', 'N', 'O', 'F'],
    }

    for mol_dict in molecules_list:
        positions = mol_dict['positions']
        atom_types = mol_dict['atom_types']

        try:
            mol = Chem.RWMol()
            for atom_idx in atom_types:
                atom_symbol = dataset_info['atom_decoder'][atom_idx]
                mol.AddAtom(Chem.Atom(atom_symbol))

            conf = Chem.Conformer(len(atom_types))
            for i, pos in enumerate(positions):
                conf.SetAtomPosition(i, tuple(pos))
            mol.AddConformer(conf)

            from rdkit.Chem import rdDetermineBonds
            rdDetermineBonds.DetermineBonds(mol, charge=0)
            mol = mol.GetMol()
            Chem.SanitizeMol(mol)
            smi = Chem.MolToSmiles(mol)
            if smi:
                valid_smiles.append(smi)
        except Exception:
            pass

    validity = len(valid_smiles) / max(len(molecules_list), 1)
    unique_smiles = set(valid_smiles)
    uniqueness = len(unique_smiles) / max(len(valid_smiles), 1) if valid_smiles else 0
    novel = unique_smiles - set(ref_smiles)
    novelty = len(novel) / max(len(unique_smiles), 1) if unique_smiles else 0

    print(f"\nTotal generated : {len(molecules_list)}")
    print(f"Validity        : {validity:.4f}")
    print(f"Uniqueness      : {uniqueness:.4f}")
    print(f"Novelty         : {novelty:.4f}")
    return {"validity": validity, "uniqueness": uniqueness, "novelty": novelty}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ae_ckpt", required=True, help="DinoAELitModule checkpoint")
    p.add_argument("--flow_ckpt", required=True, help="DinoFlowLitModule checkpoint")
    p.add_argument("--ref_data_dir", required=True, help="train .pt for novelty ref")
    p.add_argument("--lmdb_dir", default="./lmdb_cache")
    p.add_argument("--n_samples", type=int, default=1000)
    p.add_argument("--num_steps", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading autoencoder...")
    ae_model = DinoAELitModule.load_from_checkpoint(args.ae_ckpt, map_location=device)
    ae_model = ae_model.to(device)

    print("Loading flow model...")
    flow_model = DinoFlowLitModule.load_from_checkpoint(
        args.flow_ckpt, map_location=device
    )
    flow_model = flow_model.to(device)

    print("Loading reference dataset...")
    ref_ds = UnconditionalLMDBDataset(
        data_dir=args.ref_data_dir,
        split="train",
        lmdb_dir=os.path.join(args.lmdb_dir, "train"),
    )
    ref_smiles = ref_ds.all_smiles if hasattr(ref_ds, 'all_smiles') else []

    print(f"Sampling {args.n_samples} molecules...")
    molecules = sample_molecules(
        ae_model, flow_model, ref_ds,
        n_samples=args.n_samples,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        device=device,
    )

    compute_metrics(molecules, ref_smiles)


if __name__ == "__main__":
    main()
