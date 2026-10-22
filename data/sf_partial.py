
import numpy as np
import freud
from ase.io import read
from tqdm import tqdm

trajectory_file = "traj"
output_prefix = "SF"

species = "Li"
stride = 5
k_max = 10.0
k_min = 0.0
bins = 200

frames = read(trajectory_file, index=f"::{stride}")

def ase_to_freud(atoms):

    cell = np.asarray(atoms.cell.array, dtype=float)
    positions = np.asarray(atoms.positions, dtype=float)

    if np.linalg.det(cell) <= 0:
        raise ValueError("Expected a right-handed 3D cell.")


    A = cell.T


    Q, R = np.linalg.qr(A)


    signs = np.where(np.diag(R) < 0, -1.0, 1.0)
    D = np.diag(signs)
    Q = Q @ D
    R = D @ R


    transformed_positions = positions @ Q

    box = freud.box.Box.from_matrix(R)

    return box, transformed_positions

ssf = freud.diffraction.StaticStructureFactorDirect(
    k_max=k_max,
    k_min=k_min,
    bins=bins
)

results = []
atom_counts = []

for atoms in tqdm(frames):
    symbols = np.asarray(atoms.get_chemical_symbols())
    mask = symbols == species

    box, positions = ase_to_freud(atoms)
    selected = positions[mask]

    atom_counts.append(len(selected))

    ssf.compute(
        (box, selected),
        reset=True
    )
    results.append(ssf.S_k.copy())

Sk_avg = np.mean(results, axis=0)
k = ssf.bin_centers

np.savetxt(
    f"{output_prefix}.txt",
    np.column_stack([k, Sk_avg]),
    header=f"k(1/A) S_{species}{species}(k)"
)

print("Frames:", len(results))
print("Mean atoms/frame:", np.mean(atom_counts))
print("Freud minimum valid k:", ssf.min_valid_k)

