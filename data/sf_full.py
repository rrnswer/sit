import numpy as np
import matplotlib.pyplot as plt
import freud
from ase.io import read
from tqdm import tqdm

trajectory_file = "traj"
output_prefix = "SF"

stride = 5

k_max = 10
k_min = 0
bins = 200

frames = read(trajectory_file, index="::%d" % stride)
print(f"Loaded {len(frames)} frames")

ssf = freud.diffraction.StaticStructureFactorDirect(
    k_max=k_max, k_min=k_min, bins=bins
)

Sk_accumulate = None
n_frames = 0

for atoms in tqdm(frames, desc="Computing S(k)"):
    positions = atoms.get_positions()
    cell = atoms.get_cell().array
    box = freud.box.Box.from_matrix(cell)

    ssf.compute((box, positions), reset=False)

    if Sk_accumulate is None:
        Sk_accumulate = ssf.S_k.copy()
    else:
        Sk_accumulate += ssf.S_k

    n_frames += 1

Sk_avg = Sk_accumulate / n_frames

k = ssf.bin_centers

np.savetxt(
    f"{output_prefix}.txt",
    np.column_stack([k, Sk_avg]),
    header="k(1/A)   S(k)"
)


