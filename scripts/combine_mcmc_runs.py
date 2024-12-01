import numpy as np
import os

sampler = "gwg"
temp_marg = 0.01
temp_prop = 2.0
l_J = 0.01
l_h = 0.01

burnin = 0

all_data = []
total = 0

for seed in [11, 3,7,31]:

    dir = f'../gen_data/potts_init_{sampler}_seed_{seed}_J_{l_J}_h_{l_h}_lr_0.001_T_{temp_marg}'+(f'_Tp_{temp_prop}' if sampler == "gwg" else "")

    with open(os.path.join(dir, "potts.fasta"), "r") as file:
        lines = file.readlines()

    lines = [l.strip() for l in lines if not l.startswith(">")]
    lines = np.unique(lines[burnin:])

    all_data = np.concatenate([all_data, lines])

    print(f"Added {len(lines)} lines")
    print(f"Seed {seed} done")

all_data=np.unique(all_data)
print(f"Total length: {len(all_data)}")

with open(f"../gen_data/potts_adam_{sampler}_T_{temp_marg}_lr_0.001_unified.fasta", "w") as file:
    
    for l in all_data:
        file.write(f">\n{l}\n")