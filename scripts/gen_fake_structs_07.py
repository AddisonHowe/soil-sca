"""

"""

def write_toy_pdb(seq, fname):
    atom_template = "ATOM  {atom_id:5d} {atom_name:<4s}{res_name:>3s} A{res_id:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
    atom_id = 1
    res_id = 1
    lines = []

    aa_map = {
        "A": "ALA",
        "C": "CYS",
        "D": "ASP",
        "E": "GLU"
    }

    # We'll use a fixed set of atoms per residue
    atoms_map = {
        "ALA": ["N", "CA", "C", "O"],
        "CYS": ["N", "CA", "C", "O"],
        "ASP": ["N", "CA", "C", "O"],
        "GLU": ["N", "CA", "C", "O"],
    }

    for i, aa in enumerate(seq):
        if aa == "-":
            continue  # skip gaps
        res_name = aa_map.get(aa, "GLY")
        # toy coordinates: line along x-axis
        x = float(i)
        z = 0.0
        for j, atom in enumerate(atoms_map[res_name]):
            y = float(j)
            line = atom_template.format(
                atom_id=atom_id, 
                atom_name=atom,
                res_name=res_name,
                res_id=res_id, 
                x=x, y=y, z=z
            )
            lines.append(line)
            atom_id += 1
        res_id += 1

    lines.append("END")
    with open(fname, "w") as f:
        f.write("\n".join(lines) + "\n")


msafile = "tests/_data/msas/msa07.faa"
outdir = "tests/_data/structs/structs07"

with open(msafile, "r") as f:
    lines = f.readlines()
    for i in range(len(lines) // 2):
        line = lines[2*i]
        name = line[1:-1]
        print(name)
        line = lines[2*i + 1]
        seq = line[0:-1]
        print(seq)
        write_toy_pdb(seq, f"{outdir}/{name}.pdb")
        