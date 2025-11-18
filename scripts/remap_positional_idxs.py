"""Script to correct positional indices corresponding to sequences and PDB files.

"""

import argparse
import os, sys
import numpy as np


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datdir", type=str, required=True)
    parser.add_argument("-m", "--mapping_fpath", type=str, required=True)
    return parser.parse_args(args)


def main(args):
    datdir = args.datdir
    mapping_fpath = args.mapping_fpath

    sca_groups_dir = os.path.join(datdir, "sca_groups")
    pdb_sectors_dir = os.path.join(datdir, "pdb_sectors")
    new_sca_groups_dir = os.path.join(datdir, "sca_groups_remapped")
    new_pdb_sectors_dir = os.path.join(datdir, "pdb_sectors_remapped")
    
    scadir = os.path.join(datdir, "sca_results")
    # retained_positions = np.load(f"{scadir}/retained_positions.npy")

    mapping_arrays = np.load(mapping_fpath)        


    # Correct SCA group directory
    sca_groups_subdirlist = [
        d for d in os.listdir(sca_groups_dir) if d.startswith("group_")
    ]
    for i in range(len(sca_groups_subdirlist)):
        subdir = f"group_{i}"
        sca_groups_filelist = [
            f for f in os.listdir(os.path.join(sca_groups_dir, subdir))
            if f.startswith(f"group_{i}")
        ]
        outdir = os.path.join(new_sca_groups_dir, subdir)
        os.makedirs(outdir, exist_ok=True)
        count = 0
        for f in sca_groups_filelist:
            key = f.removesuffix(".npy").removeprefix(f"group_{i}_")
            if key not in mapping_arrays:
                continue
            
            mapping = mapping_arrays[key]
            fpath = os.path.join(sca_groups_dir, subdir, f)
            group_idxs_old = np.load(fpath)
            new_idxs = mapping[group_idxs_old]
            np.save(
                os.path.join(outdir, f"group_{i}_{key}.npy"),
                new_idxs
            )
            count += 1
        print(f"Subdir {subdir}. Saved {count} files")


    # Correct PDB sector directory
    pdb_sectors_subdirlist = [
        d for d in os.listdir(pdb_sectors_dir) if d.startswith("sector_")
    ]
    for i in range(len(pdb_sectors_subdirlist)):
        subdir = f"sector_{i}"
        pdb_sectors_filelist = [
            f for f in os.listdir(os.path.join(pdb_sectors_dir, subdir))
            if f.startswith(f"sector_{i}_pdbpos")
        ]
        outdir = os.path.join(new_pdb_sectors_dir, subdir)
        os.makedirs(outdir, exist_ok=True)
        count = 0
        for f in pdb_sectors_filelist:
            key = f.removesuffix(".npy").removeprefix(f"sector_{i}_pdbpos_")
            if key not in mapping_arrays:
                continue
            mapping = mapping_arrays[key]
            fpath = os.path.join(pdb_sectors_dir, subdir, f)
            sector_idxs_old = np.load(fpath)
            sector_scores_old = np.load(fpath.replace("pdbpos", "scores"))
            new_idxs = mapping[sector_idxs_old]
            new_scores = sector_scores_old
            np.save(
                os.path.join(outdir, f"sector_{i}_pdbpos_{key}.npy"),
                new_idxs
            )
            np.save(
                os.path.join(outdir, f"sector_{i}_scores_{key}.npy"),
                new_scores
            )
            count += 1
        print(f"Subdir {subdir}. Saved {count} files")


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
