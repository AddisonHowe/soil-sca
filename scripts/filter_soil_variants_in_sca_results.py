"""Filter the soil variants from the SCA results

"""

import os, sys
import argparse
import numpy as np
import pandas as pd
import json


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datdir", type=str, required=True)
    parser.add_argument("-o", "--outdir", type=str, required=True)
    parser.add_argument("-sf", "--seqids_fpath", type=str, required=True)
    parser.add_argument("-si", "--sector_idxs", type=int, nargs="+", default=-1)

    return parser.parse_args(args)


def main(args):
    datdir = args.datdir
    outdir = args.outdir
    seqids_fpath = args.seqids_fpath
    sector_idxs = args.sector_idxs

    # Housekeeping
    scadir = os.path.join(datdir, "sca_results")
    sector_dir = os.path.join(datdir, "groups")
    os.makedirs(outdir, exist_ok=True)
    if sector_idxs == -1:
        sfiles = os.listdir(sector_dir)
        sector_idxs = [i for i in range(len(sfiles))]

    # Load the full msa and all sequence IDs
    msa_full = np.load(os.path.join(scadir, "msa.npy"))
    print(msa_full.shape)
    seqids_full = np.load(os.path.join(scadir, "retained_sequence_ids.npy"))
    print(seqids_full.shape)
    retained_sequences = np.load(os.path.join(scadir, "retained_sequences.npy"))
    print(retained_sequences.shape)
    retained_positions = np.load(os.path.join(scadir, "retained_positions.npy"))
    print(retained_positions.shape)
    
    # Load soil sequence IDs
    seqids_soil = np.genfromtxt(seqids_fpath, dtype=str)

    soil_screen = np.isin(seqids_full, seqids_soil)
    msa_soil = msa_full[soil_screen]
    print(msa_soil.shape)

    # Save the MSA subsets
    sectors = []
    for sector_idx in sector_idxs:
        fname = f"group_{sector_idx}_msapos.npy"
        sector_fpath = os.path.join(sector_dir, fname)
        if not os.path.isfile(sector_fpath):
            print(f"File not found: {sector_fpath}!")
            continue
        sector_positions = np.load(sector_fpath)
        msa_sector = msa_soil[:,sector_positions]
        print(msa_sector.shape)
        sectors.append(msa_sector)

        # Save output
        outfname = f"msa_soilsubset_sector_{sector_idx}.npy"
        outfpath = os.path.join(outdir, outfname)
        np.save(outfpath, msa_sector)
    
    # Save the sequence IDs
    seqids_out = seqids_full[soil_screen]
    np.savetxt(os.path.join(outdir, "seqids.txt"), seqids_out, fmt="%s")

    


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
