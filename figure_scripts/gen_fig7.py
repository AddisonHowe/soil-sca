"""Figure 7 generation script

"""

import os, sys
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.style.use("figure_scripts/styles/fig.mplstyle")
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tqdm as tqdm
import json


variant_groups4_fpath = f"data/K00370/misc/assignments_K00370_v2.tsv"
variant_groups8_fpath = f"data/K00370/misc/assignments_K00370_v3.tsv"
parameter_values_g4_fpath = f"data/K00370/misc/nar_4groups.tsv"
parameter_values_g8_fpath = f"data/K00370/misc/nar_8groups.tsv"

HYDROPHOBIC = {'A', 'V', 'L', 'I', 'M', 'F', 'W', 'P', 'G', 'C'}
HYDROPHILIC = {'S', 'T', 'N', 'Q', 'Y', 'D', 'E', 'K', 'R', 'H'}

HYDROPHOBICITY = {
    'I':  4.5,   'V':  4.2,   'L':  3.8,   'F':  2.8,   'C':  2.5,
    'M':  1.9,   'A':  1.8,   'G': -0.4,   'T': -0.7,   'S': -0.8,
    'W': -0.9,   'Y': -1.3,   'P': -1.6,   'H': -3.2,   'E': -3.5,
    'Q': -3.5,   'D': -3.5,   'N': -3.5,   'K': -3.9,   'R': -4.5,
}


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datdir", type=str, required=True)
    parser.add_argument("-o", "--outdir", type=str, required=True)
    parser.add_argument("-sd", "--sector_dir", type=str, required=True)
    parser.add_argument("-v", "--verbosity", type=int, default=1)
    parser.add_argument("--disable_pbar", action="store_true")
    parser.add_argument("-si", "--sector_idxs", type=int, nargs="+", default=-1)

    return parser.parse_args(args)


def get_printv(verbosity, v_default=1, pbar_default=False):
    def printv(s, v=v_default, pbar=pbar_default):
        logger = tqdm.tqdm.write if pbar else print
        if v <= verbosity:
            logger(s)
    return printv


def main(args):

    # Process command line args
    scadir = args.datdir
    outdir = args.outdir
    sector_dir = args.sector_dir
    verbosity = args.verbosity
    disable_pbar = True
    sector_idxs = args.sector_idxs

    fmt = "pdf"
    transparent = True

    # Housekeeping
    os.makedirs(outdir, exist_ok=True)
    printv = get_printv(verbosity, pbar_default=not disable_pbar)

    printv("RUNNING FIG7")

    # Load data
    msa_fname = "msa.npy"
    retained_sequences_fname = "retained_sequences.npy"
    retained_positions_fname = "retained_positions.npy"
    seqids_fname = "retained_sequence_ids.npy"
    sym2int_fname = "sym2int.json"
    msa = np.load(os.path.join(scadir, msa_fname))
    retained_sequences = np.load(os.path.join(scadir, retained_sequences_fname))
    retained_positions = np.load(os.path.join(scadir, retained_positions_fname))
    retained_seqids = np.load(os.path.join(scadir, seqids_fname))
    with open(os.path.join(scadir, sym2int_fname), "r") as f:
        sym2int = json.load(f)

    printv(f"Loaded MSA of shape {msa.shape}")
    printv(f"Loaded {len(retained_sequences)} sequence indices")
    printv(f"Loaded {len(retained_positions)} position indices")
    printv(f"Loaded {len(retained_seqids)} retained sequence IDs")
    assert msa.shape == (len(retained_sequences), len(retained_positions)), \
        f"Mismatch between MSA shape and number of retained sequences/positions!"
    
    df_variant_group4_assigment = pd.read_csv(
        variant_groups4_fpath, sep=" ", header=None, index_col=0,
        names=["seqid", "group4"],
    )
    df_variant_group8_assigment = pd.read_csv(
        variant_groups8_fpath, sep=" ", header=None, index_col=0,
        names=["seqid", "group8"],
    )
    df_variant_params4_assigment = pd.read_csv(
        parameter_values_g4_fpath, sep=" ", header=None, index_col=0,
        names=["seqid", "ra_inf_g4"],
    )
    df_variant_params8_assigment = pd.read_csv(
        parameter_values_g8_fpath, sep=" ", header=None, index_col=0,
        names=["seqid", "ra_inf_g8"],
    )

    df_variant_groups = pd.concat([
        df_variant_group4_assigment, 
        df_variant_group8_assigment, 
        df_variant_params4_assigment, 
        df_variant_params8_assigment,
    ], axis=1)

    del df_variant_group4_assigment
    del df_variant_group8_assigment
    del df_variant_params4_assigment
    del df_variant_params8_assigment
    printv(f"df_variant_groups contains {len(df_variant_groups)} sequences")

    # Map sequence ID to group assignment and check if the sequence is retained
    id_to_group = {}
    id_to_retainment = {}
    for seqid, row in df_variant_groups.iterrows():
        id_to_group[seqid] = int(row["group4"])
        id_to_retainment[seqid] = seqid in retained_seqids
        if seqid not in retained_seqids:
            print(f"Sequence {seqid} was not retained in MSA preprocessing!!!")

    # List the contents of the directory defining sectors by MSA positions
    sector_files = os.listdir(sector_dir)
    if sector_idxs[0] == -1:
        sector_idxs = [i for i in range(len(sector_files))]

    # Subset the MSA to the sequences of interest
    retained_sequences_screen = np.isin(retained_seqids, df_variant_groups.index.values)
    msa = msa[retained_sequences_screen,:]
    retained_seqids = retained_seqids[retained_sequences_screen]
    # assert msa.shape == (len(df_variant_groups), msa.shape[1]), "MSA shape incorrect!"

    # Reorder the dataframe to match the MSA order
    df_variant_groups = df_variant_groups.loc[retained_seqids].reset_index()
    print(df_variant_groups)
    # assert np.all(df_variant_groups["seqid"].values == retained_seqids)

    # Subset sectors
    sector_results = {}
    binarize = True  # TODO: make clarg
    hydro = False  # TODO: make clarg
    for sector_idx in sector_idxs:
        printv(f"Processing Sector: {sector_idx}")
        sector_arrs = []
        sector_fname = f"group_{sector_idx}_msapos.npy"
        sector_fpath = os.path.join(sector_dir, sector_fname)
        if not os.path.isfile(sector_fpath):
            raise FileNotFoundError(f"File {sector_fpath} does not exist!")
        # Get MSA positions defining the sector
        sector_arrs.append(np.load(sector_fpath))
        sector_positions = np.concatenate(sector_arrs, axis=0)

        # Subset the MSA to the sector(s) of interest
        sector_msa = msa[:, sector_positions]

        # Binarize the MSA if desired
        if binarize:
            printv("Normalizing MSA (binary hydrophobic/philic scale)")
            aa_to_binary = {sym2int[aa]: 0 for aa in HYDROPHOBIC}
            aa_to_binary.update({sym2int[aa]: 1 for aa in HYDROPHILIC})
            aa_to_binary[sym2int['-']] = 0.5
            sector_msa = np.array([
                [aa_to_binary[x] for x in arr] for arr in sector_msa
            ])
        elif hydro:
            printv("Normalizing MSA (hydrophobic/philic scale)")
            aa_to_value = {
                sym2int[aa]: HYDROPHOBICITY[aa] for aa in HYDROPHOBICITY
            }
            aa_to_value[sym2int['-']] = 0.5
            sector_msa = np.array([
                [aa_to_value[x] for x in arr] for arr in sector_msa
            ])
        else:
            printv("No MSA normalization (raw sequence data)")
        
        nuniq = len(np.unique(sector_msa, axis=0))
        printv("{}/{} unique length-{} sequences (after normalization)".format(
            nuniq, sector_msa.shape[0], sector_msa.shape[1]
        ))

        # Apply PCA to the sector
        data_pca, pcs, exp_var_ratio = apply_pca(
            sector_msa, n_components=10
        )
        sector_results[sector_idx] = {}
        sector_results[sector_idx]["msa_pc"] = data_pca
        sector_results[sector_idx]["components"] = pcs
        sector_results[sector_idx]["exp_var_ratio"] = exp_var_ratio


    # Generate plots
    printv("Generating plots...")

    saveas = "expvar_sector_{}"
    bbox_inches = None
    if saveas:
        for sector_idx, sector_res in sector_results.items():
            make_subplot1(
                sector_idx, sector_res, 
                outdir=outdir,
                saveas=saveas,
                format=fmt,
                transparent=transparent,
                bbox_inches=bbox_inches, 
            )

    saveas = "pc1pc2_sector_{}"
    bbox_inches = None
    if saveas:
        for sector_idx, sector_res in sector_results.items():
            make_subplot2(
                sector_idx, sector_res, 
                outdir=outdir,
                saveas=saveas,
                format=fmt,
                transparent=transparent,
                bbox_inches=bbox_inches, 
            )

    saveas = "pc1pc2_bygroup_{}_sector_{}"
    bbox_inches = None
    key = "group4"
    if saveas:
        for sector_idx, sector_res in sector_results.items():
            make_subplot3(
                sector_idx, sector_res, df_variant_groups, key, 
                outdir=outdir,
                saveas=saveas,
                format=fmt,
                transparent=transparent,
                bbox_inches=bbox_inches, 
            )
    
    print("Done!")
    
    return


def make_subplot1(
        sector_idx, sector_res, *,
        outdir, 
        saveas, 
        format="png",
        transparent=True,
        bbox_inches=None, 
):
    data_pca = sector_res["msa_pc"]
    pcs = sector_res["components"]
    exp_var_ratio = sector_res["exp_var_ratio"]

    # Plot PCA explained variance
    fig, ax = plt.subplots(1, 1)
    ax.plot(
        1 + np.arange(len(exp_var_ratio)), np.cumsum(exp_var_ratio),
        ".-"
    )
    ax.set_xlim(0, 1 + len(exp_var_ratio))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(f"PC")
    ax.set_ylabel(f"exp var")
    ax.set_title(f"Cumulative variance explained")
    
    # Save and close
    plt.savefig(
        f"{outdir}/{saveas.format(sector_idx)}.{format}", format=format, 
        transparent=transparent, bbox_inches=bbox_inches
    )
    plt.close()
    return


def make_subplot2(
        sector_idx, sector_res, *,
        outdir, 
        saveas, 
        format="png",
        transparent=True,
        bbox_inches=None, 
):
    data_pca = sector_res["msa_pc"]
    pcs = sector_res["components"]
    exp_var_ratio = sector_res["exp_var_ratio"]

    fig, ax = plt.subplots(1, 1)
    ax.plot(data_pca[:,0], data_pca[:,1], "k.", alpha=0.6)
    ax.set_xlabel(f"PC1")
    ax.set_ylabel(f"PC2")
    ax.set_title(f"PCA Sector {sector_idx}")
    
    # Save and close
    plt.savefig(
        f"{outdir}/{saveas.format(sector_idx)}.{format}", format=format, 
        transparent=transparent, bbox_inches=bbox_inches
    )
    plt.close()
    return


def make_subplot3(
        sector_idx, sector_res, df, dfkey,  *,
        outdir, 
        saveas, 
        format="png",
        transparent=True,
        bbox_inches=None, 
):
    data_pca = sector_res["msa_pc"]
    pcs = sector_res["components"]
    exp_var_ratio = sector_res["exp_var_ratio"]

    fig, ax = plt.subplots(1, 1)
    ax.plot(data_pca[:,0], data_pca[:,1], "k.", alpha=0.6)
    sc = ax.scatter(
        add_noise(data_pca[:,0]), add_noise(data_pca[:,1]),
        c=df[dfkey],
        alpha=0.5,
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig = ax.figure
    cbar = fig.colorbar(sc, cax=cax)
    cbar.ax.set_title("Group")
    ax.set_xlabel(f"PC1")
    ax.set_ylabel(f"PC2")
    ax.set_title(f"PCA Sector {sector_idx}")
    
    # Save and close
    plt.savefig(
        f"{outdir}/{saveas.format(dfkey, sector_idx)}.{format}", format=format, 
        transparent=transparent, bbox_inches=bbox_inches
    )
    plt.close()
    return


def apply_pca(data, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(data)
    pcs = pca.components_
    exp_var_ratio = pca.explained_variance_ratio_
    data_pca = pca.transform(data)
    return data_pca, pcs, exp_var_ratio


def add_noise(data, scale=0.005):
    r = scale * (data.max() - data.min())
    return data + r * np.random.standard_normal(len(data))


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
