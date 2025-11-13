"""Figure 5 generation script

"""

import os, sys
import argparse
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
plt.style.use("figure_scripts/styles/fig.mplstyle")
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
import tqdm as tqdm

from mysca.constants import VARIANT_GROUP_COLORS, SECTOR_COLORS

NFIGS = 2

variant_groups4_fpath = f"data/K00370/misc/assignments_K00370_v2.tsv"
variant_groups8_fpath = f"data/K00370/misc/assignments_K00370_v3.tsv"
parameter_values_g4_fpath = f"data/K00370/misc/nar_4groups.tsv"
parameter_values_g8_fpath = f"data/K00370/misc/nar_8groups.tsv"

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datdir", type=str, required=True)
    parser.add_argument("-sm", "--seq_metadata_fpath", type=str, required=True)
    parser.add_argument("-tm", "--tax_metadata_fpath", type=str, required=True)
    parser.add_argument("-o", "--outdir", type=str, required=True)
    parser.add_argument("-v", "--verbosity", type=int, default=1)
    parser.add_argument("--disable_pbar", action="store_true")
    parser.add_argument("-p", "--pairs", type=int, nargs="*", default=None)
    parser.add_argument("-xl", "--xlims", type=float, nargs="*", default=None)
    parser.add_argument("-yl", "--ylims", type=float, nargs="*", default=None)
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
    seq_metadata_fpath = args.seq_metadata_fpath
    tax_metadata_fpath = args.tax_metadata_fpath
    outdir = args.outdir
    verbosity = args.verbosity
    disable_pbar = args.disable_pbar
    pairs = args.pairs
    xlims = args.xlims
    ylims = args.ylims

    fmt = "pdf"
    transparent = True

    # Housekeeping
    if pairs is None:
        pairs = np.array([(i, i+1) for i in range(Up.shape[1] - 2)])
    elif len(pairs) == 2:
        pairs = np.array(pairs)[None,:]
        assert pairs.shape == (1,2), f"{pairs.shape}"
    else:
        pairs = np.array(pairs).reshape([-1, 2])
    
    if xlims is None:
        xlims = np.zeros(pairs.shape, dtype=int)
    elif len(xlims) == 2:
        xlims = np.array(xlims)[None,:]
        assert xlims.shape == (1,2)
    else:
        xlims = np.array(xlims).reshape([-1, 2])
    
    if ylims is None:
        ylims = np.zeros(pairs.shape, dtype=int)
    elif len(ylims) == 2:
        ylims = np.array(ylims)[None,:]
        assert ylims.shape == (1,2)
    else:
        ylims = np.array(ylims).reshape([-1, 2])
    
    os.makedirs(outdir, exist_ok=True)
    pbar = tqdm.tqdm(desc="Plotting", total=NFIGS, disable=disable_pbar)
    printv = get_printv(verbosity, pbar_default=not disable_pbar)


    # Load data
    # msa = np.load(f"{scadir}/msa.npy")
    # sca_matrix = np.load(f"{scadir}/sca_matrix.npy")
    # retained_sequences = np.load(f"{scadir}/retained_sequences.npy")
    # retained_positions = np.load(f"{scadir}/retained_positions.npy")
    retained_sequence_ids = np.load(f"{scadir}/retained_sequence_ids.npy")
    # sequence_weights = np.load(f"{scadir}/sequence_weights.npy")
    fia = np.load(f"{scadir}/fia.npy")
    phi_ia = np.load(f"{scadir}/phi_ia.npy")
    Xmsa = scipy.sparse.load_npz(f"{scadir}/Xsp.npz").toarray()
    W_ica = np.load(f"{scadir}/w_ica.npy")
    evals_sca = np.load(f"{scadir}/significant_evals_sca.npy")
    evecs_sca = np.load(f"{scadir}/significant_evecs_sca.npy")

    # Compute projection matrix
    naas = fia.shape[1]
    Xmsa = Xmsa.reshape([Xmsa.shape[0], -1, naas])
    Pia = phi_ia * fia / np.sqrt(np.sum(np.square(phi_ia * fia)))
    Xsi = np.sum(Pia[None,:,:] * Xmsa, axis=-1)
    Utilde = Xsi @ evecs_sca / np.sqrt(evals_sca)
    Up = Utilde @ W_ica.T

    d = {
        "seqid": retained_sequence_ids,
        "from_soil": [
            s.startswith("Soil") or s.startswith("T0.") for s in retained_sequence_ids
        ]
    }
    for i in range(Up.shape[1]):
        d[f"Up{i}"] = Up[:,i]
    df_seqs = pd.DataFrame(d)
    printv(f"df_seqs contains {df_seqs["from_soil"].sum()} sequences from soil data")

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

    # Sequence metadata information
    df_seq_metadata = pd.read_csv(
        seq_metadata_fpath, sep="\t", header=None, 
        names=["id", "status", "prot_name", "taxidstr"]
    )

    df_seq_metadata["taxid"] = df_seq_metadata["taxidstr"].str.removeprefix("taxID:").astype(int)
    df_seq_metadata["prot_name"] = df_seq_metadata["prot_name"].str.lower()
    df_seq_metadata["seqid"] = df_seq_metadata["id"].map({
        s.split("|")[0]: s for s in retained_sequence_ids
    })

    taxid_counts = df_seq_metadata["taxid"].value_counts()

    # Taxa metadata information
    df_taxa_metadata = pd.read_csv(
        tax_metadata_fpath, sep="\t", 
    )
    df_taxa_metadata = df_taxa_metadata.drop_duplicates()

    # Merge all sequence and taxa metadata
    df_meta = df_seq_metadata.merge(df_taxa_metadata, on="taxid", how="left")

    # Now append positional information U
    df_full = df_seqs.merge(df_meta, on="seqid", how="left").merge(
        df_variant_groups, on="seqid", how="left"
    )

    df_full.to_csv(f"{outdir}/df_full.csv", sep="\t")

    sca_matrix_subset = np.load(f"{scadir}/sca_matrix_sector_subset.npy")
    msapos_to_groupidx = np.load(f"{scadir}/msapos_to_groupidx.npy")
    ngroups = len(np.unique(msapos_to_groupidx[1,:]))
    groups = []
    for gidx in range(ngroups):
        idxs = msapos_to_groupidx[1,:] == gidx
        groups.append(msapos_to_groupidx[0,idxs])

    # Generate plots
    printv("Generating plots...")

    saveas = "ic{}v{}_{}"
    bbox_inches = None
    layout="constrained"
    ARGSETS = [
        # ["ra_inf_g4", "cool", "$r_A^{(4)}$",],
        # ["ra_inf_g8", "cool", "$r_A^{(8)}$",],
        ["group4", ListedColormap(VARIANT_GROUP_COLORS, N=4), "variant group",],
        # ["group8", ListedColormap(VARIANT_GROUP_COLORS, N=8), "variant group",],
    ]
    FIGSIZE = (6.75, 4.5)
    if saveas:
        for sec_key, cmap, key_label in ARGSETS:
            for counter, (i, j) in enumerate(pairs):
                xlim = xlims[counter, :]
                ylim = ylims[counter, :]
                if np.all(xlim == 0):
                    xlim = None
                if np.all(ylim == 0):
                    ylim = None
                if f"Up{i}" not in df_full.columns or f"Up{j}" not in df_full.columns:
                    continue
                make_subplot_seqmap_scatterplots(
                    df_full, i, j, sec_key, cmap, key_label, 
                    outdir=outdir,
                    saveas=saveas,
                    format=fmt,
                    transparent=transparent,
                    bbox_inches=bbox_inches,
                    layout=layout,
                    figsize=FIGSIZE,
                    xlim=xlim,
                    ylim=ylim,
                )
    pbar.update(1)

    saveas = "sca_matrix_sector_subsets"
    bbox_inches = "tight"
    sector_color_set = SECTOR_COLORS
    if saveas:
        make_subplot_sca_matrix_sector_subsets(
            sca_matrix_subset, groups, sector_color_set, 
            outdir=outdir,
            saveas=saveas,
            format=fmt,
            transparent=transparent,
            bbox_inches=bbox_inches, 
        )
    pbar.update(1)
    
    pbar.close()
    print("Done!")


def make_subplot_seqmap_scatterplots(
        df_full, i, j, sec_key, cmap, key_label, *,
        outdir, 
        saveas, 
        format="png",
        transparent=True,
        bbox_inches=None,
        layout=None,
        figsize=None, 
        xlim=None,
        ylim=None,
):
    fig, ax = plt.subplots(1, 1, figsize=figsize, layout=layout)
    ax.scatter(
        df_full[f"Up{i}"].values, df_full[f"Up{j}"].values,
        s=3,
        c="k",
        alpha=0.1
    )
    if isinstance(cmap, ListedColormap):
        norm = BoundaryNorm(0.5 + np.arange(1 + len(cmap.colors)), cmap.N)
    else:
        norm = None

    sc = ax.scatter(
        df_full[df_full["from_soil"]][f"Up{i}"].values, 
        df_full[df_full["from_soil"]][f"Up{j}"].values,
        s=10,
        c=df_full[df_full["from_soil"]][sec_key],
        cmap=cmap,
        norm=norm,
        alpha=0.8,
    )

    cbar = fig.colorbar(sc, ax=ax)
    cbar.ax.set_ylabel(key_label)
    if isinstance(cmap, ListedColormap):
        cbar.ax.set_yticks(
            1 + np.arange(cmap.N),
            [str(i+1) for i in range(cmap.N)]
        )

    ax.set_xlabel(
        f"seq score of IC {i} $(\\tilde{{U}}_{i}^p$)", 
        color="black",
        bbox=dict(
            facecolor=SECTOR_COLORS[i],  # highlight color
            alpha=0.5,
            edgecolor="none",  # no border
            boxstyle="round,pad=0.3",  # rounded corners
        ),
    )
    ax.set_ylabel(
        f"seq score of IC {j} $(\\tilde{{U}}_{j}^p)$", 
        color="black",
        bbox=dict(
            facecolor=SECTOR_COLORS[j],  # highlight color
            alpha=0.5,
            edgecolor="none",  # no border
            boxstyle="round,pad=0.3",  # rounded corners
        ),
    )
    ax.set_title(f"")   

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    # Save and close
    plt.savefig(
        f"{outdir}/{saveas.format(i, j, sec_key)}.{format}", format=format, 
        transparent=transparent, bbox_inches=bbox_inches
    )
    plt.close()
    return


def make_subplot_sca_matrix_sector_subsets(
        sca_matrix_subset, groups, sector_color_set=None, *, 
        outdir, 
        saveas, 
        format="png",
        transparent=True,
        bbox_inches=None, 
):
    fig, ax = plt.subplots(1, 1)
    sc = ax.imshow(
        sca_matrix_subset, 
        cmap="Blues", 
        origin="lower",
        interpolation="none",
        vmax=None,
    )
    fig.colorbar(sc, label="Covariation")
    ax.set_xlabel("(Important) Position i")
    ax.set_ylabel("(Important) Position j")
    ax.set_title("SCA Matrix (Groups)")

    if sector_color_set:
        group_colors = np.concatenate([
            len(g) * [colors.to_rgb(sector_color_set[i])] for i, g in enumerate(groups)
        ], axis=0)
        divider = make_axes_locatable(ax)
        # Top rug
        ax_top = divider.append_axes("top", size="2%", pad=0.0, sharex=ax)
        ax_top.imshow(
            group_colors[None,:,:], 
            aspect="auto", 
            extent=(0, len(group_colors), 0, 1)
        )
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        ax_top.set_title(ax.get_title())
        ax.set_title("")
        # Right rug
        ax_right = divider.append_axes("right", size="2%", pad=0.0, sharey=ax)
        ax_right.imshow(
            np.flip(group_colors, axis=0)[:,None,:], 
            aspect="auto", 
            extent=(0, 1, 0, len(group_colors))
        )
        ax_right.set_xticks([])        
        ax_right.set_yticks([])
    
    # Save and close
    plt.savefig(
        f"{outdir}/{saveas}.{format}", format=format, 
        transparent=transparent, bbox_inches=bbox_inches
    )
    plt.close()


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
