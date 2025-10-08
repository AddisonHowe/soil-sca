"""Figure 6 generation script

"""

import os, sys
import argparse
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
import tqdm as tqdm

from mysca.constants import VARIANT_GROUP_COLORS

NFIGS = 1

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

    fmt = "pdf"
    transparent = True

    # Housekeeping
    os.makedirs(outdir, exist_ok=True)
    pbar = tqdm.tqdm(desc="Plotting", total=NFIGS, disable=disable_pbar)
    printv = get_printv(verbosity, pbar_default=not disable_pbar)


    # Load data
    msa = np.load(f"{scadir}/msa.npy")
    sca_matrix = np.load(f"{scadir}/sca_matrix.npy")
    retained_sequences = np.load(f"{scadir}/retained_sequences.npy")
    retained_positions = np.load(f"{scadir}/retained_positions.npy")
    retained_sequence_ids = np.load(f"{scadir}/retained_sequence_ids.npy")
    sequence_weights = np.load(f"{scadir}/sequence_weights.npy")
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
    df_nonsoil = df_full[~df_full["from_soil"]]
    df_soil = df_full[df_full["from_soil"]]

    # Generate plots
    printv("Generating plots...")

    saveas = "jointplot_{}{}_by_{}_ic{}v{}"
    bbox_inches = "tight"
    RANKS = ["kingdom", "phylum", "class"]
    ARGSETS = [
        [None, None, None,],
        ["ra_inf_g4", "cool", "$r_A$",],
        ["ra_inf_g8", "cool", "$r_A$",],
        ["group4", ListedColormap(VARIANT_GROUP_COLORS, N=4), "variant group",],
        ["group8", ListedColormap(VARIANT_GROUP_COLORS, N=8), "variant group",],
    ]
    PAIRS = [(i, i+1) for i in range(Up.shape[1] - 2)]
    PAIRS += [
        (3, 5),
        (4, 6),
    ]
    if saveas:
        for sec_key, cmap, key_label in ARGSETS:
            for rank in RANKS:
                for i, j in PAIRS:    
                    if f"Up{i}" not in df_full.columns or \
                            f"Up{j}" not in df_full.columns:
                        continue
                    make_subplot_jointplots(
                        df_soil, df_nonsoil, rank, i, j, sec_key, cmap, 
                        key_label,
                        outdir=outdir,
                        saveas=saveas,
                        format=fmt,
                        transparent=transparent,
                        bbox_inches=bbox_inches, 
                    )
    pbar.update(1)
    
    pbar.close()
    print("Done!")


def make_subplot_jointplots(
        df_soil, df_nonsoil, rank, i, j, sec_key, cmap, key_label, *,
        outdir, 
        saveas, 
        format="png",
        transparent=True,
        bbox_inches=None, 
):
    if isinstance(cmap, ListedColormap):
        norm = BoundaryNorm(0.5 + np.arange(1 + len(cmap.colors)), cmap.N)
    else:
        norm = None
    include_soil = sec_key is not None
    g = scatter_by_with_margins(
        df_nonsoil, i, j, rank, 
        k=20, 
        palette="tab20",
        size=6,
        alpha=0.8,
        legend=True,
    )
    fig = g.ax_joint.get_figure()
    if include_soil:
        sc = g.ax_joint.scatter(
            df_soil[f"Up{i}"].values, df_soil[f"Up{j}"].values, 
            c=df_soil[sec_key].values,
            alpha=1.0,
            cmap=cmap,
            norm=norm,
            s=8,
        )                
        # Position colorbar to the right of the legend
        renderer = g.ax_joint.get_figure().canvas.get_renderer()
        legend = g.ax_marg_y.get_legend()
        bbox = legend.get_window_extent(renderer=renderer)
        bbox_fig = bbox.transformed(fig.transFigure.inverted())
        pad = 0.05
        width = 0.02
        left = bbox_fig.x1 + pad
        bottom = g.ax_joint.get_position().y0
        height = g.ax_joint.get_position().height
        # Add the colorbar
        cax = fig.add_axes([left, bottom, width, height])
        cbar = plt.colorbar(sc, cax=cax)
        
        cbar.ax.set_title(key_label, fontsize=10)
        if isinstance(cmap, ListedColormap):
            cbar.ax.set_yticks(
                1 + np.arange(cmap.N),
                [str(i+1) for i in range(cmap.N)]
            )

    # Save and close
    saveas = saveas.format(
        "withsoil" if include_soil else "nonsoil", 
        "_" + sec_key if include_soil else "", 
        rank, i, j
    )
    plt.savefig(
        f"{outdir}/{saveas}.{format}", format=format, 
        transparent=transparent, bbox_inches=bbox_inches
    )
    plt.close()
    return


def scatter_by_with_margins(
        df, idx0, idx1, color_by, 
        k=None, 
        size=20, 
        alpha=1, 
        palette="viridis",
        color_nan="r",
        color_other="lightgrey",
        data_key="Up{}",
        axis_label_template="seq map of IC {}",
        legend=True,
):
    xkey = data_key.format(idx0)
    ykey = data_key.format(idx1)
    xlabel = axis_label_template.format(idx0)
    ylabel = axis_label_template.format(idx1)
    
    cat_counts = df[color_by].value_counts()
    ncats = len(cat_counts)
    if k is None or k <= 0:
        k = ncats
    
    top_k = cat_counts.nlargest(k).index
    col_group = df[color_by].where(df[color_by].isin(top_k), 'Other')
    col_group[pd.isna(df[color_by])] = "NA"
    palette = sns.color_palette(palette, n_colors=k)
    palette = dict(zip(top_k, palette))
    palette["Other"] = color_other
    palette["NA"] = color_nan
    hue_order = top_k.append(pd.Index(["Other", "NA"]))

    g = sns.JointGrid(
        data=df, x=xkey, y=ykey, 
        hue=col_group,
        hue_order=hue_order,
        palette=palette,
        marginal_ticks=True,
    )
    g.plot_joint(
        sns.scatterplot,
        s=size, 
        alpha=alpha, 
        edgecolor="none",
        legend=legend,
    )
    g.plot_marginals(
        sns.histplot,
        multiple="stack"
    )
    g.ax_joint.set_xlabel(xlabel)
    g.ax_joint.set_ylabel(ylabel)

    if legend:
        handles, labels = g.ax_joint.get_legend_handles_labels()
        labels = [
            l if l not in cat_counts else f"{l} ({cat_counts[l]})" for l in labels
        ]
        lg = g.ax_marg_y.legend(
            handles=handles, labels=labels, title=color_by.title(),
            bbox_to_anchor=(1.05, 1.0), loc='upper left',
            frameon=True,
            fontsize=8,
        )
        for handle in lg.legend_handles:
            handle.set_markersize(6)
        g.ax_joint.get_legend().remove()

    return g


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
