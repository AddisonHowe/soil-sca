"""Figure 4 generation script

"""

import os, sys
import argparse
import numpy as np
import scipy
import matplotlib.pyplot as plt
plt.style.use("figure_scripts/styles/fig.mplstyle")
import tqdm as tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
import tqdm as tqdm

import scipy
import json

from mysca.constants import SECTOR_COLORS


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datdir", type=str, required=True)
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
    outdir = args.outdir
    verbosity = args.verbosity
    disable_pbar = True

    fmt = "pdf"
    transparent = True

    # Housekeeping
    os.makedirs(outdir, exist_ok=True)
    printv = get_printv(verbosity, pbar_default=not disable_pbar)

    # Load data
    V_ica_normalized = np.load(f"{scadir}/v_ica_normalized.npy")
    sca_matrix_subset = np.load(f"{scadir}/sca_matrix_sector_subset.npy")
    msapos_to_groupidx = np.load(f"{scadir}/msapos_to_groupidx.npy")
    with open(f"{scadir}/t_dists_info.json", "r") as f:
        t_dists_info = json.load(f)
    
    ngroups = len(np.unique(msapos_to_groupidx[1,:]))
    groups = []
    for gidx in range(ngroups):
        idxs = msapos_to_groupidx[1,:] == gidx
        groups.append(msapos_to_groupidx[0,idxs])    

    # Generate plots
    printv("RUNNING FIG4")
    printv("Generating plots...")

    saveas = "t_dist_ic{}"
    bbox_inches = None
    idxs = [0, 1, 2, 3]
    if saveas:
        make_subplot_t_dists(
            V_ica_normalized, t_dists_info, idxs=idxs, 
            outdir=outdir,
            saveas=saveas,
            format=fmt,
            transparent=transparent,
            bbox_inches=bbox_inches, 
        )

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
    
    print("Done!")


def make_subplot_t_dists(
        V_ica_normalized, t_dists_info, idxs=None, *, 
        outdir, 
        saveas, 
        format="png",
        transparent=True,
        bbox_inches=None, 
):
    v = V_ica_normalized
    if idxs is None:
        idxs = np.arange(v.shape[1])

    npos, nics = v.shape    
    for i in idxs:
        fig, ax = plt.subplots(1, 1)
        vi = v[:,i]
        tinfo = t_dists_info[i]
        ax.hist(
            vi, bins=20, density=True, alpha=0.5, color="skyblue",
        )
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        x = np.linspace(*xlims, 100)
        y = scipy.stats.t.pdf(
            x, df=tinfo["df"], loc=tinfo["loc"], scale=tinfo["scale"], 
        )
        ax.vlines(tinfo["cutoff"], *ylims, colors="k", linestyles="--")
        ax.plot(x, y)
        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)
        ax.set_xlabel(f"IC {i}")
        ax.set_ylabel(f"p")
        ax.set_title(f"IC {i} Student's $t$")

        # Save and close
        plt.savefig(
            f"{outdir}/{saveas.format(i)}.{format}", format=format, 
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
