"""Figure 3 generation script

"""

import os, sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
import tqdm as tqdm

import scipy
import json

from mysca.constants import SECTOR_COLORS

NFIGS = 4


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
    disable_pbar = args.disable_pbar

    fmt = "pdf"
    transparent = True

    # Housekeeping
    os.makedirs(outdir, exist_ok=True)
    pbar = tqdm.tqdm(desc="Plotting", total=NFIGS, disable=disable_pbar)
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

    saveas = "{}{}{}_groups_{}"
    bbox_inches = "tight"
    ICIDXS_AND_GROUP_IDXS = [  # ((ICi, ICj), [group_indices])
        ((0, 1), [0, 1, 2]),
        ((1, 2), [0, 1, 2]),
        ((3, 4), [1, 3, 4]),
    ]
    if saveas:
        for icidxs, group_idxs in ICIDXS_AND_GROUP_IDXS:
            plot_data_2d(
                "ic", icidxs, group_idxs, groups, V_ica_normalized,
                group_colors=SECTOR_COLORS,
                outdir=outdir,
                saveas=saveas,
                format=fmt,
                transparent=transparent,
                bbox_inches=bbox_inches, 
            )
    pbar.update(1)

    saveas = "{}{}{}{}_groups_{}"
    bbox_inches = "tight"
    ICIDXS_AND_GROUP_IDXS = [  # ((ICi, ICj, ICk), [group_indices])
        ((0, 1, 2), [0, 1, 2]),
        ((1, 2, 3), [1, 2, 3]),
        ((4, 5, 6), [4, 5, 6]),
    ]
    if saveas:
        for icidxs, group_idxs in ICIDXS_AND_GROUP_IDXS:
            plot_data_3d(
                "ic", icidxs, group_idxs, groups, V_ica_normalized,
                group_colors=SECTOR_COLORS,
                outdir=outdir,
                saveas=saveas,
                format=fmt,
                transparent=transparent,
                bbox_inches=bbox_inches, 
            )
    pbar.update(1)
    
    pbar.close()
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
    # fig, axes = plt.subplots(nics, 1, figsize=(5, 3 * nics))    
    
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


def plot_data_2d(
        ic_or_ev, axidxs, group_idxs, groups, data, group_colors=None, *, 
        outdir, 
        saveas, 
        format="png",
        transparent=True,
        bbox_inches=None, 
):
    ic_or_ev = ic_or_ev.lower()
    if ic_or_ev.lower() == "ic":
        title = f"Groups in IC space"
    elif ic_or_ev.lower() == "ev":
        title = "Groups in EV space"
    else:
        raise RuntimeError("ic_or_ev should be `ic` or `ev`!")
    if group_idxs == "all":
        group_idxs = list(range(len(groups)))
    axi, axj = axidxs
    if axj >= data.shape[1]:
        return
    fig, ax = plt.subplots(1, 1)
    # ax.axis("equal")
    sc = ax.scatter(
        data[:,axi], data[:,axj],
        c='k', 
        alpha=0.2, 
        edgecolor='k',
    )
    for i, gidx in enumerate(group_idxs):
        if gidx >= len(groups):
            continue
        g = groups[gidx]
        if group_colors is not None:
            group_color = group_colors[gidx]
        else:
            group_color = None
        ax.scatter(
            data[g,axi], data[g,axj],
            alpha=1, 
            color=group_color,
            edgecolor='k',
            label=f"group {gidx}",
        )
    ax.plot(0, 0, "ro")
    rx, ry = ax.get_xlim()[1], ax.get_ylim()[1]
    ax.plot([0, rx], [0, 0], "k-", alpha=0.5)
    ax.plot([0, 0], [0, ry], "k-", alpha=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlabel(f"{ic_or_ev.upper()} {axi}")
    ax.set_ylabel(f"{ic_or_ev.upper()} {axj}")
    ax.set_title(title)
    groupstr = "".join([str(i) for i in group_idxs])
    
    # Save and close
    plt.savefig(
        f"{outdir}/{saveas.format(ic_or_ev, axi, axj, groupstr)}.{format}", 
        format=format, 
        transparent=transparent, bbox_inches=bbox_inches
    )
    plt.close()
    return


def plot_data_3d(
        ic_or_ev, axidxs, group_idxs, groups, data, group_colors=None, *, 
        outdir, 
        saveas, 
        format="png",
        transparent=True,
        bbox_inches=None, 
):
    ic_or_ev = ic_or_ev.lower()
    if ic_or_ev.lower() == "ic":
        title = f"ICA and identified groups"
    elif ic_or_ev.lower() == "ev":
        title = ""
    else:
        raise RuntimeError("ic_or_ev should be `ic` or `ev`!")
    if group_idxs == "all":
        group_idxs = list(range(len(groups)))
    axi, axj, axk = axidxs
    if axk >= data.shape[1]:
        return
    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(111, projection='3d')
    # ax.axis("equal")
    sc = ax.scatter(
        data[:,axi], data[:,axj], data[:,axk], 
        c="k", 
        alpha=0.2, 
        edgecolor='k',
    )
    for i, gidx in enumerate(group_idxs):
        if gidx >= len(groups):
            continue
        g = groups[gidx]
        if group_colors is not None:
            group_color = group_colors[gidx]
        else:
            group_color = None
        ax.scatter(
            data[g,axi], data[g,axj], data[g,axk], 
            alpha=1, 
            color=group_color,
            edgecolor='k',
            label=f"group {gidx}",
        )
    ax.plot(0, 0, "ro")
    rx, ry, rz = ax.get_xlim()[1], ax.get_ylim()[1], ax.get_zlim()[1]
    ax.plot([0, rx], [0, 0], [0, 0], "k-", alpha=0.5)
    ax.plot([0, 0], [0, ry], [0, 0], "k-", alpha=0.5)
    ax.plot([0, 0], [0, 0], [0, rz], "k-", alpha=0.5)
    ax.view_init(elev=30, azim=40)   # elev ~ tilt, azim ~ around z
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlabel(f"{ic_or_ev.upper()} {axi}")
    ax.set_ylabel(f"{ic_or_ev.upper()} {axj}")
    ax.set_zlabel(f"{ic_or_ev.upper()} {axk}")
    ax.set_title(title)
    groupstr = "".join([str(i) for i in group_idxs])
    
    # Save and close
    plt.savefig(
        f"{outdir}/{saveas.format(ic_or_ev, axi, axj, axk, groupstr)}.{format}", 
        format=format, 
        transparent=transparent, bbox_inches=bbox_inches
    )
    plt.close()
    return


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
