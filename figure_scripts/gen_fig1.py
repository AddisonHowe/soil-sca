"""Figure 1 generation script

"""

import os, sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("figure_scripts/styles/fig.mplstyle")
import tqdm as tqdm

import scipy
from scipy.spatial.distance import pdist, squareform
import scipy.cluster.hierarchy as sch

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
    printv("Loading data...")
    fia = np.load(f"{scadir}/fia.npy")
    msa = np.load(f"{scadir}/msa.npy")
    fi0_pretrunc = np.load(f"{scadir}/fi0_pretrunc.npy")
    position_gap_thresh = np.genfromtxt(f"{scadir}/position_gap_thresh.txt")
    Xmsa = scipy.sparse.load_npz(f"{scadir}/Xsp.npz").toarray()
    naas = fia.shape[1]
    Xmsa = Xmsa.reshape([Xmsa.shape[0], -1, naas])

    # Generate plots
    printv("Generating plots...")

    saveas = "gap_freq_by_position"
    bbox_inches = None
    if saveas:
        make_gap_freq_plot(
            msa, fi0_pretrunc, position_gap_thresh,
            outdir=outdir,
            saveas=saveas,
            format=fmt,
            transparent=transparent,
            bbox_inches=bbox_inches, 
        )
    pbar.update(1)

    saveas = "sequence_similarity"
    bbox_inches = "tight"
    if saveas:
        make_seq_similarity_plot(
            Xmsa,
            outdir=outdir,
            saveas=saveas,
            format=fmt,
            transparent=transparent,
            bbox_inches=bbox_inches, 
        )
    pbar.update(1)

    saveas = "reference_similarity"
    bbox_inches = None
    if saveas:
        make_ref_similarity_plot(
            #data...
            outdir=outdir,
            saveas=saveas,
            format=fmt,
            transparent=transparent,
            bbox_inches=bbox_inches, 
        )
    pbar.update(1)

    saveas = "taxonomic_makeup"
    bbox_inches = None
    if saveas:
        make_taxonomic_makeup_plot(
            #data...
            outdir=outdir,
            saveas=saveas,
            format=fmt,
            transparent=transparent,
            bbox_inches=bbox_inches, 
        )
    pbar.update(1)
    
    pbar.close()
    print("Done!")


def make_gap_freq_plot(
        msa, fi0_pretrunc, position_gap_thresh, *,
        outdir, 
        saveas, 
        format="png",
        transparent=True,
        bbox_inches=None, 
):
    fig, ax = plt.subplots(1, 1)
    ax.plot(fi0_pretrunc, ".")
    ax.hlines(
        position_gap_thresh, *ax.get_xlim(), 
        linestyle='--', 
        color="r", 
        label="cutoff"
    )
    ax.legend()
    ax.set_xlim(0, 10 + msa.shape[1])
    ax.set_xlabel(f"position")
    ax.set_ylabel(f"gap frequency")
    ax.set_title(f"Gap frequency by position")
    plt.savefig(
        f"{outdir}/{saveas}.{format}", format=format, 
        transparent=transparent, bbox_inches=bbox_inches
    )
    plt.close()
    return


def make_seq_similarity_plot(
        xmsa, *, 
        outdir, 
        saveas, 
        format="png",
        transparent=True,
        bbox_inches=None, 
):
    npos = xmsa.shape[1]
    xmsa = xmsa.argmax(axis=-1)  # conversion
    distances = pdist(xmsa, metric="hamming")
    similarities = 1 - distances
    similarity_matrix = 1 - squareform(distances)
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8,5))

    Z = sch.linkage(distances, method="complete", metric="hamming")
    dendro = sch.dendrogram(Z, no_plot=True)
    idxs = dendro["leaves"]
    
    ax1.hist(similarities, int(round(npos/2)))
    ax1.set_xlabel("Pairwise sequence identities")
    ax1.set_ylabel("Count")

    sc = ax2.imshow(
        similarity_matrix[np.ix_(idxs, idxs)],
        vmin=0, vmax=1,
        interpolation="none",
    )
    plt.colorbar(sc)
    plt.savefig(
        f"{outdir}/{saveas}.{format}", format=format,
        transparent=transparent, bbox_inches=bbox_inches
    )
    plt.close()


def make_ref_similarity_plot(
        *, 
        outdir, 
        saveas, 
        format="png",
        transparent=True,
        bbox_inches=None, 
):
    return


def make_taxonomic_makeup_plot(
        *, 
        outdir, 
        saveas, 
        format="png",
        transparent=True,
        bbox_inches=None, 
):
    return


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
