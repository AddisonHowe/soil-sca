"""Figure 2 generation script

"""

import os, sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tqdm as tqdm

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
    retained_positions = np.load(f"{scadir}/retained_positions.npy")
    Di = np.load(f"{scadir}/conservation.npy")
    Cij = np.load(f"{scadir}/sca_matrix.npy")
    topk_conserved_msa_pos = np.load(f"{scadir}/topk_conserved_msa_pos.npy")
    top_conserved_Di = np.load(f"{scadir}/top_conserved_Di.npy")
    evals_sca = np.load(f"{scadir}/all_evals_sca.npy")
    evals_shuff = np.load(f"{scadir}/evals_shuff.npy")
    num_pos_orig = np.genfromtxt(f"{scadir}/npos_original.txt", dtype=int)
    eigenvalue_cutoff = np.genfromtxt(f"{scadir}/eigenvalue_cutoff.txt")

    # Generate plots
    printv("Generating plots...")

    saveas = "positional_conservation"
    bbox_inches = None
    if saveas:
        make_subplot_positional_conservation(
            retained_positions, Di, 
            topk_conserved_msa_pos, top_conserved_Di, num_pos_orig, 
            outdir=outdir,
            saveas=saveas,
            format=fmt,
            transparent=transparent,
            bbox_inches=bbox_inches, 
        )
    pbar.update(1)

    saveas = "sca_matrix"
    bbox_inches = None
    if saveas:
        make_subplot_sca_matrix(
            Cij, 
            outdir=outdir,
            saveas=saveas,
            format=fmt,
            transparent=transparent,
            bbox_inches=bbox_inches, 
        )
    pbar.update(1)

    saveas = "sca_shuffling_distribution"
    bbox_inches = None
    if saveas:
        make_subplot_shuffling_distribution(
            evals_sca, evals_shuff, 
            outdir=outdir,
            saveas=saveas,
            format=fmt,
            transparent=transparent,
            bbox_inches=bbox_inches, 
        )
    pbar.update(1)

    saveas = "sca_matrix_spectrum_vs_null"
    bbox_inches = None
    if saveas:
        make_subplot_sca_matrix_spectrum_vs_null(
            evals_sca, evals_shuff, eigenvalue_cutoff, 
            outdir=outdir,
            saveas=saveas,
            format=fmt,
            transparent=transparent,
            bbox_inches=bbox_inches, 
        )
    pbar.update(1)
    

    pbar.close()
    print("Done!")


def make_subplot_positional_conservation(
        retained_positions, Di, 
        topk_conserved_msa_pos, top_conserved_Di, num_pos_orig, *,
        outdir, 
        saveas, 
        format="png",
        transparent=True,
        bbox_inches=None, 
):
    fig, ax = plt.subplots(1, 1, figsize=(10,4))
    ax.plot(
        retained_positions, Di, "o",
        color="Blue",
        alpha=0.2
    )
    ax.plot(
        topk_conserved_msa_pos, top_conserved_Di, "o",
        color="Green",
        alpha=0.5
    )
    ax.set_xlim(0, num_pos_orig)
    ax.set_xlabel(f"Position")
    ax.set_ylabel("Relative Entropy (KL Divergence, $D_i$)")
    ax.set_title(f"Position-wise Conservation")
    
    # Save and close
    plt.savefig(
        f"{outdir}/{saveas}.{format}", format=format, 
        transparent=transparent, bbox_inches=bbox_inches
    )
    plt.close()
    return


def make_subplot_sca_matrix(
        Cij, *,
        outdir, 
        saveas, 
        format="png",
        transparent=True,
        bbox_inches=None, 
):
    fig, ax = plt.subplots(1, 1)
    sc = ax.imshow(
        Cij, 
        cmap="Blues", 
        origin="lower",
        interpolation="none",
        vmax=None,
    )
    fig.colorbar(sc, label="Covariation")
    ax.set_xlabel("(Retained) Position i")
    ax.set_ylabel("(Retained) Position j")
    ax.set_title("SCA Matrix")

    # Save and close
    plt.savefig(
        f"{outdir}/{saveas}.{format}", format=format, 
        transparent=transparent, bbox_inches=bbox_inches
    )
    plt.close()
    return


def make_subplot_shuffling_distribution(
        evals_sca, evals_shuff, *, 
        outdir, 
        saveas, 
        format="png",
        transparent=True,
        bbox_inches=None, 
):
    fig, ax = plt.subplots(1, 1)
    for e in evals_shuff:
        ax.plot(
            1 + np.arange(len(e)), e, ".",
            markersize=3
        )
    ax.plot(
        1 + np.arange(len(evals_sca)), evals_sca,
        "k.",
        markersize=2,
        label="data",
    )
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlabel(f"$\\lambda$ index")
    ax.set_ylabel(f"$\\lambda$")
    ax.set_title(f"$\\tilde{{C}}_{{ij}}$ Spectrum (data vs null)")

    # Save and close
    plt.savefig(
        f"{outdir}/{saveas}.{format}", format=format, 
        transparent=transparent, bbox_inches=bbox_inches
    )
    plt.close()
    return

def make_subplot_sca_matrix_spectrum_vs_null(
        evals_sca, evals_shuff, cutoff, *, 
        outdir, 
        saveas, 
        format="png",
        transparent=True,
        bbox_inches=None, 
):
    fig, ax = plt.subplots(1, 1)
    # Histogram of data eigenvalues
    counts, bins, patches = ax.hist(
        evals_sca, bins=100, color="black", alpha=0.8, log=True, label="Data"
    )
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    h, bin_edges = np.histogram(evals_shuff.flatten(), bins=bins)
    ax.axvline(cutoff, 0, 1, linestyle="--", color="grey")
    ax.plot(
        bin_centers, h / evals_shuff.shape[0], 
        color="red", 
        lw=1.5, 
        label="Null"
    )
    ax.legend()
    ax.set_xlabel(f"$\\lambda$")
    ax.set_ylabel(f"Count")
    ax.set_title(f"Spectral decomposition")
    
    # Save and close
    plt.savefig(
        f"{outdir}/{saveas}.{format}", format=format, 
        transparent=transparent, bbox_inches=bbox_inches
    )
    plt.close()
    return


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
