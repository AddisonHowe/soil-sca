"""SCA pipeline

See references:
    [1] SI to Rivoire et al., 2016

"""

import argparse
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import tqdm as tqdm
import json

import scipy.cluster.hierarchy as sch
from scipy import sparse
from scipy.spatial.distance import pdist

from mysca.io import load_msa
from mysca.preprocess import preprocess_msa
from mysca.preprocess import compute_background_freqs
from mysca.core import run_sca, run_ica
from mysca.helpers import get_top_k_conserved_retained_positions
from mysca.helpers import get_rawseq_positions_in_groups
from mysca.helpers import get_group_rawseq_positions_by_entry
from mysca.helpers import get_rawseq_indices_of_msa

DEFAULT_BACKGROUND_FREQ = {
        'A': 0.078, 'C': 0.020, 'D': 0.053, 'E': 0.063,
        'F': 0.039, 'G': 0.072, 'H': 0.023, 'I': 0.053,
        'K': 0.059, 'L': 0.091, 'M': 0.022, 'N': 0.043,
        'P': 0.052, 'Q': 0.042, 'R': 0.051, 'S': 0.071,
        'T': 0.058, 'V': 0.066, 'W': 0.014, 'Y': 0.033,
    }


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-msa", "--msa_fpath", type=str, required=True,
                        help="Filepath of input MSA in fasta format.")
    parser.add_argument("-o", "--outdir", type=str, required=True, 
                        help="Output directory.")
    parser.add_argument("--pbar", action="store_true")
    parser.add_argument("-v", "--verbosity", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)

    sca_params = parser.add_argument_group("SCA parameters")
    sca_params.add_argument("--gap_truncation_thresh", type=float, default=0.4,
                            help="SCA parameter gap_truncation_thresh")
    sca_params.add_argument("--sequence_gap_thresh", type=float, default=0.2,
                            help="SCA parameter sequence_gap_thresh γ_{seq}")
    sca_params.add_argument("--reference", type=str, default=None, 
                            help="SCA optional reference entry in MSA")
    sca_params.add_argument("--reference_similarity_thresh", type=float, default=0.2,
                            help="SCA parameter reference_similarity_thresh Δ")
    sca_params.add_argument("--sequence_similarity_thresh", type=float, default=0.8,
                            help="SCA parameter sequence_similarity_thresh δ")
    sca_params.add_argument("--position_gap_thresh", type=float, default=0.2,
                            help="SCA parameter position_gap_thresh γ_{pos}")
    sca_params.add_argument("--regularization", type=float, default=0.03,
                            help="SCA regularization parameter λ")
    sca_params.add_argument("--background", type=str, default=None,
                            help="Path to file describing background frequency." \
                            " If None, use default.")
    sca_params.add_argument("-nc", "--n_top_conserved", type=int, required=True, 
                            help="Number of top conserved residues to consider.")
    sca_params.add_argument("-nb", "--n_boot", type=int, default=10, 
                            help="Number of bootstraps to use for eval threshold.")
    sca_params.add_argument("-k", "--kstar", type=int, default=0, 
                            help="Value of k_start to override bootstrap estimate.")
    sca_params.add_argument("-p", "--pstar", type=int, default=95, 
                            help="Percentile defining IC groups.")

    return parser.parse_args(args)


def main(args):

    # Process command line args
    MSA_FPATH = args.msa_fpath
    reference_id = args.reference
    OUTDIR = args.outdir
    verbosity = args.verbosity
    n_top_conserved = args.n_top_conserved
    N_BOOT = args.n_boot
    PBAR = args.pbar
    SEED = args.seed

    gap_truncation_thresh = args.gap_truncation_thresh
    sequence_gap_thresh = args.sequence_gap_thresh
    reference_id = args.reference
    reference_similarity_thresh = args.reference_similarity_thresh
    sequence_similarity_thresh = args.sequence_similarity_thresh
    position_gap_thresh = args.position_gap_thresh
    regularization = args.regularization
    background_freq = args.background
    kstar = args.kstar
    pstar = args.pstar
    
    # Housekeeping
    if SEED is None or SEED <= 0:
        SEED = np.random.randint(2**32)
    rng = np.random.default_rng(seed=SEED)

    if reference_id is None or reference_id.lower() == "none":
        if verbosity:
            print("No reference entry specified.")
        reference_id = None

    do_compute_background = False
    if isinstance(background_freq, str) and background_freq.lower() == "default":
        background_freq = DEFAULT_BACKGROUND_FREQ
    elif background_freq is None or (
            isinstance(background_freq, str) and background_freq.lower() == "none"
    ):
        # Mark to compute background frequency from MSA
        do_compute_background = True
        background_freq = None
    else:
        msg = f"Cannot handle given argument for background: {background_freq}"
        raise RuntimeError(msg)

    SCADIR = os.path.join(OUTDIR, "sca_results")
    IMGDIR = os.path.join(OUTDIR, "images")
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(SCADIR, exist_ok=True)
    os.makedirs(IMGDIR, exist_ok=True)

    # Load MSA
    msa_obj_orig, msa_orig, seqids_orig, sym_map = load_msa(
        MSA_FPATH, format="fasta", 
        mapping=None,  # TODO: consider allowing for specified mapping
        verbosity=1
    )
    _, NUM_POS_ORIG = msa_orig.shape
    NSYMS = len(sym_map)
    
    if verbosity:
        print(f"Loaded MSA. shape: {msa_orig.shape} (sequences x positions)")
        print(f"Symbols: {sym_map.aa_list}")

    # Preprocessing
    msa, xmsa, seqids, weights, fi0_pretrunc, \
    retained_sequences, retained_positions, ref_results = preprocess_msa(
        msa_orig, seqids_orig, 
        mapping=sym_map,
        gap_truncation_thresh=gap_truncation_thresh,
        sequence_gap_thresh=sequence_gap_thresh,
        reference_id=reference_id,
        reference_similarity_thresh=reference_similarity_thresh,
        sequence_similarity_thresh=sequence_similarity_thresh,
        position_gap_thresh=position_gap_thresh,
        verbosity=1,
    )

    np.save(f"{SCADIR}/retained_sequences.npy", retained_sequences)
    np.save(f"{SCADIR}/retained_positions.npy", retained_positions)
    np.save(f"{SCADIR}/retained_sequence_ids.npy", seqids)
    np.save(f"{SCADIR}/sequence_weights.npy", weights)
    np.save(f"{SCADIR}/msa.npy", msa)
    sparse.save_npz(
        f"{SCADIR}/Xsp.npz", 
        sparse.csr_matrix(xmsa.reshape([xmsa.shape[0], -1]))
    )
    with open(f"{SCADIR}/sym2int.json", "w") as f:
        json.dump(sym_map.sym2int, f)

    # Plot gap frequency by position
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
    plt.savefig(f"{IMGDIR}/gap_freq_by_position.png")
    plt.close()

    # Compute the background frequencies if needed and store as an array
    if do_compute_background:
        if verbosity:
            print("Computing background frequency from full MSA")
        background_freq = compute_background_freqs(msa_obj_orig, gapstr="-")
    if verbosity:
        print("Background frequencies:")
        print("  ", ", ".join([
            f"{k}: {background_freq[k]:.3g}" 
            for k in np.sort(list(background_freq.keys()))
        ]))
    background_freq_array = np.zeros(len(background_freq))
    for a in background_freq:
        background_freq_array[sym_map[a]] = background_freq[a]    
    background_freq_array = background_freq_array / background_freq_array.sum()

    # Plot sequence similarity
    plot_sequence_similarity(
        xmsa, IMGDIR,
    )
    
    # Run SCA
    sca_results = run_sca(
        xmsa, weights,
        background_map=background_freq,
        mapping=sym_map,
        background_arr=background_freq_array,
        regularization=regularization,
        return_keys=["Di", "Cij_raw", "Cij_corr"],
        pbar=PBAR,
        leave_pbar=True,
    )

    # fi0 = sca_results["fi0"]
    # fia = sca_results["fia"]
    # fijab = sca_results["fijab"]
    # Dia = sca_results["Dia"]
    Di = sca_results["Di"]
    # Cijab_raw = sca_results["Cijab_raw"]
    Cij_raw = sca_results["Cij_raw"]
    # phi_ia = sca_results["phi_ia"]
    # Cijab_corr = sca_results["Cijab_corr"]
    Cij = sca_results["Cij_corr"]
    del sca_results  # relieve memory

    # Save SCA results
    np.save(f"{SCADIR}/conservation.npy", Di)
    np.save(f"{SCADIR}/sca_matrix.npy", Cij)
    
    # Determine the top conserved positions
    topk_conserved_msa_pos, top_conserved_Di = get_top_k_conserved_retained_positions(
        retained_positions, Di, n_top_conserved
    )

    if verbosity:
        print("top k conserved MSA positions:", topk_conserved_msa_pos)

    # Plot conservation
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
    ax.set_xlim(0, NUM_POS_ORIG)
    ax.set_xlabel(f"Position")
    ax.set_ylabel("Relative Entropy (KL Divergence, $D_i$)")
    ax.set_title(f"Position-wise Conservation")
    plt.savefig(f"{IMGDIR}/positional_conservation.png")
    plt.close()

    # Eigendecomposition of C_ij (raw and corrected)
    # evals_cov, _ = np.linalg.eigh(Cij_raw)
    # evals_cov = np.flip(evals_cov)

    evals_sca, evecs_sca = np.linalg.eigh(Cij)
    evals_sca = np.flip(evals_sca)
    evecs_sca = np.flip(evecs_sca, axis=1)

    if verbosity:
        # print(f"Eigenvalue spectrum of Covariance Matrix: " + 
        #     f"{evals_cov.min():.3g}, {evals_cov.max():.3f}")
        print(f"Eigenvalue spectrum of SCA Matrix: " + 
            f"{evals_sca.min():.3g}, {evals_sca.max():.3f}")
    
    # Plot Covariance Matrix
    fig, ax = plt.subplots(1, 1)
    sc = ax.imshow(
        Cij_raw, 
        cmap="Blues", 
        origin="lower",
        vmax=None,
    )
    fig.colorbar(sc, label="Covariation")
    ax.set_xlabel("(Retained) Position i")
    ax.set_ylabel("(Retained) Position j")
    ax.set_title("Covariance Matrix")
    plt.savefig(f"{IMGDIR}/covariance_matrix.png")
    plt.close()

    # Plot SCA Matrix
    fig, ax = plt.subplots(1, 1)
    sc = ax.imshow(
        Cij, 
        cmap="Blues", 
        origin="lower",
        vmax=None,
    )
    fig.colorbar(sc, label="Covariation")
    ax.set_xlabel("(Retained) Position i")
    ax.set_ylabel("(Retained) Position j")
    ax.set_title("SCA Matrix")
    plt.savefig(f"{IMGDIR}/sca_matrix.png")
    plt.close()
    
    # Perform bootstrapping to get eigenvalue null distribution
    DO_SHUFFLING = N_BOOT > 0
    evals_shuff_saveas = f"{SCADIR}/evals_shuff.npy"
    
    def shuffle_columns(m, rng=None):
        rng = np.random.default_rng(rng)
        r, c = m.shape
        idx = np.argsort(rng.random((r, c)), axis=0)
        return m[idx, np.arange(c)]

    evals_shuff = np.full([N_BOOT, *evals_sca.shape], np.nan)
    if DO_SHUFFLING:
        for iteridx in tqdm.trange(N_BOOT):
            msa_shuff = shuffle_columns(msa, rng=rng)
            xmsa_shuff = np.eye(NSYMS, dtype=bool)[msa_shuff][:,:,:-1]
            res = run_sca(
                xmsa_shuff, weights,
                background_map=background_freq,
                mapping=sym_map,
                background_arr=background_freq_array,
                regularization=regularization,
                return_keys=["Cij_corr"],
                pbar=PBAR,
                leave_pbar=False,
            )
            cij_shuff = res["Cij_corr"]
            evals = np.linalg.eigvalsh(cij_shuff)
            evals_shuff[iteridx] = np.flip(evals)
        np.save(evals_shuff_saveas, evals_shuff)
    elif os.path.isfile(evals_shuff_saveas):
        if verbosity:
            print("Skipping bootstrap. Loading existing null evals at: ".format(
                evals_shuff_saveas
            ))
        evals_shuff = np.load(evals_shuff_saveas)
    else:
        evals_shuff = []
        if verbosity:
            print("Skipping bootstrap. No existing eigenvalue data found.")

    # Plot SCA matrix spectrum null vs data
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
    plt.savefig(f"{IMGDIR}/sca_matrix_spectrum.png")
    plt.close()
    
    # Determine k^*, the number of significant eigenvalues. See SI G of [1]
    cutoff = np.mean(evals_shuff[:,1]) + 2 * np.std(evals_shuff[:,1])
    kstar_id = np.sum(evals_sca > cutoff)
    if verbosity:
        print("significant eigenvalue cutoff:", cutoff)
        print(f"Identified {kstar_id} significant eigenvalues:\n", 
              evals_sca[:kstar_id])
    if kstar <= 0:
        kstar = kstar_id
        if verbosity:
            print(f"Setting kstar={kstar}")
    else:
        kstar = min(kstar, len(evals_sca))
        if verbosity:
            print(f"Overriding kstar from command line input!")
            print(f"Setting kstar={kstar}")
    
    # Consider top kstar values, excluding top value
    sig_evals_sca = evals_sca[:kstar]
    sig_evecs_sca = evecs_sca[:,:kstar]

    # Save kstar, full eigendecomp, and significant eigendecomp
    np.savetxt(f"{SCADIR}/kstar_identified.txt", [kstar_id], fmt="%d")
    np.savetxt(f"{SCADIR}/kstar.txt", [kstar], fmt="%d")
    np.save(f"{SCADIR}/all_evals_sca.npy", evals_sca)
    np.save(f"{SCADIR}/all_evecs_sca.npy", evecs_sca)
    np.save(f"{SCADIR}/significant_evals_sca.npy", sig_evals_sca)
    np.save(f"{SCADIR}/significant_evecs_sca.npy", sig_evecs_sca)

    # Plot eigenvalue distribution null vs data
    fig, ax = plt.subplots(1, 1)
    # Histogram of data eigenvalues
    counts, bins, patches = ax.hist(
        evals_sca, bins=100, color="black", alpha=0.8, log=True, label="Data"
    )
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    h, bin_edges = np.histogram(evals_shuff.flatten(), bins=bins)
    ax.axvline(cutoff, 0, 1, linestyle="--", color="grey")
    ax.plot(
        bin_centers, h / N_BOOT, 
        color="red", 
        lw=1.5, 
        label="Null"
    )
    ax.legend()
    ax.set_xlabel(f"$\\lambda$")
    ax.set_ylabel(f"Count")
    ax.set_title(f"Spectral decomposition")
    plt.savefig(f"{IMGDIR}/sca_matrix_spectrum_vs_null.png")
    plt.close()

    # Dendrogram of SCA matrix
    plot_dendrogram(Cij, nclusters=kstar, imgdir=IMGDIR)
    
    # Apply ICA
    rho = 1e-1
    tol = 1e-7
    v_ica_normalized, _, _ = apply_ica(
        sig_evecs_sca, 
        rho=rho, tol=tol, maxiter=1E6, 
        max_attempts=5, 
        verbosity=verbosity,
    )
    
    # Get groups from top p% empirical distribution
    groups = get_groups(v_ica_normalized, p=pstar)

    # Save groups in MSA coordinates
    subdir = f"{OUTDIR}/groups"
    os.makedirs(subdir, exist_ok=True)
    for i in range(len(groups)):
        np.save(f"{subdir}/group_{i+1}_msapos.npy", groups[i])

    # Plot data and groups in EV coords (2-dimensional)
    EVIDXS_AND_GROUP_IDXS = [  # ((EVi, EVj), [group_indices])
        ((0, 1), "all"),
        ((1, 2), "all"),
        ((2, 3), "all"),
        ((3, 4), "all"),
        ((4, 5), "all"),
        ((5, 6), "all"),
        ((0, 1), [0, 1, 2]),
        ((1, 2), [0, 1, 2]),
    ]
    for evidxs, group_idxs in EVIDXS_AND_GROUP_IDXS:
        plot_data_2d(
            "ev", evidxs, group_idxs, groups, sig_evecs_sca, IMGDIR,
        )
    
    # Plot data and groups in EV coords (3-dimensional)
    EVIDXS_AND_GROUP_IDXS = [  # ((EVi, EVj, EVk), [group_indices])
        ((0, 1, 2), "all"),
        ((1, 2, 3), "all"),
        ((0, 1, 2), [0, 1, 2]),
        ((1, 2, 3), [0, 1, 2]),
    ]
    for evidxs, group_idxs in EVIDXS_AND_GROUP_IDXS:
        plot_data_3d(
            "ev", evidxs, group_idxs, groups, sig_evecs_sca, IMGDIR,
        )
    
    # Plot data and groups in IC coords (2-dimensional)
    ICIDXS_AND_GROUP_IDXS = [  # ((ICi, ICj), [group_indices])
        ((0, 1), "all"),
        ((1, 2), "all"),
        ((2, 3), "all"),
        ((3, 4), "all"),
        ((4, 5), "all"),
        ((5, 6), "all"),
        ((0, 1), [0, 1, 2]),
        ((1, 2), [0, 1, 2]),
    ]
    for icidxs, group_idxs in ICIDXS_AND_GROUP_IDXS:
        plot_data_2d(
            "ic", icidxs, group_idxs, groups, v_ica_normalized, IMGDIR,
        )
    
    # Plot data and groups in IC coords (3-dimensional)
    ICIDXS_AND_GROUP_IDXS = [  # ((ICi, ICj, ICk), [group_indices])
        ((0, 1, 2), "all"),
        ((1, 2, 3), "all"),
        ((0, 1, 2), [0, 1, 2]),
        ((1, 2, 3), [0, 1, 2]),
    ]
    for icidxs, group_idxs in ICIDXS_AND_GROUP_IDXS:
        plot_data_3d(
            "ic", icidxs, group_idxs, groups, v_ica_normalized, IMGDIR,
        )

    # Map MSA positions to raw sequence positions, then save
    rawseq_idxs = get_rawseq_indices_of_msa(msa_obj_orig)
    rawseq_idxs = rawseq_idxs[retained_sequences,:]
    rawseq_idxs = rawseq_idxs[:,retained_positions]
    
    # Save residue groups by raw sequence position
    group_rawseq_positions = get_rawseq_positions_in_groups(
        rawseq_idxs, groups
    )
    group_rawseq_positions_by_entry = get_group_rawseq_positions_by_entry(
        msa_obj_orig, retained_sequences, groups, group_rawseq_positions
    )
    for gidx in range(len(groups)):
        subdir = f"{OUTDIR}/sca_groups/group_{gidx + 1}"
        os.makedirs(subdir, exist_ok=True)
        for i, seqidx in enumerate(retained_sequences):
            entry = msa_obj_orig[int(seqidx)]
            id = entry.id
            group_arr = group_rawseq_positions_by_entry[id][gidx]
            np.save(f"{subdir}/group_{gidx + 1}_{id}.npy", group_arr)
    
    if verbosity:
        print("Done!")


def apply_ica(
        sig_evecs_sca, *, 
        rho,
        tol,
        maxiter, 
        max_attempts,
        verbosity=1,
):
    n_attempts = 0
    while n_attempts < max_attempts:
        n_attempts += 1
        w_ica, ica_delta = run_ica(
            sig_evecs_sca.T, 
            rho=rho,
            tol=tol,
            maxiter=maxiter,
        )
        if w_ica is None:
            # ICA failed to converge
            if verbosity:
                msg = f"ICA did not converge with parameters rho={rho:3g}, " + \
                        f"tol={tol:.3g}, maxiter={maxiter}. " + \
                        f"(Reached tol={ica_delta:.3})"
                print(msg)
            maxiter *= 2
            rho /= 2
        else:
            # ICA succeeded
            v_ica = sig_evecs_sca @ w_ica.T
            if verbosity:
                print(f"ICA succeeded after {n_attempts} attempts. (tol={tol:.2g})")
            break
    
    # Check success
    if w_ica is None:
        raise RuntimeError(f"ICA failed to converge in {max_attempts} attempts.")

    # Normalize V and ensure positivity of maximum entry.
    v_ica_normalized = v_ica / np.sqrt(np.sum(np.square(v_ica), axis=0))
    for i in range(v_ica.shape[1]):
        maxpos = np.argmax(np.abs(v_ica_normalized[:,i]))
        if v_ica_normalized[maxpos,i] < 0:
            v_ica_normalized[:,i] *= -1
    return v_ica_normalized, v_ica, w_ica


def get_groups(v_ica_normalized, p=95):
    groups = []
    to_be_assigned = np.ones(len(v_ica_normalized), dtype=bool)
    for i in range(v_ica_normalized.shape[1]):
        top_p_idxs = np.where(
            (v_ica_normalized[:,i] >= np.percentile(
                v_ica_normalized[to_be_assigned,i], p)) \
            & (to_be_assigned)
        )[0]
        to_be_assigned[top_p_idxs] = False
        groups.append(top_p_idxs)
    return groups


def plot_data_2d(
        ic_or_ev, axidxs, group_idxs, groups, 
        data, 
        imgdir,
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
        ax.scatter(
            data[g,axi], data[g,axj],
            alpha=1, 
            edgecolor='k',
            label=f"group {gidx + 1}",
        )
    ax.plot(0, 0, "ro")
    rx, ry = ax.get_xlim()[1], ax.get_ylim()[1]
    ax.plot([0, rx], [0, 0], "k-", alpha=0.5)
    ax.plot([0, 0], [0, ry], "k-", alpha=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlabel(f"{ic_or_ev.upper()} {axi + 1}")
    ax.set_ylabel(f"{ic_or_ev.upper()} {axj + 1}")
    ax.set_title(title)
    groupstr = "".join([str(i+1) for i in group_idxs])
    plt.tight_layout()
    plt.savefig(f"{imgdir}/{ic_or_ev}{axi+1}{axj+1}_groups_{groupstr}.png",
                bbox_inches="tight")
    plt.close()
    return


def plot_data_3d(
        ic_or_ev, axidxs, group_idxs, groups, 
        data, 
        imgdir,
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
        ax.scatter(
            data[g,axi], data[g,axj], data[g,axk], 
            alpha=1, 
            edgecolor='k',
            label=f"group {gidx + 1}",
        )
    ax.plot(0, 0, "ro")
    rx, ry, rz = ax.get_xlim()[1], ax.get_ylim()[1], ax.get_zlim()[1]
    ax.plot([0, rx], [0, 0], [0, 0], "k-", alpha=0.5)
    ax.plot([0, 0], [0, ry], [0, 0], "k-", alpha=0.5)
    ax.plot([0, 0], [0, 0], [0, rz], "k-", alpha=0.5)
    ax.view_init(elev=30, azim=40)   # elev ~ tilt, azim ~ around z
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlabel(f"{ic_or_ev.upper()} {axi + 1}")
    ax.set_ylabel(f"{ic_or_ev.upper()} {axj + 1}")
    ax.set_zlabel(f"{ic_or_ev.upper()} {axk + 1}")
    ax.set_title(title)
    groupstr = "".join([str(i+1) for i in group_idxs])
    plt.tight_layout()
    plt.savefig(f"{imgdir}/{ic_or_ev}{axi+1}{axj+1}{axk+1}_groups_{groupstr}.png", 
                bbox_inches="tight")
    plt.close()
    return


def plot_dendrogram(
        Cij, *, 
        nclusters=10,
        imgdir,
):
    Z = sch.linkage(pdist(Cij, metric='euclidean'), method='ward')
    clusters = sch.fcluster(Z, t=nclusters, criterion='maxclust')
    dendro = sch.dendrogram(Z, no_plot=True)
    leaf_indices = dendro['leaves']
    cmap = plt.cm.turbo
    cluster_colors = [to_hex(cmap(i)) for i in np.linspace(0, 1, nclusters)]
    def color_func(link_idx):
        if link_idx < len(clusters):  # Only color leaf nodes
            return cluster_colors[clusters[link_idx] - 1]
        return "#000000"
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7, 6), 
        gridspec_kw={'width_ratios': [0.2, 1]}
    )
    sch.dendrogram(
        Z,
        orientation='left',
        ax=ax1,
    #    color_threshold=max(Z[-nclusters+1, 2], 0.1),
        link_color_func=color_func,
        above_threshold_color='k'
    )
    ax1.set_ylabel('Position', fontsize='x-large')
    ax1.set_xticks([])
    ax1.set_yticks([])
    rearranged_data = Cij[leaf_indices][:, leaf_indices]
    im = ax2.imshow(
        rearranged_data, 
        aspect='auto', 
        cmap='Blues',
        interpolation='nearest', 
        origin='lower', 
        # vmin=0, vmax=1,
    )
    boundaries = np.where(np.diff(clusters[leaf_indices]))[0]
    for b in boundaries:
        ax2.axhline(b + 0.5, color='black', linestyle='--')
        ax2.axvline(b + 0.5, color='black', linestyle='--')
    ax2.set_title('Clustering of Positions', fontsize='x-large')
    ax2.set_xlabel('Position', fontsize='x-large')
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.tight_layout()
    plt.savefig(f"{imgdir}/dendrogram.png", bbox_inches="tight")
    plt.close()
    return


def plot_sequence_similarity(
        xmsa, imgdir
):
    npos = xmsa.shape[1]
    xmsa = xmsa.reshape([xmsa.shape[0], -1])
    similarity_matrix = (xmsa @ xmsa.T) / npos
    upper_vals = similarity_matrix[np.triu_indices_from(similarity_matrix)]
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8,5))

    Z = sch.linkage(similarity_matrix, method="complete", metric="cityblock")
    dendro = sch.dendrogram(Z, no_plot=True)
    idxs = dendro["leaves"]
    
    ax1.hist(upper_vals, int(round(npos/2)))
    ax1.set_xlabel("Pairwise sequence identities")
    ax1.set_ylabel("Count")

    sc = ax2.imshow(
        similarity_matrix[np.ix_(idxs, idxs)],
        vmin=0, vmax=1
    )
    plt.colorbar(sc)
    plt.tight_layout()
    plt.savefig(f"{imgdir}/sequence_similarity.png", bbox_inches="tight")
    plt.close()
    return


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
