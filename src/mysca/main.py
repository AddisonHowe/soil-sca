"""Full SCA pipeline

See references:
    [1] SI to Rivoire et al., 2016

"""

import argparse
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tqdm as tqdm
from mpl_toolkits.mplot3d import Axes3D

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA

from mysca.io import load_msa
from mysca.io import get_residue_sequence_from_pdb_structure
from mysca.io import load_pdb_structure
from mysca.mappings import SymMap
from mysca.preprocess import preprocess_msa
from mysca.preprocess import compute_background_freqs
from mysca.core import run_sca, run_ica
from mysca.helpers import get_top_k_conserved_retained_positions
from mysca.helpers import get_conserved_rawseq_positions
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
    parser.add_argument("-s", "--structure_dir", type=str, default=None,
                        help="Path to directory containing structure pdb files.")
    parser.add_argument("-o", "--outdir", type=str, required=True, 
                        help="Output directory.")
    parser.add_argument("--pbar", action="store_true")
    parser.add_argument("-v", "--verbosity", type=int, default=1)

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


    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(args)


def main(args):

    # Process command line args
    struct_dir = args.structure_dir
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
    
    # Housekeeping
    if SEED is None or SEED <= 0:
        SEED = np.random.randint(2**32)
    rng = np.random.default_rng(seed=SEED)

    if reference_id is None or reference_id.lower() == "none":
        if verbosity:
            print("No reference entry specified.")
        reference_id = None
    
    if struct_dir is None or struct_dir.lower() == "none":
        if verbosity:
            print("No structure directory specified.")
        struct_dir = None

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

    IMGDIR = os.path.join(OUTDIR, "images")
    os.makedirs(OUTDIR, exist_ok=True)
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

    # Plot gap frequency by position
    fig, ax = plt.subplots(1, 1)
    ax.plot(fi0_pretrunc, ".")
    ax.hlines(position_gap_thresh, *ax.get_xlim(), linestyle='--', color="r", label="cutoff")
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
    
    # Run SCA
    sca_results = run_sca(
        xmsa, weights,
        background_map=background_freq,
        mapping=sym_map,
        background_arr=background_freq_array,
        regularization=regularization,
        return_keys="all",
        pbar=PBAR,
        leave_pbar=True,
    )

    fi0 = sca_results["fi0"]
    fia = sca_results["fia"]
    fijab = sca_results["fijab"]
    Dia = sca_results["Dia"]
    Di = sca_results["Di"]
    Cijab_raw = sca_results["Cijab_raw"]
    Cij_raw = sca_results["Cij_raw"]
    phi_ia = sca_results["phi_ia"]
    Cijab_corr = sca_results["Cijab_corr"]
    Cij = sca_results["Cij_corr"]
    
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

    # Map MSA positions to raw sequence positions
    rawseq_idxs = get_rawseq_indices_of_msa(msa_obj_orig)
    rawseq_idxs = rawseq_idxs[retained_sequences,:]
    rawseq_idxs = rawseq_idxs[:,retained_positions]

    # Eigendecomposition of C_ij (raw and corrected)
    evals_sca_raw, evecs_sca_raw = np.linalg.eigh(Cij_raw)
    evals_sca_raw = np.flip(evals_sca_raw)
    evecs_sca_raw = np.flip(evecs_sca_raw, axis=1)

    evals_sca, evecs_sca = np.linalg.eigh(Cij)
    evals_sca = np.flip(evals_sca)
    evecs_sca = np.flip(evecs_sca, axis=1)

    if verbosity:
        print(f"      Eigenvalue spectrum of Cij (raw): " + 
            f"{evals_sca_raw.min():.3g}, {evals_sca_raw.max():.3f}")
        print(f"Eigenvalue spectrum of Cij (corrected): " + 
            f"{evals_sca.min():.3g}, {evals_sca.max():.3f}")
    
    # Plot Covariance and SCA matrices
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

    # Dendrogram
    Z = linkage(pdist(Cij, metric='euclidean'), method='ward')
    n_clusters = 10
    clusters = fcluster(Z, t=n_clusters, criterion='maxclust')
    dendro = dendrogram(Z, no_plot=True)
    leaf_indices = dendro['leaves']
    cmap = plt.cm.turbo
    cluster_colors = [to_hex(cmap(i)) for i in np.linspace(0, 1, n_clusters)]
    def color_func(link_idx):
        if link_idx < len(clusters):  # Only color leaf nodes
            return cluster_colors[clusters[link_idx] - 1]
        return "#000000"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 6), 
                                gridspec_kw={'width_ratios': [0.2, 1]})
    dendrogram(
        Z,
        orientation='left',
        ax=ax1,
    #    color_threshold=max(Z[-n_clusters+1, 2], 0.1),
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
    plt.savefig(f"{IMGDIR}/dendrogram.png", bbox_inches="tight")
    plt.close()
    
    # Determine the conserved positions in each raw sequence
    conserved_aa_idxs = get_conserved_rawseq_positions(
        msa_obj_orig, retained_sequences, topk_conserved_msa_pos
    )

    # Load PDB structures if available
    if struct_dir is None:
        if verbosity:
            print("No structure directory specified. Skipping analysis of PDB files.")
    else:
        pdb_mappings = {}
        missing_pdb_entries = []
        nan_filler = np.array([np.nan, np.nan, np.nan])
        for i, seqidx in enumerate(retained_sequences):
            entry = msa_obj_orig[int(seqidx)]
            id = entry.id
            conserved_positions = conserved_aa_idxs[i]
            pdbfpath = f"{struct_dir}/{id}.pdb"
            if not os.path.isfile(pdbfpath):
                missing_pdb_entries.append(id)
                continue
            if -1 in conserved_positions:
                print(f"Entry {id} does not contain all conserved positions.")
                continue
            structure = load_pdb_structure(pdbfpath, id=id, quiet=True)
            residues = get_residue_sequence_from_pdb_structure(structure)
            conserved_residues = [
                residues[i] if i >= 0 else None for i in conserved_positions
            ]
            conserved_residue_positions = np.array(
                [nan_filler if r is None else r['CA'].coord for r in conserved_residues]
            )
            pdb_mappings[id] = conserved_residue_positions

        if len(pdb_mappings) == 0:
            print("No PDB files found!")
        else:
            # Compute pairwise distance matrix for conserved positions.
            ncombs = n_top_conserved * (n_top_conserved - 1) // 2
            all_pdists = np.nan * np.ones([len(pdb_mappings), ncombs])

            for i, id in enumerate(sorted(list(pdb_mappings.keys()))):
                x = pdb_mappings[id]
                dists = pdist(x, metric="euclidean")
                all_pdists[i] = dists

            # Plot pairwise distances between conserved residues
            fig, ax = plt.subplots(1, 1)
            sc = ax.imshow(all_pdists, cmap="plasma")
            ax.set_xlabel("pairwise distance")
            ax.set_ylabel("variant")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig = ax.figure
            cbar = fig.colorbar(sc, cax=cax)
            cbar.ax.set_ylabel("Distance (Angstroms)")
            plt.savefig(f"{IMGDIR}/conserved_residues_pdists.png")
            plt.close()

            # Normalize the pairwise distance data and plot
            all_pdists_centered = (all_pdists - all_pdists.mean(0)) / all_pdists.std(0)

            fig, ax = plt.subplots(1, 1)
            sc = ax.imshow(all_pdists_centered, cmap="plasma")
            ax.set_xlabel("pairwise distance")
            ax.set_ylabel("variant")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig = ax.figure
            cbar = fig.colorbar(sc, cax=cax)
            cbar.ax.set_ylabel("Distance (Normalized)")
            plt.savefig(f"{IMGDIR}/conserved_residues_pdists_normalized.png")
            plt.close()

            # Apply pca to pairwise distance data
            pca = PCA(n_components=min(20, ncombs))
            pca.fit(all_pdists_centered)
            data_pca = pca.transform(all_pdists_centered)

            # Plot PCs 1 and 2
            fig, ax = plt.subplots(1, 1)
            ax.plot(
                data_pca[:,0], data_pca[:,1], "."
            )
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("Conserved residues pairwise distance PCA")
            plt.savefig(f"{IMGDIR}/conserved_residues_pdists_pc1_pc2.png")
            plt.close()

            fig, ax = plt.subplots(1, 1)
            ax.plot(
                1 + np.arange(len(pca.explained_variance_ratio_)), 
                np.cumsum(pca.explained_variance_ratio_)
            )
            ax.set_xlabel("PC")
            ax.set_ylabel("explained variance")
            ax.set_title("Explained variance (cumulative proportion)")
            plt.savefig(f"{IMGDIR}/conserved_residues_pdists_exp_var.png")
            plt.close()

    # Perform bootstrapping

    def shuffle_columns(m, rng=None):
        rng = np.random.default_rng(rng)
        r, c = m.shape
        idx = np.argsort(rng.random((r, c)), axis=0)
        return m[idx, np.arange(c)]

    DO_SHUFFLING = N_BOOT > 0
    shuffling_saveas = f"{OUTDIR}/shuffled_cijs_corrected.npy"

    if DO_SHUFFLING:
        cijs_shuffled = np.full([N_BOOT, *Cij.shape], np.nan)
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
            cijs_shuffled[iteridx] = res["Cij_corr"]

        np.save(shuffling_saveas, cijs_shuffled)
    elif os.path.isfile(shuffling_saveas):
        if verbosity:
            print("Skipping bootstrap. Loading existing Cij_corr data at: ".format(
                shuffling_saveas
            ))
        cijs_shuffled = np.load(shuffling_saveas)
    else:
        if verbosity:
            print("Skipping bootstrap. No existing data found. Halting.")
        return  # TODO: Handle halt more gracefully. Allow for some continuation.

    # Compute null eigenvalue distribution
    evals_shuff = np.full([len(cijs_shuffled), *evals_sca.shape], np.nan)
    for i, cij_shuff in enumerate(cijs_shuffled):
        evals = np.linalg.eigvalsh(cij_shuff)
        evals_shuff[i] = np.flip(evals)

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
    sig_evals_sca = evals_sca[1:kstar]
    sig_evecs_sca = evecs_sca[:,1:kstar]

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
    
    # Apply ICA
    max_attemps = 5
    n_attempts = 0
    rho = 1e-4
    tol = 1e-6
    maxiter = 100000
    while n_attempts < max_attemps:
        n_attempts += 1
        w_ica, ica_delta = run_ica(
            sig_evecs_sca.T, 
            rho=rho,
            tol=tol,
            maxiter=maxiter
        )
        if w_ica is None:
            # ICA failed to converge
            if verbosity:
                msg = f"ICA did not converge with parameters rho={rho:3g}, " + \
                        f"tol={tol:.3g}, maxiter={maxiter}. " + \
                        f"(Reached tol={ica_delta:.3})"
                print(msg)
            maxiter *= 2
        else:
            # ICA succeeded
            v_ica = sig_evecs_sca @ w_ica.T
            if verbosity:
                print(f"ICA succeeded after {n_attempts} attempts. (tol={tol:.2g})")
            break
    if w_ica is None:
        raise RuntimeError(f"ICA failed to converge in {max_attemps} attempts.")

    v_ica_normalized = v_ica / np.sqrt(np.sum(np.square(v_ica), axis=0))
    for i in range(v_ica.shape[1]):
        maxpos = np.argmax(np.abs(v_ica_normalized[:,i]))
        if v_ica_normalized[maxpos,i] < 0:
            v_ica_normalized[:,i] *= -1
    
    # Get groups from top p% empirical distribution
    groups = []
    p = 95
    to_be_assigned = np.ones(len(v_ica_normalized), dtype=bool)
    for i in range(v_ica_normalized.shape[1]):
        top_p_idxs = np.where(
            (v_ica_normalized[:,i] >= np.percentile(
                v_ica_normalized[to_be_assigned,i], p)) \
            & (to_be_assigned)
        )[0]
        to_be_assigned[top_p_idxs] = False
        groups.append(top_p_idxs)

    # Save groups in MSA coordinates
    subdir = f"{OUTDIR}/groups"
    os.makedirs(subdir, exist_ok=True)
    for i in range(len(groups)):
        np.save(f"{subdir}/group_{i+1}_msapos.npy", groups[i])

    # Plot data and groups in IC coords (2-dimensional)
    ICIDXS_AND_GROUP_IDXS = [  # ((ICi, ICj), [group_indices])
        ((0, 1), "all"),
        ((1, 2), "all"),
        ((0, 1), [0, 1, 2]),
        ((1, 2), [0, 1, 2]),
    ]
    for icidxs, group_idxs in ICIDXS_AND_GROUP_IDXS:
        if group_idxs == "all":
            group_idxs = list(range(len(groups)))
        fig, ax = plt.subplots(1, 1)
        ici, icj = icidxs
        if icj >= v_ica_normalized.shape[1]:
            continue
        sc = ax.scatter(
            v_ica_normalized[:,ici], v_ica_normalized[:,icj],
            c='k', 
            alpha=0.2, 
            edgecolor='k',
        )
        for i, gidx in enumerate(group_idxs):
            if gidx >= len(groups):
                continue
            g = groups[gidx]
            ax.scatter(
                v_ica_normalized[g,ici], v_ica_normalized[g,icj],
                alpha=1, 
                edgecolor='k',
                label=f"group {gidx + 1}",
            )
        ax.plot(0, 0, "ro")
        rx, ry = ax.get_xlim()[1], ax.get_ylim()[1]
        ax.plot([0, rx], [0, 0], "k-", alpha=0.5)
        ax.plot([0, 0], [0, ry], "k-", alpha=0.5)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xlabel(f"IC {ici + 1}")
        ax.set_ylabel(f"IC {icj + 1}")
        ax.set_title(f"ICA and identified groups")
        groupstr = "".join([str(i+1) for i in group_idxs])
        plt.tight_layout()
        plt.savefig(f"{IMGDIR}/ic{ici+1}{icj+1}_groups_{groupstr}.png",
                    bbox_inches="tight")
        plt.close()

    
    # Plot data and groups in IC coords (3-dimensional)
    ICIDXS_AND_GROUP_IDXS = [  # ((ICi, ICj, ICk), [group_indices])
        ((0, 1, 2), "all"),
        ((1, 2, 3), "all"),
        ((0, 1, 2), [0, 1, 2]),
        ((1, 2, 3), [0, 1, 2]),
    ]
    for icidxs, group_idxs in ICIDXS_AND_GROUP_IDXS:
        if group_idxs == "all":
            group_idxs = list(range(len(groups)))
        fig = plt.figure(figsize=(12,5))
        ax = fig.add_subplot(111, projection='3d')
        ici, icj, ick = icidxs
        if ick >= v_ica_normalized.shape[1]:
            continue
        sc = ax.scatter(
            v_ica_normalized[:,ici], v_ica_normalized[:,icj], v_ica_normalized[:,ick], 
            c="k", 
            alpha=0.2, 
            edgecolor='k',
        )
        for i, gidx in enumerate(group_idxs):
            if gidx >= len(groups):
                continue
            g = groups[gidx]
            ax.scatter(
                v_ica_normalized[g,ici], v_ica_normalized[g,icj], v_ica_normalized[g,ick], 
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
        ax.set_xlabel(f"IC {ici + 1}")
        ax.set_ylabel(f"IC {icj + 1}")
        ax.set_zlabel(f"IC {ick + 1}")
        ax.set_title(f"ICA and identified groups")
        groupstr = "".join([str(i+1) for i in group_idxs])
        plt.tight_layout()
        plt.savefig(f"{IMGDIR}/ic{ici+1}{icj+1}{ick+1}_groups_{groupstr}.png", 
                    bbox_inches="tight")
        plt.close()
    

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
            # pdbfpath = f"{struct_dir}/{id}.pdb"
            group_arr = group_rawseq_positions_by_entry[id][gidx]
            # if os.path.isfile(pdbfpath):
            np.save(f"{subdir}/group_{gidx + 1}_{id}.npy", group_arr)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
