"""Core SCA functionality.

"""

import numpy as np
from numpy.typing import NDArray
import tqdm

from mysca.mappings import SymMap, DEFAULT_MAP


def run_sca(
        xmsa: NDArray[np.bool],
        ws: NDArray[np.float64],
        background_map: dict[str, float],
        mapping: SymMap = DEFAULT_MAP,
        background_arr: NDArray[np.float64] | None = None,
        regularization: float = 0.03,
        return_keys: str = "all",
        pbar: bool = True,
        leave_pbar: bool = True,
):
    """Run SCA algorithm on given MSA matrix

    Args: # TODO
        xmsa (_type_): _description_
        ws (_type_): _description_
        background_map (_type_): _description_
        mapping (SymMap): SymMap mapping symbols to integer values.
        background_arr (_type_, optional): _description_. Defaults to None.
        regularization (float, optional): _description_. Defaults to 0.03.
        return_keys (str, optional): _description_. Defaults to "all".
        pbar (bool, optional): _description_. Defaults to True.
        leave_pbar (bool, optional): _description_. Defaults to True.

    Returns:  # TODO
        _type_: _description_
    """
    lam = regularization  # brevity
    qa = background_arr  # brevity
    results = {}

    nseq, npos, naas = xmsa.shape

    # Dictionary size
    nsyms = naas + 1

    # Compute positional conservation
    ws_norm = ws / ws.sum()
    fi0 = 1 - np.sum(ws[:,None,None] * xmsa, axis=(0,2)) / ws.sum()
    fia = (1 - lam) * np.sum(ws_norm[:,None,None] * xmsa, axis=0) + lam / nsyms

    # Compute correlated conservation
    fijab = np.full([npos, npos, naas, naas], np.nan)
    for i in tqdm.trange(npos, disable=not pbar, leave=leave_pbar):
        ci = xmsa[:,i,:]
        for j in range(i, npos):
            cj = xmsa[:,j,:]
            f = (1 - lam) * (ci.T @ (ws_norm[:, None] * cj)) + lam / nsyms**2
            fijab[i,j,:,:] = f
            fijab[j,i,:,:] = f.T

    if qa is None:
        qa = np.zeros(naas)
        for a in background_map:
            qa[mapping[a]] = background_map[a]
        qa = qa / qa.sum()

    Dia = np.nan * np.ones([npos, naas])
    Dia[:] = fia * np.log(fia / qa) + (1 - fia) * np.log((1 - fia) / (1 - qa))
    q0hat = np.sum(~np.any(xmsa, axis=-1)) / (nseq * npos)
    qahat = (1 - q0hat) * qa
    Di = np.sum(fia * np.log(fia / qahat), axis=1)
    if q0hat > 0:
        with np.errstate(divide='ignore'):
            Di += fi0 * np.where(fi0 > 0, np.log(fi0 / q0hat), 0)

    Cijab_raw = fijab - fia[:,None,:,None] * fia[None,:,None,:]
    Cij_raw = np.sqrt(np.sum(np.square(Cijab_raw), axis=(-1, -2)))
    # Cij_raw = (Cij_raw + Cij_raw.T) / 2
    phi_ia = np.log((fia * (1 - qa)) / ((1 - fia) * qa))
    Cijab_corr = phi_ia[:,None,:,None] * phi_ia[None,:,None,:] * Cijab_raw
    Cij_corr = np.sqrt(np.sum(np.square(Cijab_corr), axis=(-1,-2)))
    # Cij = (Cij + Cij.T) / 2

    if return_keys == "all":
        results["fi0"] = fi0
        results["fia"] = fia
        results["fijab"] = fijab
        results["Dia"] = Dia
        results["Di"] = Di
        results["Cijab_raw"] = Cijab_raw
        results["Cij_raw"] = Cij_raw
        results["phi_ia"] = phi_ia
        results["Cijab_corr"] = Cijab_corr
        results["Cij_corr"] = Cij_corr
    else:
        for k in return_keys:
            results[k] = eval(k)
    return results


def run_ica(
        v: NDArray, 
        rho: float = 1e-4, 
        tol: float = 1e-7, 
        maxiter: int = 1000000,
        verbosity: int = 1,
):
    """Implements ICA using the infomax algorithm.

    Independent components V^* are computed by applying the returned matrix
    W to the eigenvectors in input V: V* = WV, with V of shape (m, k)

    Refs: 
        [1] Bell and Sejnowski, 1995
        [2] SI to Rivoire et al., 2016
    
    Args:
        (NDArray) v: 2d eigenvector array. Shape (k, m), where k is the number 
            of eigenvectors.
        (float) rho: ICA stepsize parameter. Default 1e-4.
        (float) tol: Convergence threshold. Default 1e-7
        (int) maxiter: Maximum steps before halting. Default 1000000.
        (int) verbosity: Verbosity level. Default 1.

    Returns:
        (NDArray) Independent Component vectors W. Shape (k, k).
    """
    k, m = v.shape
    id = np.eye(k)
    w = np.eye(k)
    dw = 100.
    itercount = 0
    while itercount < maxiter:
        y = w @ v
        g = 1 / (1 + np.exp(-y))
        dw = rho * (id + (1 - 2 * g) @ y.T) @ w
        w += dw
        if np.max(np.abs(dw)) < tol:
            if verbosity:
                print(f"Converged in {itercount} iterations")
            return w
        itercount += 1
    print(f"Did not converge in {maxiter} iterations")
