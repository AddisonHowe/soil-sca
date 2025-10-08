"""Figure generation script

"""

import os, sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tqdm as tqdm


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

    NFIGS = 2
    fmt = "pdf"
    transparent = True

    # Housekeeping
    os.makedirs(outdir, exist_ok=True)
    pbar = tqdm.tqdm(desc="Plotting", total=NFIGS, disable=disable_pbar)
    printv = get_printv(verbosity, pbar_default=not disable_pbar)


    # Load data
    # ...

    # Generate plots
    printv("Generating plots...")

    saveas = "saveasname1"
    bbox_inches = None
    if saveas:
        make_subplot1(
            # args...
            outdir=outdir,
            saveas=saveas,
            format=fmt,
            transparent=transparent,
            bbox_inches=bbox_inches, 
        )
    pbar.update(1)

    saveas = "saveasname2"
    bbox_inches = None
    if saveas:
        make_subplot2(
            # args...
            outdir=outdir,
            saveas=saveas,
            format=fmt,
            transparent=transparent,
            bbox_inches=bbox_inches, 
        )
    pbar.update(1)
    
    pbar.close()
    print("Done!")


def make_subplot1(
        *,
        outdir, 
        saveas, 
        format="png",
        transparent=True,
        bbox_inches=None, 
):
    fig, ax = plt.subplots(1, 1)
    # Do plotting
    # ...

    # Save and close
    plt.savefig(
        f"{outdir}/{saveas}.{format}", format=format, 
        transparent=transparent, bbox_inches=bbox_inches
    )
    plt.close()
    return


def make_subplot2(
        *,
        outdir, 
        saveas, 
        format="png",
        transparent=True,
        bbox_inches=None, 
):
    fig, ax = plt.subplots(1, 1)
    # Do plotting
    # ...

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
