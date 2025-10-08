"""
Example usage:
    python scripts/pymol_sca.py -s Soil14.scaffold_576820813_c1_40 \
        --pdb_dir out/structure/K00370 \
        --groups_dir out/sca/K00370/sca_groups \
        --outdir out/sca/K00370/images \
        --groups 0 1 2
"""

import argparse
import os
import sys
import pymol
from pymol import cmd
import numpy as np

from mysca.constants import SECTOR_COLORS


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scaffold", type=str, required=True)
    parser.add_argument("--pdb_dir", type=str, required=True)
    parser.add_argument("--groups_dir", type=str, required=True)
    parser.add_argument("--groups", type=int, nargs='*',
                        help="Group indices, (starting at 0) that correspond " \
                        "to subdirectories group_<idx> of groups_dir. If -1, " \
                        "produce plots for all groups.")
    parser.add_argument("--multisector_group_idxs", type=int, nargs='*', 
                        default=None, 
                        help="Group indices to be plotted simultaneously, "
                        "(starting at 0) that correspond " \
                        "to subdirectories group_<idx> of groups_dir. If -1, " \
                        "produce plots for all groups.")
    parser.add_argument("-r", "--reference", type=str, default=None,
                        help="If specified, align the input scaffold to this " \
                        "reference.")
    parser.add_argument("-o", "--outdir", type=str, default=None)
    parser.add_argument("-v", "--verbosity", type=int, default=1)

    return parser.parse_args(args)


def plot_scaffold_by_sectors(
        scaffold, group_idxs, groups_basedir, *,
        struct_color, 
        group_colors, 
        group_style, 
        ref_scaffold, 
        outdir, 
        verbosity=1,
):
    gdir = f"{groups_basedir}"
    viewsdir = os.path.join(outdir, "views")
    os.makedirs(viewsdir, exist_ok=True)
    for gidx in group_idxs:
        group_color = group_colors[gidx]
        group_fpath = f"{gdir}/group_{gidx}/group_{gidx}_{scaffold}.npy"
        if os.path.isfile(group_fpath):
            group_selection = "group_selection"
            group = np.load(group_fpath)
            res_idxs = 1 + group
            selection_string = "resi " + "+".join(map(str, res_idxs))
            cmd.select(group_selection, selection_string)
            cmd.show(group_style, group_selection)
            cmd.color(group_color, group_selection)
        else:
            group_selection = None
            group = None
            if verbosity:
                print(f"Structure {scaffold} group {gidx} file not found: {group_fpath}")

        # Align the structure to the reference
        if ref_scaffold:
            cmd.align("struct", "ref_struct")
        
        # Save the primary plot
        cmd.png(f"{outdir}/{scaffold}_group{gidx}.png", dpi=300)

        # Save views of each side
        for ri in range(4):
            cmd.png(f"{viewsdir}/{scaffold}_group{gidx}_view{ri}.png", dpi=300)
            cmd.rotate("y", 90, "struct")
            if ref_scaffold:
                cmd.rotate("y", 90, "ref_struct")
        
        # Reset
        if group_selection:
            cmd.hide(group_style, group_selection)
            cmd.color(struct_color, group_selection)  # reset color
            cmd.delete(group_selection)
    return


def plot_scaffold_with_all_sectors(
        scaffold, group_idxs, groups_basedir, *, 
        group_colors, 
        ref_scaffold, 
        outdir, 
        nmax=None, 
        verbosity=1, 
):
    gdir = f"{groups_basedir}"
    if nmax is None:
        nmax = len(group_idxs)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        viewsdir = os.path.join(outdir, "views")
        os.makedirs(viewsdir, exist_ok=True)
    
    struct_color = "gray70"
    struct_style = "sticks"
    group_styles = ["spheres"] * len(group_idxs)

    if ref_scaffold:
        cmd.hide("everything", "ref_struct")

    cmd.hide("everything", "struct")
    cmd.show(struct_style, "struct")
    cmd.color(struct_color, "struct")
    cmd.bg_color("white")

    if ref_scaffold:
        cmd.align("struct", "ref_struct")
    group_idxs = group_idxs[0:min(nmax, len(group_idxs))]
    for i, gidx in enumerate(group_idxs):
        group_fpath = f"{gdir}/group_{gidx}/group_{gidx}_{scaffold}.npy"
        group_color = group_colors[gidx]
        if os.path.isfile(group_fpath):
            group_selection = f"group_selection{i}"
            group = np.load(group_fpath)
            res_idxs = 1 + group
            selection_string = "resi " + "+".join(map(str, res_idxs))
            cmd.select(group_selection, selection_string)
            cmd.show(group_styles[i], group_selection)
            cmd.color(group_color, group_selection)
        else:
            group_selection = None
            group = None
            if verbosity:
                print(f"Group {gidx} file not found: {group_fpath}")

    for ri in range(4):
        cmd.png(f"{viewsdir}/{scaffold}_groups_{",".join([str(i) for i in group_idxs])}_view{ri}.png", dpi=300)
        cmd.rotate("y", 90, "struct")
        if ref_scaffold:
                cmd.rotate("y", 90, "ref_struct")
    return


def _hex2color(x):
    return "0x" + x[1:]


def main(args):
    scaffold = args.scaffold
    pdb_dir = args.pdb_dir
    groups_basedir = args.groups_dir
    group_idxs = args.groups
    multisector_group_idxs = args.multisector_group_idxs
    ref_scaffold = args.reference
    outdir = args.outdir
    verbosity = args.verbosity

    if ref_scaffold is None or ref_scaffold.lower() == "none":
        ref_scaffold = None

    gdir = f"{groups_basedir}"
    
    # Get all sectors/groups if group_idxs is specified as -1 at the cmd line
    if len(group_idxs) == 1 and group_idxs[0] == -1:
        # Check group directory and include all groups present
        group_files = os.listdir(gdir)
        prefix = "group_"
        group_idxs = [
            f.removeprefix(prefix) for f in group_files if f.startswith(prefix)
        ]
        group_idxs = np.sort([int(x) for x in group_idxs])

    if multisector_group_idxs is None:
        multisector_group_idxs = group_idxs

    if outdir:
        os.makedirs(outdir, exist_ok=True)
    
    # Load the specified structure
    if verbosity:
        print("Scaffold:", scaffold)
    pdbfile = f"{pdb_dir}/{scaffold}.pdb"
    cmd.load(pdbfile, "struct")

    # Load a reference, if given
    if ref_scaffold:
        reffile = f"{pdb_dir}/{ref_scaffold}.pdb"
        cmd.load(reffile, "ref_struct")
    
    # Colors and styles
    struct_color = "gray70"
    struct_style = "sticks"
    group_colors = [_hex2color(x) for x in SECTOR_COLORS]
    group_style = "spheres"

    # Hide the loaded structure (and reference)
    cmd.hide("everything", "struct")
    if ref_scaffold:
        cmd.hide("everything", "ref_struct")
    
    cmd.show(struct_style, "struct")
    cmd.color(struct_color, "struct")
    cmd.bg_color("white")

    plot_scaffold_by_sectors(
            scaffold, group_idxs, gdir,
            struct_color=struct_color, 
            group_colors=group_colors, 
            group_style=group_style, 
            ref_scaffold=ref_scaffold, 
            outdir=outdir, 
            verbosity=verbosity,
    )
    
    subdir = os.path.join(outdir, "groups_comb")
    os.makedirs(subdir, exist_ok=True)
    plot_scaffold_with_all_sectors(
        scaffold, multisector_group_idxs, gdir, 
        group_colors=group_colors, 
        ref_scaffold=ref_scaffold, 
        outdir=subdir, 
        verbosity=verbosity,
    )


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
