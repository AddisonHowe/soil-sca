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

MO_COLOR = None
SF4_COLOR = None

DEFAULT_STRUCT_COLOR = "gray70"
DEFAULT_STRUCT_STYLE = "sticks"
DEFAULT_STRUCT_ALPHA = 0.5
DEFAULT_SECTOR_COLORS = SECTOR_COLORS
DEFAULT_SECTOR_STYLE = "spheres"

DEFAULT_BG_COLOR = "white"

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scaffold", type=str, required=True)
    parser.add_argument("--pdb_dir", type=str, required=True)
    parser.add_argument("--groups_dir", type=str, required=True)
    parser.add_argument("--groups", type=int, nargs='*', default=-1,
                        help="Group indices, (starting at 0) that correspond " \
                        "to subdirectories group_<idx> of groups_dir. If -1, " \
                        "produce plots for all groups.")
    parser.add_argument("--multisector", action="store_true", 
                        help="Plot sectors simultaneously on the same protein.")
    parser.add_argument("-r", "--reference", type=str, default=None,
                        help="If specified, align the input scaffold to this " \
                        "reference.")
    parser.add_argument("--scores_dir", type=str, default=None)
    parser.add_argument("--views", action="store_true")
    parser.add_argument("--show_molybdenum", action="store_true")
    parser.add_argument("-o", "--outdir", type=str, default=None)
    parser.add_argument("-v", "--verbosity", type=int, default=1)

    return parser.parse_args(args)


def main(args):
    scaffold = args.scaffold
    pdb_dir = args.pdb_dir
    groups_basedir = args.groups_dir
    group_idxs = args.groups
    multisector = args.multisector
    ref_scaffold = args.reference
    scores_dir = args.scores_dir
    show_molybdenum = args.show_molybdenum
    outdir = args.outdir
    verbosity = args.verbosity
    views = args.views

    if ref_scaffold is None or ref_scaffold.lower() == "none":
        ref_scaffold = None
    
    # Get all sectors/groups if group_idxs is specified as -1 at the cmd line
    # Check group directory and include all groups present
    group_files = os.listdir(groups_basedir)
    prefix = "group_"
    found_group_idxs = list(np.sort([
        int(f.removeprefix(prefix)) for f in group_files if f.startswith(prefix)
    ]))

    if group_idxs[0] == -1:
        group_idxs = found_group_idxs

    if scores_dir is None or scores_dir.lower() == "none":
        scores_dir = None
    elif os.path.isdir(scores_dir):
        if verbosity:
            print(f"Scores directory specified: {scores_dir}")
    else:
        raise RuntimeError(f"Scores directory {scores_dir} does not exist!")

    if outdir:
        os.makedirs(outdir, exist_ok=True)
    
    # Colors and styles
    struct_color = DEFAULT_STRUCT_COLOR
    struct_style = DEFAULT_STRUCT_STYLE
    struct_alpha = DEFAULT_STRUCT_ALPHA
    sector_colors = [_hex2color(x) for x in DEFAULT_SECTOR_COLORS]
    sector_style = DEFAULT_SECTOR_STYLE
    background_color = DEFAULT_BG_COLOR

    # Load the specified structure
    if verbosity:
        print("Scaffold:", scaffold)
    pdbfile = f"{pdb_dir}/{scaffold}.pdb"
    cmd.load(pdbfile, "struct")

    # Load a reference, if given
    if ref_scaffold:
        reffile = f"{pdb_dir}/{ref_scaffold}.pdb"
        cmd.load(reffile, "ref_struct")

    # Hide the loaded structure (and reference)
    cmd.hide("everything", "struct")
    if ref_scaffold:
        cmd.hide("everything", "ref_struct")
    
    # Set background color
    cmd.bg_color(background_color)

    # Show structure
    cmd.show(struct_style, "struct")
    cmd.color(struct_color, "struct")
    cmd.set(
        {"sticks": "stick_transparency"}.get(struct_style, DEFAULT_STRUCT_STYLE), 
        1 - struct_alpha, 
        "struct"
    )

    if multisector:
        if verbosity:
            print(f"Plotting {scaffold} with all sectors...")
        
        plot_scaffold_with_multiple_sectors(
            scaffold, group_idxs, groups_basedir, 
            struct_color=struct_color, 
            sector_colors=sector_colors, 
            sector_style=sector_style, 
            ref_scaffold=ref_scaffold, 
            scores_dir=scores_dir,
            show_molybdenum=show_molybdenum,
            views=views,
            outdir=outdir, 
            verbosity=verbosity,
        )
    else:
        if verbosity:
            print(f"Plotting {scaffold} by sector...")
        
        plot_scaffold_by_sectors(
            scaffold, group_idxs, groups_basedir,
            struct_color=struct_color, 
            sector_colors=sector_colors, 
            sector_style=sector_style, 
            ref_scaffold=ref_scaffold, 
            scores_dir=scores_dir,
            show_molybdenum=show_molybdenum,
            views=views,
            outdir=outdir, 
            verbosity=verbosity,
        )
    
    if verbosity:
        print("Done!")


def plot_scaffold_by_sectors(
        scaffold, group_idxs, groups_basedir, *,
        struct_color, 
        sector_colors, 
        sector_style, 
        ref_scaffold, 
        outdir, 
        verbosity=1,
        scores_dir=None,
        show_molybdenum=False,
        views=True,
):
    gdir = f"{groups_basedir}"
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    for gidx in group_idxs:
        sector_color = sector_colors[gidx]
        group_fpath = f"{gdir}/group_{gidx}/group_{gidx}_{scaffold}.npy"
        if os.path.isfile(group_fpath):
            group_selection = "group_selection"
            group = np.load(group_fpath)
            res_idxs = 1 + group
            selection_string = "resi " + "+".join(map(str, res_idxs))
            cmd.select(group_selection, selection_string)
            cmd.show(sector_style, group_selection)
            cmd.color(sector_color, group_selection)
        else:
            group_selection = None
            group = None
            if verbosity:
                print(f"Structure {scaffold} group {gidx} file not found: {group_fpath}")
        
        # Load scores, if a directory is specified
        scores = None
        alphas = None
        if scores_dir:
            sdir = f"{scores_dir}/sector_{gidx}/"
            scores_fpath = f"{sdir}/sector_{gidx}_scores_{scaffold}.npy"
            if os.path.isfile(scores_fpath):
                MIN_ALPHA = 0.5
                scores = np.load(scores_fpath)
                svals = np.square(scores)
                s0, s1 = svals.min(), svals.max()
                a0, a1 = MIN_ALPHA, 1
                alphas = (a1 - a0) / (s1 - s0) * (svals - s1) + a1
        if alphas is not None:
            # Apply transparency per residue
            if verbosity > 1:
                print(f"Applying alphas [{alphas.min():.4g}, {alphas.max():.4g}]")
            for resi, alpha in zip(res_idxs, alphas):
                cmd.set(
                    {"spheres": "sphere_transparency"}.get(
                        sector_style, DEFAULT_SECTOR_STYLE
                    ), 
                    1 - alpha, 
                    f"{group_selection} and resi {resi}"
                )

        # Align the structure to the reference
        if ref_scaffold:
            cmd.align("struct", "ref_struct")
        
        # Show extra features
        if show_molybdenum:
            _show_molybdenum("ref_struct", color=MO_COLOR)
            _show_sf4("ref_struct", color=SF4_COLOR)
        
        # Save the primary plot
        cmd.png(f"{outdir}/{scaffold}_group{gidx}.png", dpi=300)

        # Save views of each side
        viewsdir = os.path.join(outdir, "views")
        if views:
            os.makedirs(viewsdir, exist_ok=True)
            for ri in range(4):
                cmd.png(f"{viewsdir}/{scaffold}_group{gidx}_view{ri}.png", dpi=300)
                cmd.rotate("y", 90, "struct")
                if ref_scaffold:
                    cmd.rotate("y", 90, "ref_struct")
        
        # Reset
        if group_selection:
            cmd.hide(sector_style, group_selection)
            cmd.color(struct_color, group_selection)  # reset color
            cmd.delete(group_selection)

    return


def plot_scaffold_with_multiple_sectors(
        scaffold, group_idxs, groups_basedir, *, 
        struct_color, 
        sector_colors, 
        sector_style, 
        ref_scaffold, 
        outdir, 
        verbosity=1, 
        scores_dir=None,
        show_molybdenum=False,
        views=True,
):
    gdir = f"{groups_basedir}"
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    
    sector_styles = [sector_style] * len(group_idxs)

    for i, gidx in enumerate(group_idxs):
        group_fpath = f"{gdir}/group_{gidx}/group_{gidx}_{scaffold}.npy"
        sector_color = sector_colors[gidx]
        if os.path.isfile(group_fpath):
            group_selection = f"group_selection{i}"
            group = np.load(group_fpath)
            res_idxs = 1 + group
            selection_string = "resi " + "+".join(map(str, res_idxs))
            cmd.select(group_selection, selection_string)
            cmd.show(sector_styles[i], group_selection)
            cmd.color(sector_color, group_selection)
        else:
            group_selection = None
            group = None
            if verbosity:
                print(f"Group {gidx} file not found: {group_fpath}")

        # Load scores, if a directory is specified
        scores = None
        alphas = None
        if scores_dir:
            sdir = f"{scores_dir}/sector_{gidx}/"
            scores_fpath = f"{sdir}/sector_{gidx}_scores_{scaffold}.npy"
            if os.path.isfile(scores_fpath):
                MIN_ALPHA = 0.5
                scores = np.load(scores_fpath)
                svals = np.square(scores)
                s0, s1 = svals.min(), svals.max()
                a0, a1 = MIN_ALPHA, 1
                alphas = (a1 - a0) / (s1 - s0) * (svals - s1) + a1
        if alphas is not None:
            # Apply transparency per residue
            if verbosity > 1:
                print(f"Applying alphas [{alphas.min():.4g}, {alphas.max():.4g}]")
            for resi, alpha in zip(res_idxs, alphas):
                cmd.set(
                    {"spheres": "sphere_transparency"}.get(
                        sector_style, DEFAULT_SECTOR_STYLE
                    ), 
                    1 - alpha, 
                    f"{group_selection} and resi {resi}"
                )
    
    # Align the structure to the reference
    if ref_scaffold:
        cmd.align("struct", "ref_struct")

    # Show extra features
    if show_molybdenum:
        _show_molybdenum("ref_struct", color=MO_COLOR)
        _show_sf4("ref_struct", color=SF4_COLOR)

    # Save the primary plot
        cmd.png(f"{outdir}/{scaffold}_groups_{",".join(
                [str(i) for i in group_idxs])}.png", dpi=300)
    
    # Save views of each side
    viewsdir = os.path.join(outdir, "views")
    if views:
        os.makedirs(viewsdir, exist_ok=True)
        for ri in range(4):
            cmd.png(f"{viewsdir}/{scaffold}_groups_{",".join(
                [str(i) for i in group_idxs])}_view{ri}.png", dpi=300)
            cmd.rotate("y", 90, "struct")
            if ref_scaffold:
                    cmd.rotate("y", 90, "ref_struct")
    
    return


def _hex2color(x):
    return "0x" + x[1:]


def _show_molybdenum(
        struct,
        color=None
):
    cmd.select("mo", f"{struct}/F/A/6MO`1302/MO")
    cmd.show("everything", "mo")
    if isinstance(color, str):
        cmd.color(color, "mo")
    return


def _show_sf4(
        struct,
        color=None,
):
    cmd.select("sf4", f"{struct}/G/A/SF4`1401/*")
    cmd.show("everything", "sf4")
    if isinstance(color, str):
        cmd.color(color, "sf4")
    return
    

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
