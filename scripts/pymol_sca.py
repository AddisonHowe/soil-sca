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


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scaffold", type=str, required=True)
    parser.add_argument("--pdb_dir", type=str, required=True)
    parser.add_argument("--groups_dir", type=str, required=True)
    parser.add_argument("--groups", type=int, nargs='*',
                        help="Group indices, (starting at 1) that correspond " \
                        "to subdirectories group_<idx> of groups_dir. If -1, " \
                        "produce plots for all groups.")
    parser.add_argument("-r", "--reference", type=str, default=None,
                        help="If specified, align the input scaffold to this " \
                        "reference.")
    parser.add_argument("-o", "--outdir", type=str, default=None)
    return parser.parse_args(args)


def plot_scaffold_with_all_groups(
        scaffold, groups_basedir, group_idxs, ref_scaffold, outdir, 
):
    gdir = f"{groups_basedir}"

    if outdir:
        os.makedirs(outdir, exist_ok=True)
    
    struct_color = "gray70"
    struct_style = "sticks"
    group_colors = ["red", "orange", "yellow", "green", "blue", "purple"]
    group_styles = ["spheres"] * len(group_idxs)

    if ref_scaffold:
        cmd.hide("everything", "ref_struct")

    cmd.hide("everything", "struct")
    cmd.show(struct_style, "struct")
    cmd.color(struct_color, "struct")
    cmd.bg_color("white")

    if ref_scaffold:
        cmd.align("struct", "ref_struct")

    topk = 5
    for i, gidx in enumerate(group_idxs[0:min(topk, len(group_idxs))]):
        group_fpath = f"{gdir}/group_{gidx}/group_{gidx}_{scaffold}.npy"
        if os.path.isfile(group_fpath):
            group_selection = f"group_selection{i}"
            group = np.load(group_fpath)
            res_idxs = 1 + group
            selection_string = "resi " + "+".join(map(str, res_idxs))
            cmd.select(group_selection, selection_string)
            cmd.show(group_styles[i], group_selection)
            cmd.color(group_colors[i], group_selection)
        else:
            group_selection = None
            group = None
            print(f"Group {gidx} file not found: {group_fpath}")

    for ri in range(4):
        cmd.rotate("y", 90 * ri, "struct")
        cmd.png(f"{outdir}/{scaffold}_all_groups_view{ri}.png", dpi=300)

    return

def main(args):
    scaffold = args.scaffold
    pdb_dir = args.pdb_dir
    groups_basedir = args.groups_dir
    group_idxs = args.groups
    ref_scaffold = args.reference
    outdir = args.outdir

    if ref_scaffold is None or ref_scaffold.lower() == "none":
        ref_scaffold = None

    gdir = f"{groups_basedir}"
    
    if len(group_idxs) == 1 and group_idxs[0] == -1:
        # Check group directory and include all groups present
        group_files = os.listdir(gdir)
        prefix = "group_"
        group_idxs = [
            f.removeprefix(prefix) for f in group_files if f.startswith(prefix)
        ]
        group_idxs = [int(x) for x in group_idxs]

    if outdir:
        os.makedirs(outdir, exist_ok=True)
    
    print("Scaffold:", scaffold)
    pdbfile = f"{pdb_dir}/{scaffold}.pdb"
    cmd.load(pdbfile, "struct")

    if ref_scaffold:
        reffile = f"{pdb_dir}/{ref_scaffold}.pdb"
        cmd.load(reffile, "ref_struct")
    
    struct_color = "gray70"
    struct_style = "sticks"
    group_color = "red"
    group_style = "spheres"

    if ref_scaffold:
        cmd.hide("everything", "ref_struct")

    cmd.hide("everything", "struct")
    cmd.show(struct_style, "struct")
    cmd.color(struct_color, "struct")
    cmd.bg_color("white")

    for gidx in group_idxs:
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
            print(f"Group {gidx} file not found: {group_fpath}")

        if ref_scaffold:
            cmd.align("struct", "ref_struct")
        
        cmd.png(f"{outdir}/{scaffold}_group{gidx}.png", dpi=300)

        # for ri in range(4):
        #     cmd.rotate("y", 90 * ri, "struct")
        #     cmd.png(f"{outdir}/{scaffold}_group{gidx}_view{ri}.png", dpi=300)
        
        # reset
        if group_selection:
            cmd.hide(group_style, group_selection)
            cmd.color(struct_color, group_selection)  # reset color
            cmd.delete(group_selection)
    
    subdir = os.path.join(outdir, "groups_comb")
    os.makedirs(subdir, exist_ok=True)
    plot_scaffold_with_all_groups(
        scaffold, groups_basedir, group_idxs, ref_scaffold, subdir, 
    )


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
