#!/usr/bin/env bash

datdir=data/K00370
outdir=out_input/sca_by_ko/K00370

# scaffold=$1
# scaffold=Soil14.scaffold_576820813_c1_40

for f in ${datdir}/structures/Soil*.pdb; do
    s=$(basename $f)
    s=${s/.pdb/}
    echo $s
    python scripts/pymol_sca.py \
        -s $s \
        --pdb_dir ${datdir}/structures \
        --groups_dir ${outdir}/sca_groups \
        --outdir ${outdir}/images \
        --groups 0 1 2 3

done

