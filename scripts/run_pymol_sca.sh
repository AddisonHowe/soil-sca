#!/usr/bin/env bash

datdir=data/K00370
outdir=out/K00370/MSA_800
reference="1Q16"


for f in ${datdir}/structures/*.pdb; do
    s=$(basename $f)
    s=${s/.pdb/}
    echo $s
    python scripts/pymol_sca.py \
        -s ${s} \
        -r ${reference} \
        --pdb_dir ${datdir}/structures \
        --groups_dir ${outdir}/sca_groups \
        --outdir ${outdir}/pymol_images \
        --groups 0 1
done
