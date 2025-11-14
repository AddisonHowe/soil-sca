#!/usr/bin/env bash


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  V1

datdir=out/K00370/TIGR01580_with_soil_seqs_1000
outdir=out/data_transfers/v1_TIGR01580_with_soil_seqs_1000

python scripts/filter_soil_variants_in_sca_results.py \
    -d ${datdir} -o ${outdir} \
    -sf data/K00370/misc/seqids_57.txt \
    -si 3 5

cp -r ${datdir}/groups ${outdir}
cp -r ${datdir}/sca_results/msa_sectors ${outdir}
cp ${datdir}/sca_results/conservation.npy ${outdir}
cp ${datdir}/sca_results/msa.npy ${outdir}
gzip ${outdir}/msa.npy


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  V2

datdir=out/K00370/TIGR01580_with_soil_seqs_57
outdir=out/data_transfers/v2_TIGR01580_with_soil_seqs_57

python scripts/filter_soil_variants_in_sca_results.py \
    -d ${datdir} -o ${outdir} \
    -sf data/K00370/misc/seqids_57.txt \
    -si 3 5

cp -r ${datdir}/groups ${outdir}/groups
cp -r ${datdir}/sca_results/msa_sectors ${outdir}/msa_sectors
cp ${datdir}/sca_results/conservation.npy ${outdir}
cp ${datdir}/sca_results/msa.npy ${outdir}
gzip ${outdir}/msa.npy


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  V3

datdir=out/K00370/TIGR01580_noX_with_soil_seqs_57
outdir=out/data_transfers/v3_TIGR01580_noX_with_soil_seqs_57

python scripts/filter_soil_variants_in_sca_results.py \
    -d ${datdir} -o ${outdir} \
    -sf data/K00370/misc/seqids_57.txt \
    -si 3 6

cp -r ${datdir}/groups ${outdir}/groups
cp -r ${datdir}/sca_results/msa_sectors ${outdir}/msa_sectors
cp ${datdir}/sca_results/conservation.npy ${outdir}
cp ${datdir}/sca_results/msa.npy ${outdir}
gzip ${outdir}/msa.npy
