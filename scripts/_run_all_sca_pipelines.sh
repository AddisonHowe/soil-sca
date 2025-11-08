#!/usr/bin/env bash

sh scripts/run_sca/run_sca_pipeline_K00370_MSA_TIGR01580_noX_weighted_subset_phylum_nmax100_v1_with_soil_seqs_57.sh \
    --no-runpymol -n 0 --load existing
sh scripts/run_sca/run_sca_pipeline_K00370_MSA_TIGR01580_noX_with_soil_seqs_57_p98.sh \
    --no-runpymol -n 0 --load existing
sh scripts/run_sca/run_sca_pipeline_K00370_MSA_TIGR01580_noX_with_soil_seqs_57.sh \
    --no-runpymol -n 0 --load existing
sh scripts/run_sca/run_sca_pipeline_K00370_MSA_TIGR01580_weighted_subset_phylum_nmax100_v1_with_soil_seqs_57.sh \
    --no-runpymol -n 0 --load existing
sh scripts/run_sca/run_sca_pipeline_K00370_MSA_TIGR01580_weighted_subset_phylum_nmax100_v1_with_soil_seqs_1000.sh \
    --no-runpymol -n 0 --load existing
sh scripts/run_sca/run_sca_pipeline_K00370_MSA_TIGR01580_with_soil_seqs_57.sh \
    --no-runpymol -n 0 --load existing
sh scripts/run_sca/run_sca_pipeline_K00370_MSA_TIGR01580_with_soil_seqs_1000_p98.sh \
    --no-runpymol -n 0 --load existing
sh scripts/run_sca/run_sca_pipeline_K00370_MSA_TIGR01580_with_soil_seqs_1000.sh \
    --no-runpymol -n 0 --load existing
