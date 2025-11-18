#!/usr/bin/env bash

runarg="1 2 3 4 pymolall allprojections pymolallanimate"

sh figure_scripts/runscripts/run_gen_all_figs_K00370_with_soil_seqs_57.sh "${runarg}"
sh figure_scripts/runscripts/run_gen_all_figs_K00370_noX_with_soil_seqs_57.sh "${runarg}"
sh figure_scripts/runscripts/run_gen_all_figs_K00370_noX_with_soil_seqs_57_isolates.sh "${runarg}"

sh figure_scripts/runscripts/run_gen_all_figs_K00370_noX_weighted_subset_phylum_nmax100_v1_with_soil_seqs_57.sh "${runarg}"

sh figure_scripts/runscripts/run_gen_all_figs_K00370_noX_minlength1100_maxlength1350_with_soil_seqs_57.sh "${runarg}"
sh figure_scripts/runscripts/run_gen_all_figs_K00370_noX_minlength1100_maxlength1350_with_soil_seqs_57_isolates_seg_0_0_750.sh "${runarg}"
sh figure_scripts/runscripts/run_gen_all_figs_K00370_noX_minlength1100_maxlength1350_with_soil_seqs_57_isolates_seg_1_750_1500.sh "${runarg}"
sh figure_scripts/runscripts/run_gen_all_figs_K00370_noX_minlength1100_maxlength1350_with_soil_seqs_57_isolates_seg_2_1500_2279.sh "${runarg}"

# sh figure_scripts/runscripts/run_gen_all_figs_K00370_with_soil_seqs_1000.sh "${runarg}"

# sh figure_scripts/runscripts/run_gen_all_figs_K00370_cutswap_with_soil_seqs_57.sh "${runarg}"
