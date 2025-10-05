# Data: K00370 (narG)

## Raw sequences

* `K00370_rep.faa`: Representative sequences of KO entry K00370, from the KEGG database. Copied from the `soil-ko-wrangling` pipeline.
* `TIGR01580.fasta`: Complete set of 9557 sequences acquired from [interpro](https://www.ebi.ac.uk/interpro/entry/ncbifam/TIGR01580/protein/UniProt/#table) that are part of the TIGR01580 accession.
* `soil_seqs_800.fasta`: Soil ORFs that contain a start and stop codon, and are at least 800 amino acids in length. Copied from the `soil-metagenomics` repository.
* `soil_seqs_1000.fasta`: Soil ORFs that contain a start and stop codon, and are at least 1000 amino acids in length. Copied from the `soil-metagenomics` repository.
* `soil_seqs_800_with_reference.fasta`: As above, but also containing reference sequence 1Q16 acquired online.  # TODO: where?
* `soil_seqs_1000_with_reference.fasta`: As above, but also containing reference sequence 1Q16 acquired online.  # TODO: where?
* `TIGR01580subset1000_with_soil_seqs_1000.fasta`: Union of `soil_seqs_1000.fasta` and the first 1000 sequences of `TIGR01580.fasta`.
* `TIGR01580_weighted_subset_phylum_nmax100_v1.fasta`: A random subset of the `TIGR01580.fasta` file, generated with the script `data/K00370/misc_scripts/run_subset_TIGR01580_sequences.sh`. It takes at most `nmax=100` sequences from each phylum in the original file.

## MSAs

MSAs are produced using [clustal-omega](https://www.ebi.ac.uk/jdispatcher/msa/clustalo?stype=protein&outfmt=fa) with fasta output format and are named in accordance with the input fasta file.

## Structures


## Misc.

The metadata files `misc/metadata_TIGR01580.tsv` and `misc/taxids_TIGR01580.txt` are generated with the following script:

```bash
sh data/K00370/misc_scripts/gen_metadata.sh
```

Then taxonomic information for all taxa in the TIGR01580 accession can be compiled through the script:

```bash
python data/K00370/misc_scripts/fetch_taxid_info.py -i data/K00370/misc/taxids_TIGR01580.txt -o data/K00370/misc/taxids_TIGR01580_metadata.tsv
```

The `rankcounts` subdirectory contains output files resulting from the script `data/K00370/misc_scripts/run_subset_TIGR01580_sequences.sh`. It contains a file for each taxonomic rank, specifying the number of sequences associated to each genus, family, etc..

* `misc/assignments_K00370_v2.tsv`: From the `soil-metagenomics` project. Assignment of each sequence to a "variant group." (Total of 4 groups used.)
* `misc/assignments_K00370_v3.tsv`: From the `soil-metagenomics` project. Assignment of each sequence to a "variant group." (Total of 8 groups used.)
* `misc/nar_4groups.tsv`: From the `soil-metagenomics` project (edited to contain sequence ID in the first column). Assignment of sequences to a rate parameter $r_A$. (Total of 4 groups used.)
* `misc/nar_8groups.tsv`: From the `soil-metagenomics` project (edited to contain sequence ID in the first column). Assignment of sequences to a rate parameter $r_A$. (Total of 8 groups used.)

