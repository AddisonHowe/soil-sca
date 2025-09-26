# Data: K00370 (narG)

## Raw sequences

* `K00370_rep.faa`: Representative sequences of KO entry K00370, from the KEGG database. Copied from the `soil-ko-wrangling` pipeline.
* `TIGR01580.fasta`: Complete set of 9557 sequences acquired from [interpro]() that are part of the TIGR01580 accession. # TODO: link?
* `soil_seqs_800.fasta`: Soil ORFs that contain a start and stop codon, and are at least 800 amino acids in length. Copied from the `soil-metagenomics` repository.
* `soil_seqs_1000.fasta`: Soil ORFs that contain a start and stop codon, and are at least 1000 amino acids in length. Copied from the `soil-metagenomics` repository.
* `soil_seqs_800_with_reference.fasta`: As above, but also containing reference sequence 1Q16 acquired online.  # TODO: where?
* `soil_seqs_1000_with_reference.fasta`: As above, but also containing reference sequence 1Q16 acquired online.  # TODO: where?
* `TIGR01580subset1000_with_soil_seqs_1000.fasta`: Union of `soil_seqs_1000.fasta` and the first 1000 sequences of `TIGR01580.fasta`.

## MSAs

MSAs are produced using [clustal-omega](https://www.ebi.ac.uk/jdispatcher/msa/clustalo?stype=protein&outfmt=fa) with fasta output format and are named in accordance with the input fasta file.

## Structures
