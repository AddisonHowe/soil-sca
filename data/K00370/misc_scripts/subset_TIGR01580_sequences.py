"""Subset TIGR01580 fasta file, weighting by phylogenic rank.

"""

import os, sys
import argparse
import numpy as np
import pandas as pd

from Bio import SeqIO

RANKS = ["kingdom", "phylum", "class", "order", "family", "genus"]

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infile", type=str, required=True)
    parser.add_argument("-n", "--nsamp_max", type=int, required=True)
    parser.add_argument("-o", "--outfile", type=str, required=True)
    parser.add_argument("-sm", "--seqs_metafile", type=str, required=True)
    parser.add_argument("-tm", "--taxa_metafile", type=str, required=True)
    parser.add_argument("-r", "--rank", type=str, required=True, 
                        choices=RANKS)
    parser.add_argument("-ro", "--rankoutdir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(args)


def main(args):
    fasta_fpath = args.infile
    n_max = args.nsamp_max
    out_fpath = args.outfile
    seqs_metadata_fpath = args.seqs_metafile
    taxa_metadata_fpath = args.taxa_metafile
    rank = args.rank
    rankoutdir = args.rankoutdir
    seed = args.seed

    # Housekeeping
    if seed == 0:
        seed = np.random.randing(2**32)
    print(f"Seed: {seed}")
    rng = np.random.default_rng(seed=seed)

    # Load the input fasta file
    records_in = list(SeqIO.parse(fasta_fpath, "fasta"))
    fastaseqs = {str(e.id): e  for e in records_in}
    print(f"Loaded {len(records_in)} sequences from input file {fasta_fpath}")

    # Load the sequence metadata information
    df_seq_metadata = pd.read_csv(
        seqs_metadata_fpath, sep="\t", header=None, 
        names=["id", "status", "prot_name", "taxidstr"]
    )
    df_seq_metadata["taxid"] = df_seq_metadata["taxidstr"].str.removeprefix("taxID:").astype(int)
    df_seq_metadata["prot_name"] = df_seq_metadata["prot_name"].str.lower()
    df_seq_metadata["seqid"] = df_seq_metadata["id"].map({
        s.split("|")[0]: s for s in fastaseqs
    })
    print(f"Loaded sequence metadata information from {seqs_metadata_fpath}")
    print("df_seq_metadata shape:", df_seq_metadata.shape)

    # Load the taxonomic metadata information
    df_taxa_metadata = pd.read_csv(
        taxa_metadata_fpath, sep="\t", 
    )
    df_taxa_metadata = df_taxa_metadata.drop_duplicates()
    print(f"Loaded taxonomic metadata information from {taxa_metadata_fpath}")
    print("df_taxa_metadata shape:", df_taxa_metadata.shape)

    # Merge metadata
    df_meta = df_seq_metadata.merge(df_taxa_metadata, on="taxid", how="left")
    print("Merged df_seq_metadata and df_taxa_metadata.")
    print("df_meta shape:", df_meta.shape)

    # Count number of taxa represented in fasta file, by given taxonomic rank
    rankvals = df_meta[rank]
    valcounts = rankvals.value_counts()
    print(f"Value counts of rank {rank}")
    print(valcounts)

    if rankoutdir:
        print(f"Saving rank counts in {rankoutdir}")
        os.makedirs(rankoutdir, exist_ok=True)
        valcounts.to_csv(f"{rankoutdir}/rankcounts_{rank}.tsv", sep="\t")

    # Sample records
    records_out = sample_records(
        fastaseqs, df_meta, rank, valcounts,
        n_max, rng
    )

    # Write output
    SeqIO.write(records_out, out_fpath, "fasta")
    print(f"Saved {len(records_out)} of {len(records_in)} sequences to {out_fpath}")
    print("Done!")


def sample_records(id2entry, df_seqs_metadata, rank, value_counts, n_max, rng):
    sample = []
    for rval in value_counts.index:
        count = value_counts[rval]
        print(rval, count)
        seqid_subset = df_seqs_metadata[df_seqs_metadata[rank] == rval]["seqid"]
        if count > n_max:
            # Sample
            idxs = rng.choice(count, n_max, replace=False)
            seqid_subset = seqid_subset.iloc[idxs]
        sample += [id2entry[seqid] for seqid in seqid_subset]
    return sample

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
