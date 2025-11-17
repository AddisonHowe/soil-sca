"""Script to partition a given MSA

Given an MSA X and set of indices (idx0, idx1, ..., idxN), 
with 0 < idx[i] < len(MSA), partition the MSA into segments.

"""

import os, sys
import numpy as np
import argparse
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--idxs", type=int, nargs="+", required=True)
    parser.add_argument("-m", "--msa_fpath", type=str, required=True)
    parser.add_argument("-v", "--verbosity", type=int, default=1)
    parser.add_argument("-o", "--outdir", type=str, required=True)
    parser.add_argument("-n", "--outfame", type=str, default="segmented_msa")
    return parser.parse_args(args)


def main(args):
    idxs = args.idxs
    msa_fpath = args.msa_fpath
    verbosity = args.verbosity
    outdir = args.outdir
    outfame = args.outfame

    print(f"Processing MSA at {msa_fpath}")
    print(f"Segmenting based on indices {idxs}")

    msa_records = list(SeqIO.parse(msa_fpath, "fasta"))
    idxs.append(len(msa_records[0].seq))
    print(idxs)
    nsegments = len(idxs)
    segment_records = [[] for _ in range(nsegments)]
    segment_origpositions = [{} for _ in range(nsegments)]
    for record in msa_records:
        seq = str(record.seq)
        id = str(record.id)
        desc = str(record.description)

        idx0 = 0
        seqpos = 0
        for j, idx in enumerate(idxs):
            subseq = seq[idx0:idx]
            subseq_nogap = subseq.replace("-", "")
            new_record = SeqRecord(Seq(subseq), id=id, description=desc)
            segment_records[j].append(new_record)

            segment_origpositions[j][id] = seqpos + np.arange(0, len(subseq_nogap))
            seqpos += len(subseq_nogap)
            idx0 = idx

    os.makedirs(outdir, exist_ok=True)
    print(f"Saving output to {outdir}")
    idx0 = 0
    for j, records in enumerate(segment_records):
        idx1 = idxs[j]
        print(f"Saving segmentation {idx0}-{idx1}")
        SeqIO.write(records, f"{outdir}/{outfame}_{j}_{idx0}_{idx1}.aln-fasta", "fasta")
        idxs_in_msa_orig = np.arange(idx0, idx1)
        np.save(
            f"{outdir}/{outfame}_{j}_{idx0}_{idx1}_idxs_in_msa_orig.npy",
            idxs_in_msa_orig
        )
        np.savez(
            f"{outdir}/{outfame}_{j}_{idx0}_{idx1}.origpositions.npz",
            **segment_origpositions[j]
        )
        idx0 = idx1


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
