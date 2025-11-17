"""Based on an MSA, identify a relatively conserved position near the center,
and use this position to transform every original sequence in the MSA, so that
each altered sequence consists of the second half, after the identified 
position, followed by the first half.

"""

import os, sys
import argparse
import numpy as np
from Bio import AlignIO, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord



def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--msa_fpath", type=str, required=True)
    parser.add_argument("-r", "--scadir", type=str, required=True)
    parser.add_argument("-o", "--outdir", type=str, required=True)
    parser.add_argument("-n", "--outfname", type=str, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--percentile", type=float, default=None)
    return parser.parse_args(args)


def main(args):
    msa_fpath = args.msa_fpath
    scadir = args.scadir
    outdir = args.outdir
    outfname = args.outfname
    start = args.start
    end = args.end
    percentile = args.percentile

    outfpath = os.path.join(outdir, outfname)

    msa = AlignIO.read(msa_fpath, "fasta")
    nseq = len(msa)
    npos_msa_orig = msa.get_alignment_length()
    print(f"Loaded original MSA of shape ({nseq, npos_msa_orig})")
    Di = np.load(os.path.join(scadir, "conservation.npy"))
    retained_positions = np.load(os.path.join(scadir, "retained_positions.npy"))
    npos_retained = len(retained_positions)
    print("Number of retained positions:", npos_retained)
    
    if percentile is not None:
        assert 0 < percentile and percentile <= 100
        half_pos = int(npos_retained // 2)
        delta = int(npos_retained * (percentile / 100) / 2)
        assert delta > 0
        start = half_pos - delta
        end = half_pos + delta
    else:
        if end == -1:
            end = npos_retained
        if start < 0 or start > len(npos_retained):
            msg = f"start must between 0 and {len(npos_retained)} (inclusive)"
            raise IndexError(msg)
        if end < 0 or end > len(npos_retained):
            msg = f"end must between 0 and {len(npos_retained)} (inclusive)"
            raise IndexError(msg)

    print(f"Identifying split index within retained positions: [{start}, {end}]")
    split_idx = start + np.argmax(Di[start:end])
    print("Split index:", split_idx)
    
    # Need to determine which position the split index corresponds to in the 
    # original MSA.
    split_idx_msa_orig = retained_positions[split_idx]
    print(f"Split index {split_idx} -> index {split_idx_msa_orig} in original MSA")
    records = []
    new_orders = {}
    for i, entry in enumerate(msa):
        seq = str(entry.seq)
        seq0 = seq[0:split_idx_msa_orig].replace("-", "")
        seq1 = seq[split_idx_msa_orig:].replace("-", "")
        altseq =  seq1 + seq0
        new_entry = SeqRecord(
            Seq(altseq),
            id=entry.id,
            name=entry.name,
            description=entry.description,
        )
        records.append(new_entry)
        new_orders[entry.id] = np.concatenate([
            len(seq0) + np.arange(len(seq1)), np.arange(len(seq0))
        ])

    print(f"Writing results to {outfpath}")
    SeqIO.write(records, outfpath, "fasta")
    np.savez(f"{outdir}/{outfname.replace(".fasta", "")}.origpositions.npz", **new_orders)
    print("Done!")


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
