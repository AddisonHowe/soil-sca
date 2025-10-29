#!/usr/bin/env python3
from Bio import SeqIO
import sys

def remove_sequences_with_X(input_fasta, output_fasta):
    """Read a FASTA file and write only sequences that do NOT contain 'X'."""
    with open(input_fasta) as infile, open(output_fasta, "w") as outfile:
        kept_records = (
            record for record in SeqIO.parse(infile, "fasta")
            if "X" not in str(record.seq)
        )
        count = SeqIO.write(kept_records, outfile, "fasta")

    print(f"Wrote {count} sequences (removed those containing 'X').")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python filter_noX_fasta.py input.fasta output.fasta")
        sys.exit(1)

    input_fasta = sys.argv[1]
    output_fasta = sys.argv[2]

    remove_sequences_with_X(input_fasta, output_fasta)
