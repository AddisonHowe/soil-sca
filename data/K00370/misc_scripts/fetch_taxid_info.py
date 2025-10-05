"""Fetch information about a given taxonomid ID

"""

import os, sys
import csv
import argparse
import numpy as np
import pandas as pd
import tqdm as tqdm
from Bio import Entrez


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infile", type=str, required=True,
                        help="Path to file with each line a taxid")
    parser.add_argument("-o", "--outfile", type=str, required=True,
                        help="Output file to generate or overwrite")
    return parser.parse_args(args)


def main(args):
    taxa_list_fpath = args.infile
    outfpath = args.outfile
    
    print(f"Reading input file {taxa_list_fpath}")
    with open(taxa_list_fpath, "r") as f:
        csvreader = csv.reader(f, delimiter=",")
        taxid_list = []
        for row in csvreader:  # process each row
            taxid = row[0]
            taxid_list.append(taxid)

    print(f"Read {len(taxid_list)} taxids from input file")

    print("Fetching records...")
    fetched_records = fetch_records(taxid_list)
    print("Fetch complete.")
    df = build_df_from_records(fetched_records)
    df.to_csv(outfpath, sep="\t", index=None)
    print(f"Saved output to {outfpath}")
    print("Done!")


def fetch_records(taxids):
    Entrez.email = "your_email@example.com"
    handle = Entrez.efetch(db="taxonomy", id=taxids, retmode="xml")
    records = Entrez.read(handle)
    return records


def build_df_from_records(records):
    results = []
    for record in records:
        ranks = {e["Rank"]: e["ScientificName"] for e in record["LineageEx"]}
        ranks[record["Rank"]] = record["ScientificName"]  # include the queried taxon

        row = {
            "taxid": record["TaxId"],
            "scientific_name": record["ScientificName"],
            "rank": record["Rank"],
        }

        # Standard taxonomy ranks in biological order
        for rank in [
            "superkingdom", "kingdom", "phylum", "class", "order",
            "family", "genus", "species"
        ]:
            row[rank] = ranks.get(rank, "")
        row["full_lineage"] = record["Lineage"]
        results.append(row)
    return pd.DataFrame(results)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
