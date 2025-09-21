# soil-sca

## About

## Setup

Create a project environment as follows:

```bash
mamba env create -p ./env -f environment.yml
conda activate env
```

Next install the project source code.
From the project directory, activate the environment and run:

```bash
conda activate env
python -m pip install -e '.[dev]'
```

Verify things have installed successfully by running:

```bash
pytest tests
```


## Description of data

### K00370

#### Raw sequences (`seqs`)

| filename | description |
| ------- | ------- |
| `kegg_seqs_unclust.faa` | All sequences in the KEGG database associated to the KO K00370. Originally located in `soil-ko-wrangling` repository. |
| `soil_seqs_1000.fasta` | Copied from `soil-metagenomics/out/aaseqs/K00370/1000_seqs.fasta` |
| `soil_seqs_800.fasta` |  Copied from `soil-metagenomics/out/aaseqs/K00370/800_seqs.fasta` |


## References
