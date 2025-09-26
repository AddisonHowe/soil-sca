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

## Directories

## References

[1] N. Halabi, O. Rivoire, S. Leibler, and R. Ranganathan, Protein Sectors: Evolutionary Units of Three-Dimensional Structure, Cell 138, 774 (2009).

[2] O. Rivoire, K. A. Reynolds, and R. Ranganathan, Evolution-Based Functional Decomposition of Proteins, PLoS Comput Biol 12, e1004817 (2016).
