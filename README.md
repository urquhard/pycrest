# pycrest

Simple script that imitates HEPForge CReS weight resampling for ROOT ntuples.

## Setup

Install the Python dependencies before running the script:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Prepare your inputs

`main.py` is configured with a few project-specific values that must match your data:

- `files`: the list of ROOT files to process
- `checkpoint_interval`: how often to write an intermediate checkpoint ROOT file
- `output_dir`: where checkpoint and final output files are written

Edit those values in `main.py` before running the script. The output directory should exist, or you should create it first.

Place your input ROOT ntuples in the `input/` directory, or update the paths in `files` to point to their actual location.

## Create `columns.json`

The script loads `columns.json` and uses that list when computing nearest neighbors. The file must contain every ntuple column you want available during reweighting.

The format is a plain JSON array of branch names:

```json
[
	"nLeps",
	"nJets",
	"weight"
]
```

You can generate the file from an input ROOT file with `uproot`:

```python
import json
import uproot

root_file = "input/your_ntuple.root"
tree_name = "HyWW"

with uproot.open(root_file) as f:
		columns = list(f[tree_name].keys())

with open("columns.json", "w") as handle:
		json.dump(columns, handle, indent=4)
```

Make sure the resulting list includes all columns present in your ntuple that should participate in the distance calculation.

## Reweight multiple target columns

Reweighting is driven by the `target_columns` list passed to `iterate_over_columns` in the main loop at the bottom of `main.py`.

By default it is set to:

```python
iterate_over_columns(file_name, output_dir, target_columns=["weight"])
```

To reweight additional columns, add them to that list:

```python
iterate_over_columns(file_name, output_dir, target_columns=["weight", "your_other_weight"])
```

Each target column will be processed in sequence for every input file.

## Run

After the dependencies, inputs, `columns.json`, and `main.py` settings are ready, run:

```bash
python main.py
```

The script writes periodic checkpoint files to `output_dir` using the pattern `checkpoint_<N>_<input-file-name>.root` and reuses the latest checkpoint when it exists.
