# Data availability

This codebase does not contain HEAPO data and does not grant additional rights
to the dataset.

## Source

- Project repository: <https://github.com/tbrumue/heapo>
- Data record used by the optional downloader:
  <https://zenodo.org/records/15056919>
- HEAPO repository snapshot used during adapter development:
  `38768a8092552e9e0f0927a51f30ad31a27bc002`

Users are responsible for reviewing the current dataset license, access terms,
recommended citation, and any ethical or privacy conditions at the source.
The data authors' citation requirements apply independently of this software's
MIT license.

## Expected local structure

Configure `data.root` to the directory containing:

```text
meta_data/
reports/
smart_meter_data/
weather_data/
```

The adapter reads the documented semicolon-separated metadata, 15-minute smart
meter, and hourly weather files. The raw files are never rewritten.

## Version control policy

The `.gitignore` excludes `data/`, `heapo_data/`, `outputs/`, Parquet files,
model bundles, and checkpoints. Before pushing, run:

```bash
git status --short
git ls-files | grep -E '\.(parquet|joblib|ckpt|pth|zip)$' || true
```

Only source code, configuration examples, documentation, and synthetic tests
belong in the public repository.
