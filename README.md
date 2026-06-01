# Discogs Recommendation

Personal vinyl/electronic release recommender built on Discogs catalog data, label-family graph expansion, and (optional) Essentia EffNet audio embeddings.

Previous notebook experiments live in **`../Archive Discogs Recommendation`**. Large data files live outside this repo under **`~/DiscogsData`**.

## Prerequisites

- Python 3.10+
- Discogs monthly data dump (optional if you already built the SQLite DB)
- For audio scoring: [Essentia TensorFlow models](https://essentia.upf.edu/models.html) and `requirements-audio.txt`

## Data layout

All heavy artifacts are under `DISCOGS_DATA_ROOT` (default `~/DiscogsData`):

```text
~/DiscogsData/
  dumps/          # .xml.gz from Discogs
  xml/            # decompressed XML
  csv/            # discogs-xml2db CSV export
  db/discogs.sqlite
  catalog/        # label_data.json, label_family.json
  embeddings/     # EffNet model + collection .npz
  profile/        # API collection/wantlist exports (future)
  exports/        # recommendation CSV output
  candidate_audio/
```

## Setup

```bash
cd "/Users/barneypinkerton/Projects/Discogs Recommendation"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# discogs-xml2db parser (git submodule)
git submodule update --init --recursive
pip install -r vendor/discogs-xml2db/requirements.txt

cp .env.example .env
# Edit .env — set DISCOGS_DATA_ROOT and Discogs API token if needed
```

Optional audio dependencies:

```bash
pip install -r requirements-audio.txt
```

## Run the pipeline

```bash
# List stages
python main.py --list-stages

# Full run (needs .env with Discogs token)
python main.py --through score

# Or step by step
python main.py build_labels
python main.py sync_profile
python main.py discover
python main.py score
```

| Stage | Description |
|-------|-------------|
| `build_labels` | Verify or build label family JSON cache |
| `sync_profile` | Fetch collection + wantlist → `profile/releases.json` |
| `discover` | SQL candidate pool → `exports/candidates.csv` |
| `score` | Filter, rank, optional have/want → `exports/recommendations.csv` |

Use `--force` on any stage to rebuild cached outputs.

## Vendor: discogs-xml2db

XML → CSV → SQLite ETL is delegated to [discogs-xml2db](https://github.com/philipmat/discogs-xml2db) in `vendor/discogs-xml2db`. See that project’s README for dump export and SQLite import; wrapper modules will be added under `src/discogs_recommender/etl/`.

## Development

```bash
export PYTHONPATH=src
python -m pytest tests/
```

## Archive

The renamed folder **`Archive Discogs Recommendation`** retains all Jupyter notebooks (v4–v13), scoring experiments, and plans. Port logic from there into `src/discogs_recommender/` as needed.
