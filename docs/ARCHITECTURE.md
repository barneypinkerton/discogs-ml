# Architecture

## Data flow

```text
Discogs API → profile/releases.json
Discogs dumps → discogs-xml2db → CSV → SQLite
profile + SQLite → exports/candidates.csv → exports/recommendations.csv
```

## Pipeline stages

| Stage | Module | Status |
|-------|--------|--------|
| `build_labels` | `catalog/label_graph.py` | Done |
| `sync_profile` | `profile/sync.py` | Done |
| `discover` | `recommend/discover.py` | Done |
| `score` | `recommend/score.py` | Done |
| `export_csv` | `etl/` | Planned |
| `load_sqlite` | `etl/` | Planned |
| `embed_audio` | `profile/` | Planned |

## Run full pipeline

```bash
source .venv/bin/activate
cp .env.example .env   # set DISCOGS_USER_TOKEN, DISCOGS_USERNAME
python main.py --through score
```

Outputs:

- `~/DiscogsData/profile/releases.json`
- `~/DiscogsData/exports/candidates.csv`
- `~/DiscogsData/exports/recommendations.csv`

## Archive reference

- Label graph: v12 `02.2_database_setup with label family creation.ipynb`
- SQL discovery: v12 `03.3_candidate_scoring.ipynb`
- API profile: v11 `discogs_ml_v11.ipynb`
- Scoring formula: v11 `discogs_ml_v11.ipynb` (have/want + overknown penalty)
