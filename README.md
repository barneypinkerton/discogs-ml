# Discogs Recommender

A personal vinyl recommender for underground electronic music, built on the full Discogs catalog. It combines metadata-based affinity scoring (label families, artist graph, style/country/year affinities) with Essentia EffNet audio embeddings of your collection to surface the 10 records most aligned with your actual taste.

Previous notebook experiments live in **`../Archive Discogs Recommendation`**. Large data files live outside this repo under `~/DiscogsData`.

---

## How it works

The pipeline runs in five stages:

```
Discogs data dump → SQLite
                         ↓
          build_labels → label family graph
          sync_profile → collection + wantlist from API
          discover     → 50k candidate pool (affinity + style-discovery + compilations)
          score        → filter, rank, enrich → top 100 recommendations.csv
          audio_rank   → download audio, EffNet embeddings, cosine similarity → top10.csv
```

**Metadata scoring** (`score`) ranks candidates using artist/label affinity scores built from your collection and wantlist, then applies style, country, and year affinities. Hard caps on `have_count` and `want_count` (≤600 each) keep results underground. Owned artists and label families are deduplicated so no single artist or label dominates.

**Audio ranking** (`audio_rank`) fetches every YouTube video linked on each Discogs release page (up to 6 per release), downloads the full audio in parallel, runs Essentia EffNet to produce a 1280-d embedding per track, mean-pools across all tracks on the release, then ranks by cosine similarity to the weighted centroid of your collection embeddings.

---

## Data layout

```
~/DiscogsData/
  dumps/                  # .xml.gz monthly dumps from Discogs
  xml/                    # decompressed XML
  csv/                    # discogs-xml2db CSV export
  db/discogs.sqlite       # main catalog DB (~10GB)
  catalog/                # label_data.json, label_family.json
  embeddings/
    discogs_artist_embeddings-effnet-bs64-1.pb   # Essentia EffNet model
    my_collection_embeddings.npz                 # pre-computed collection profile
  profile/
    releases.json         # API collection + wantlist export
  exports/
    candidates.csv        # discover stage output (~50k rows)
    recommendations.csv   # score stage output (top 100)
    top10.csv             # audio_rank output (final 10)
  candidate_audio/
    {release_id}_t{n}.wav # downloaded track audio (cached)
    url_cache.json        # cached Discogs video URL lookups
```

### Collection embeddings format

`my_collection_embeddings.npz` must contain:

| Key | Shape | Description |
|-----|-------|-------------|
| `embeddings` | `(N, 1280)` | Per-release EffNet embeddings |
| `centroid` | `(1, 1280)` | Weighted mean of the collection (used directly) |
| `filenames` | `(N,)` | Release identifiers |
| `weights` | `(N,)` | Per-release weights (collection vs wantlist) |

---

## Prerequisites

- Python 3.10+
- Discogs API token — [discogs.com/settings/developers](https://www.discogs.com/settings/developers)
- Discogs monthly data dump (or a pre-built SQLite DB)
- `ffmpeg` in PATH (used by yt-dlp for audio conversion)
- For audio ranking: [Essentia EffNet Discogs model](https://essentia.upf.edu/models.html) and `requirements-audio.txt`

---

## Setup

```bash
cd ~/discogs-ml
python3 -m venv .venv
source .venv/bin/activate

# Core dependencies
pip install -r requirements.txt

# Audio dependencies (EffNet embeddings + yt-dlp)
pip install -r requirements-audio.txt

# discogs-xml2db submodule (ETL from dump → SQLite)
git submodule update --init --recursive
pip install -r vendor/discogs-xml2db/requirements.txt

# Configure
cp .env.example .env
# Edit .env: set DISCOGS_USER_TOKEN, DISCOGS_USERNAME, DISCOGS_DATA_ROOT
```

---

## Step 0: Build your taste profile

Before running `audio_rank` you need a collection embeddings file (`my_collection_embeddings.npz`). This is generated once from your local music files and only needs to be re-run when you add new tracks.

**What you need:**
- A folder of MP3/WAV/FLAC files representing your taste (your bought/ripped collection, saved tracks, mixes, etc.)
- The Essentia EffNet model file — download `discogs_artist_embeddings-effnet-bs64-1.pb` from [essentia.upf.edu/models.html](https://essentia.upf.edu/models.html) (Audio classification → Discogs-EffNet) and place it at `~/DiscogsData/embeddings/`
- `requirements-audio.txt` installed

```bash
cd ~/discogs-ml
source .venv/bin/activate

python scripts/build_collection_profile.py \
  --mp3-dir /path/to/your/music \
  --output ~/DiscogsData/embeddings/my_collection_embeddings.npz
```

The script will:
1. Scan the folder for `.mp3`, `.wav`, `.flac`, `.aiff`, `.m4a` files
2. Run EffNet on each track, mean-pooling frame embeddings into one 1280-d vector per track
3. Compute the collection centroid (mean across all tracks)
4. Save `filenames`, `embeddings`, and `centroid` to the `.npz` file
5. Print the most and least representative tracks as a sanity check

**CLI options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--mp3-dir` | *(required)* | Folder containing your music files |
| `--model` | `$DISCOGS_DATA_ROOT/embeddings/discogs_artist_embeddings-effnet-bs64-1.pb` | Path to EffNet model |
| `--output` | `$DISCOGS_DATA_ROOT/embeddings/my_collection_embeddings.npz` | Where to write the profile |
| `--force` | off | Rebuild even if output already exists |

Set `DISCOGS_DATA_ROOT` in `.env` to change the default data directory (default: `~/DiscogsData`).

---

## Running the pipeline

```bash
cd ~/discogs-ml
source .venv/bin/activate

# List all stages
python3 main.py --list-stages

# Run everything end-to-end (interactive preference wizard)
python3 main.py --through score --interactive
python3 main.py audio_rank

# Or stage by stage
python3 main.py build_labels
python3 main.py sync_profile
python3 main.py discover
python3 main.py score --interactive
python3 main.py audio_rank
```

### Pipeline stages

| Stage | Output | Description |
|-------|--------|-------------|
| `build_labels` | `catalog/label_family.json` | Builds label family graph from Discogs XML |
| `sync_profile` | `profile/releases.json` | Fetches your collection + wantlist via Discogs API |
| `discover` | `exports/candidates.csv` | SQL candidate pool: artist/label affinity + style discovery |
| `score` | `exports/recommendations.csv` | Filters, ranks and enriches top 100; applies have/want caps |
| `audio_rank` | `exports/top10.csv` | Downloads audio, computes EffNet embeddings, returns top 10 |

Use `--force` on any stage to rebuild cached outputs. Use `--interactive` with `score` to launch the preference wizard (styles, countries, year range, boost strength).

### Typical re-run after updating your collection

```bash
python3 main.py sync_profile --force
python3 main.py score --force --interactive
python3 main.py audio_rank --force
```

---

## Output: `top10.csv`

| Column | Description |
|--------|-------------|
| `release_id` | Discogs release ID |
| `title` | Release title |
| `artists` | Artist name(s) |
| `labels` | Label name(s) |
| `styles` | Discogs style tags |
| `country` / `released` | Origin and year |
| `score` | Final metadata score |
| `audio_sim` | Cosine similarity to collection centroid (0–1) |
| `audio_blend_score` | Blended rank score (`audio_weight * audio_sim + (1 - audio_weight) * norm_score`) |
| `have_count` / `want_count` | Community stats (master-level) |
| `image_url` | Discogs cover art URL |
| `discogs_url` | Direct link to the release |
| `youtube_url` | First YouTube video from the release page |

---

## Configuration

Key settings in `config/default.yaml`:

```yaml
score:
  max_have_count: 600        # exclude records with more than this many haves
  max_want_count: 600        # exclude records with more than this many wants
  exclude_owned_artists: true # filter out artists already in your collection
  max_per_artist: 2
  max_per_label: 3
  top_n_export: 100

discover:
  min_style_match_count: 2   # discovery bucket requires ≥N style overlaps with profile
  candidate_pool_limit: 50000

audio:
  top_n_final: 10
  audio_weight: 0.6          # blend: 60% audio similarity, 40% metadata score
  max_videos_per_release: 6  # cap on YouTube tracks downloaded per release
  download_workers: 4        # parallel yt-dlp download threads
  cache_audio: true          # skip re-downloading already-cached wav files
```

Override any value via environment variable or a custom YAML passed with `--config`.

---

## Development

```bash
export PYTHONPATH=src
python3 -m pytest tests/
```

## Archive

**`../Archive Discogs Recommendation`** retains all Jupyter notebooks (v4–v13), scoring experiments, and plans. Port logic from there into `src/discogs_recommender/` as needed.
