# Lumen / Image Ranker

A private, local-first taste engine for photography. Choose between two images, maintain a live Elo collection, train a personalized pairwise vision model, and discover licensed high-resolution work that the model predicts you will value.

## Quick start

Requires Python 3.9+.

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/image-ranker seed --limit 60
.venv/bin/image-ranker serve
```

Open `http://127.0.0.1:8787`. Use left/right arrow to focus an image and space to choose it, or click either image. The collection view is always ordered by live Elo.

On a phone, tap either photograph to choose it, swipe left or right to choose that side, and swipe up to skip the pair.

After at least 20 choices, install the model worker and train:

```bash
.venv/bin/pip install -e '.[ml]'
.venv/bin/image-ranker train
```

You can also import your own high-resolution JPEG, PNG, or WebP files:

```bash
.venv/bin/image-ranker import ~/Pictures/one.jpg ~/Pictures/two.png
```

## Local maintenance jobs

Run one maintenance check with `image-ranker jobs`, or keep a foreground loop running locally:

```bash
.venv/bin/image-ranker jobs --watch --interval 900
```

The first model trains at 20 comparisons and retrains after each additional 50. When fewer than 20 active images have never been compared, the job imports up to 60 more candidates. Before the first model, discovery uses curated high-quality sources; afterward, both `seed` and the maintenance job score a 3× candidate pool with the latest private model and keep the strongest candidates. Each crawl report identifies its `curated-seed` or `model-guided` mode.

Override the thresholds with `--train-minimum`, `--train-batch`, `--unranked-threshold`, `--crawl-limit`, and `--epochs`; the matching environment variables are `IMAGE_RANKER_TRAIN_MINIMUM`, `IMAGE_RANKER_TRAIN_BATCH`, `IMAGE_RANKER_UNRANKED_THRESHOLD`, `IMAGE_RANKER_CRAWL_LIMIT`, and `IMAGE_RANKER_TRAIN_EPOCHS`. `IMAGE_RANKER_JOBS_INTERVAL` controls the default watch interval. The loop only reads and writes the private directory selected by `IMAGE_RANKER_DATA` (default: local `data/`); keep that directory outside synced or public Git folders if desired.

## Mobile app and private connection

The hosted app at `https://lumen-ranker.vercel.app` is an installable, static PWA. On iPhone, open it in Safari and choose **Share → Add to Home Screen**; on Android, use the browser’s **Install app** action. Its offline cache contains only the HTML, CSS, JavaScript, manifest, and icons—not photographs, rankings, comparisons, or model artifacts.

Run the separate authenticated endpoint on the Mac with a private token and the exact production app origin:

```bash
IMAGE_RANKER_REMOTE_TOKEN_FILE=./data/remote-token \
IMAGE_RANKER_ALLOWED_ORIGIN=https://lumen-ranker.vercel.app \
.venv/bin/image-ranker serve --remote
```

Remote mode listens on port `8788` by default. The production private API address is `https://sheep-biter.tail1b4cdd.ts.net`; the tunnel or reverse proxy must send that address to the remote port. Local mode on `127.0.0.1:8787` remains unauthenticated and loopback-only.

The remote listener itself also stays on `127.0.0.1`; only the authenticated tunnel should expose it. Every `/api/`, `/media/`, and `/thumb/` request requires both the exact configured `Origin` and `Authorization: Bearer <token>`, including the minimal `/api/health` endpoint. Preflights accept only the methods valid for that route and the `Authorization`/`Content-Type` headers, while private image responses are never marked as publicly cacheable.

`IMAGE_RANKER_REMOTE_PORT` changes the separate remote port. `IMAGE_RANKER_REMOTE_TOKEN_FILE` overrides the default `<IMAGE_RANKER_DATA>/remote-token`; `IMAGE_RANKER_REMOTE_TOKEN` is also supported, but the owner-only token file is preferred so the secret stays out of shell history and service definitions. Remote startup refuses tokens shorter than 32 characters, obviously low-entropy values, non-owner-private token files, missing origins, and origins containing paths or insecure non-loopback HTTP.

Connect a new device with a one-time setup URL in this form, percent-encoding both values:

```text
https://lumen-ranker.vercel.app/#connect?api=https%3A%2F%2Fsheep-biter.tail1b4cdd.ts.net&token=YOUR_ENCODED_TOKEN
```

The fragment after `#` is not sent in the HTTP request to Vercel. Lumen stores the server address and token only in that browser’s local storage, immediately removes the fragment from the address bar and history entry, and fetches every private image as an authenticated in-memory blob. Treat the setup URL like the token itself; use **Connect** in the header and **Forget this device** to erase the saved connection.

`vercel.json` selects `web/` as a no-build static output directory, so the Vercel deployment has no Functions, database, storage, crawler, or training process. `.vercelignore` adds a second boundary around the local data directories and token files.

## Keep services running on macOS

The CLI generates three per-user LaunchAgents: the local app on `127.0.0.1:8787`, the authenticated remote API on `127.0.0.1:8788`, and a one-shot maintenance pass every 15 minutes.

LaunchAgents do not inherit an interactive terminal's macOS privacy grants. If this checkout is under `Documents`, `Desktop`, or another TCC-protected location and a service log reports `Operation not permitted`, stage both the runtime and private data under `~/Library/Application Support/Lumen` instead of granting broad disk access:

```bash
LUMEN_HOME="$HOME/Library/Application Support/Lumen"
LUMEN_RUNTIME="$LUMEN_HOME/runtime"
LUMEN_DATA="$LUMEN_HOME/data"
mkdir -p "$LUMEN_RUNTIME" "$LUMEN_DATA"
rsync -a --delete --exclude '.git/' --exclude '.venv/' --exclude '.vercel/' --exclude 'data/' ./ "$LUMEN_RUNTIME/"
rsync -a data/ "$LUMEN_DATA/"
if [ ! -s "$LUMEN_DATA/remote-token" ]; then
  umask 077
  openssl rand -hex 32 > "$LUMEN_DATA/remote-token"
fi
python3 -m venv "$LUMEN_RUNTIME/.venv"
"$LUMEN_RUNTIME/.venv/bin/pip" install -e "${LUMEN_RUNTIME}[ml]"
```

Then render and install from the staged runtime so every executable, working-directory, data, token, and log path is outside the protected folder:

```bash
IMAGE_RANKER_ROOT="$LUMEN_RUNTIME" IMAGE_RANKER_DATA="$LUMEN_DATA" \
  "$LUMEN_RUNTIME/.venv/bin/image-ranker" launchd render
IMAGE_RANKER_ROOT="$LUMEN_RUNTIME" IMAGE_RANKER_DATA="$LUMEN_DATA" \
  "$LUMEN_RUNTIME/.venv/bin/image-ranker" launchd install
```

`IMAGE_RANKER_ROOT` defaults to the checkout containing the installed package, so existing local commands are unchanged when it is unset. Every generated agent persists the resolved root and data paths, preventing a background process from falling back to the protected checkout; `rsync -a` also preserves the existing private token mode while staging data.

For an unprotected checkout, review the generated templates with:

```bash
.venv/bin/image-ranker launchd render
```

Remote mode requires an existing high-entropy token that is readable only by your account. For the default private data directory, create it once with:

```bash
mkdir -p data
umask 077
openssl rand -hex 32 > data/remote-token
```

Then install or inspect the services:

```bash
.venv/bin/image-ranker launchd install
.venv/bin/image-ranker launchd status
```

Installation is idempotent: unchanged loaded services are left alone, while only a changed plist is safely unloaded and reloaded. The token itself is never copied into a plist or modified by the installer. Every process uses a `077` umask and writes stdout/stderr only under private `data/logs/`; the scheduled job is launchd-managed rather than a permanent polling process. To stop the services and remove only their plists, leaving images, rankings, models, token, and logs untouched:

```bash
.venv/bin/image-ranker launchd uninstall
```

Use `--token-file` and `--executable` with `launchd install` when the private token or virtual environment lives elsewhere, and `--jobs-interval` to change the maintenance cadence. The remote service remains loopback-only; a separately authenticated tunnel or proxy must be responsible for public ingress.

## Privacy and open-source boundary

Only code and documentation belong in Git. `.gitignore` excludes `data/`, which contains downloaded images, the SQLite comparison history, rankings, cached embeddings, and trained model artifacts. The server binds to localhost by default. Wikimedia files retain source, creator, and license metadata; each work remains governed by its own license.

## How it works

- **Ranking:** adaptive Elo gives instant, legible feedback. Every comparison is also stored immutably so rankings can later be reconstructed or refit with Bradley–Terry.
- **Pair selection:** early sessions prioritize under-compared items and close Elo neighbors while avoiding immediate repeats. Once a model exists, the sampler favors predicted close calls while preserving graph coverage and explicit random exploration.
- **Vision model:** a frozen OpenCLIP encoder turns each image into a cached vector. A small regularized utility head learns `P(A > B) = sigmoid(score(A) - score(B))`; the frozen encoder is deliberately not fine-tuned with scarce labels.
- **Discovery:** the first adapter ingests Wikimedia Commons Featured Pictures through its official API, validates every decode and rejects images below 1200px on either edge or 2.5MP. Provider adapters should preserve attribution and explicit rights metadata.

See [RESEARCH.md](RESEARCH.md) for the literature review, scaling plan, and evaluation gates.

## Roadmap

1. Label 100–200 diverse comparisons and establish preference consistency against the existing grouped chronological holdout.
2. Add image-disjoint, photographer-disjoint, and near-duplicate-cluster evaluation once the library is large enough.
3. Add rights-clean adapters for the Smithsonian, Cleveland Museum, and Art Institute of Chicago.
4. At 500–1,000 labels, compare Bayesian logistic regression, bootstrap ensembles, and a two-layer head against the linear baseline.
5. Only after several thousand labels and a held-out gain, test LoRA or last-block encoder tuning.

## License

MIT. Images and metadata are not covered by this repository’s license.
