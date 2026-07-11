# CLAUDE.md — volleyball-tracker

Standing behavioral contract for Claude Code sessions in this repo. Read this before making changes.

Source of truth for *what* to build and in what order is [`plan.md`](plan.md) — it defines the phased roadmap (footage → detection → tracking → calibration → stats → events → storage/viz) and explicitly says **do not skip ahead**. This file governs *how* to work, not what to build next.

---

## 1. Project overview & tech stack

**What this is:** a computer-vision pipeline that takes volleyball match video in and produces player/ball tracking data out (and eventually derived stats). Currently in Phase 2–3 of `plan.md` (detection + tracking); Phase 7 (FastAPI + React storage/viz layer) does not exist yet — until that phase starts, **this is a pure Python project**. Do not introduce a `package.json`, TypeScript, or a C++ build unless a plan.md phase explicitly calls for it.

**Stack:**
- **Python 3.12**, venv at `.venv/` (PowerShell: `.venv\Scripts\Activate.ps1`; scripts run directly via `.venv\Scripts\python.exe` also work without activating)
- **Detection:** Ultralytics YOLO (`ultralytics`) — pretrained COCO weights for players, fine-tuned checkpoints for the ball
- **Tracking:** `supervision` (ByteTrack)
- **Video/image I/O:** OpenCV (`opencv-python`)
- **Dataset sourcing/labeling:** `roboflow`, `yt-dlp` for pulling match footage
- **Secrets:** `python-dotenv`, loaded from `.env` — required keys are listed in `.env.example` (currently `ROBOFLOW_API_KEY`, `GEMINI_API_KEY`)
- **Multimodal verification:** `google-genai` SDK, talking to Gemini directly — see §3 for when/why
- **Test/lint:** `pytest`, `ruff`

**Where things live (do not restructure without asking):**

| Path | Contents | In git? |
|---|---|---|
| `data/raw/` | Source match clips (`.mp4`) | gitignored (large binaries) |
| `data/frames/`, `data/frames_dense*/` | Extracted frames for annotation/testing | gitignored |
| `data/annotations/` | Manual/self labels — CSVs (`ball_dense_scan*.csv`) are tracked; large image subfolders (`detect_review/`, `ball_zero_shot/`, `v2_precision_check/`) are gitignored | mixed, check `.gitignore` before adding a new subfolder |
| `data/annotations/video_detections/` | Cached per-video per-frame detection CSVs (output of `detect_video.py`) | tracked |
| `data/self_labeled/*/images/`, `labels.cache` | Self-labeled training data | gitignored |
| `data/tracked/` | Rendered annotated output videos (output of `track_video.py`) | gitignored |
| `models/*.pt` | YOLO checkpoints (player + ball, multiple versions) | gitignored — never commit weights |
| `runs/` | Ultralytics training run artifacts | gitignored |
| `scripts/` | All pipeline code — flat, no package structure. Scripts import each other directly (e.g. `track_video.py` does `from detect_frame import BALL_CONF, PLAYER_CONF`), which only works because each script's directory is implicitly on `sys.path` when run as `python scripts/foo.py` from repo root. Keep this pattern; don't turn `scripts/` into a package without a good reason. |
| `tests/` | pytest suite | tracked |
| `.env` / `.env.example` | API keys (gitignored / template) | `.env` never committed |

Coordinate convention already in use: bounding boxes are `[x1, y1, x2, y2]` in **pixel space**, top-left origin. Preserve this everywhere until Phase 4 (court calibration) introduces court-space (meters) coordinates — see §5 for the convention to use once that lands.

---

## 2. Build, test, and verification loops

### Install
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # PowerShell
pip install -r requirements.txt
pip install -r requirements-torch.txt   # separate: pins a CUDA 11.8 index for torch/torchvision
```

### Lint
```bash
ruff check .
ruff check . --fix     # auto-fix safe violations
```

### Test
```bash
pytest                 # whole suite
pytest -v tests/test_tracking_math.py   # single file, verbose
```
`tests/test_tracking_math.py` covers the pure logic in `track_video.py` (`box_center`, `best_box`, `interpolate_gaps`, `to_sv_detections`) — no GPU or model weights required, runs in seconds. **Add a test here whenever you touch geometric/state-tracking logic** (interpolation, coordinate math, track bookkeeping). Detection functions that require loading YOLO weights (`detect_frame.py`) are not unit-tested with mocks — verify those with the sanity-check commands below instead, since a mocked YOLO model would test nothing real.

### Run the pipeline
```bash
# single-frame detection sanity check
python scripts/detect_frame.py --image data/frames/<clip>/frame_0000.jpg --save-annotated

# cache detections across a clip (use --max-frames for fast iteration)
python scripts/detect_video.py --video data/raw/<clip>.mp4 --max-frames 200

# track + render annotated output (auto-runs detect_video.py first if no cache exists)
python scripts/track_video.py --video data/raw/<clip>.mp4 --max-frames 200
```

### Mandatory autonomy rule

Whenever you change code in `scripts/` or `tests/`, before reporting the task done you MUST, without waiting for permission:
1. `ruff check .` — fix violations.
2. `pytest` — all tests green.
3. A **fast** sanity run of whatever you touched: `detect_frame.py` on one image, or `detect_video.py` / `track_video.py` with `--max-frames 50`–`200` on an existing clip in `data/raw/`. Confirm output looks structurally sane (right JSON shape, non-empty CSV, video file written) — you don't need to visually inspect every frame yourself.

Iterate autonomously through failures at this fast-check tier — fix, rerun, repeat — without asking for feedback, **unless** you hit the escalation criteria in §4 (3 failed attempts at the same failure, needs elevated privileges, touches a full/expensive job, or git state is unclear). Full-clip runs, training/fine-tuning jobs, and anything that ties up the GPU for more than a minute or two are **not** part of this fast-check loop — see §4 for those.

---

## 3. Task routing: Claude model tiers, and Claude vs. Gemini

Two independent routing decisions live in this section: which *Claude* model tier handles a given piece of work, and when to reach for *Gemini* instead of Claude entirely. Neither happens automatically — both require deliberately invoking the right tool (`Agent` with a `model` override, or `scripts/gemini_verify.py`).

### 3a. Claude model-tier delegation (via the `Agent` tool)

The default session model handles ordinary implementation work directly — no delegation needed for typical `scripts/` changes. Delegate to a different tier only when the task shape clearly calls for it:

| Tier | When | Example in this repo |
|---|---|---|
| **Haiku** (`Agent` with `model: haiku`) | Cheap, high-volume, mechanical text work that doesn't need strong reasoning | Summarizing a long `ultralytics` training log; tailing/grepping across many files |
| **Explore agent** (read-only, no write tools) | Open-ended codebase search that would otherwise burn main-thread context reading files one by one | "find every place `BALL_CONF` or `MAX_BALL_GAP_FRAMES` is referenced" |
| **Sonnet** (default) | Normal implementation, debugging, test writing | Everything in §2 as written |
| **Opus** (`Agent` with `model: opus`) | Rare — a hard architectural decision with real downstream cost if wrong | Designing the Phase 4 homography/court-calibration approach; Phase 6 event-detection architecture, if pursued |

Don't spawn subagents reflexively — per the general operating rules, only delegate when it actually saves context or lets independent work run in parallel. A single-file edit doesn't need a Haiku subagent; a genuinely large log or multi-hundred-file search does.

### 3b. When to use Gemini (via `scripts/gemini_verify.py`)

Gemini calls are cheap relative to the value of a correct label or a caught bug — **don't ration them for cost reasons.** The key is a paid key with real headroom (not free-tier), so the only real constraint is rate limits, and `gemini_verify.py` already handles that itself (concurrent `--frames-dir` sweeps, default 8 workers, plus retry-with-backoff on 429 honoring `Retry-After` when the server sends one). That means:

- **Batch freely, and let it run concurrently.** If ten frames are flagged as low-confidence ball detections, verify all ten in one `--frames-dir` sweep — don't hand-pick "a couple" to be sparing, and don't add artificial sequential pacing on top of what the script already does.
- **Reach for it proactively**, not just as a last resort: e.g. when computing the precision/recall numbers `plan.md`'s Phase 2 deliverable calls for, Gemini spot-checks on the held-out set are a reasonable way to semi-automate ground-truth verification, not something to avoid because it costs a call.
- If you see sustained 429s even after the built-in backoff exhausts its retries, lower `--concurrency` rather than avoiding the tool — that's the one real signal this setup produces.

Use it for:
- **Multimodal spot-checks**: `python scripts/gemini_verify.py --image <path>` (single) or `--frames-dir <dir> --out results.json` (concurrent batch) when the local YOLO detector reports no ball / low confidence and you want a second opinion.
- **Digesting large raw artifacts** you don't want to pull into your own context wholesale — a multi-thousand-row `detect_video.py` CSV, a long training log.
- Parsing third-party docs pulled via `WebFetch`/`WebSearch` when you want a second read on something ambiguous.

Still use Claude (not Gemini) for repo file management, architecture, geometric/state-tracking logic, tests, and all terminal/git operations — Gemini is for multimodal judgment calls and bulk digestion, not for writing or reasoning about this codebase.

### On the `gemini-api-docs-mcp.dev` MCP server
No such MCP server is connected in this environment, and its existence is unverified. **Do not assume it exists or route work through it.** If Gemini API documentation lookup is needed, use `WebFetch`/`WebSearch` against Google's official Gemini API docs instead, and flag to the user if a dedicated MCP server would be genuinely useful so they can set one up deliberately.

---

## 4. Cooldown & escalation protocols

Stop executing and ask the user for input when:

- **3 failed iterations** on the same test/lint/sanity-check failure without a new hypothesis — don't keep retrying the same fix.
- A command needs elevated privileges (Windows equivalent of `sudo` — running as Administrator, modifying system PATH, installing global tools outside the venv).
- You're about to start a **heavy/expensive job**: full-clip `detect_video.py`/`track_video.py` runs (no `--max-frames` cap), YOLO fine-tuning/training runs, or anything that will hold the GPU for more than a couple minutes. Confirm scope (which clip, how long it'll take) before kicking these off — the fast-autonomy rule in §2 covers small sanity runs only, not this tier.
- Git state is unclear or conflicting: uncommitted changes you didn't make, merge conflicts, detached HEAD, or anything that would require `git reset --hard`, `git clean -f`, force-push, or discarding work to resolve.
- A fix would require deleting or overwriting tracked model checkpoints, footage, or annotation data — these are expensive to regenerate (labeling time, training time, or footage that may not be re-downloadable).
- The right fix would change `plan.md`'s phase order or scope (e.g. jumping to Phase 4 calibration before Phase 3 tracking is solid) — scope discipline is explicitly called out in `plan.md` as a project priority; flag the temptation instead of acting on it.
- Anything touching secrets: `.env` contents, API keys, or credential handling beyond the established `python-dotenv` pattern.

---

## 5. Coding style & architectural constraints

- **Type hints everywhere**, using modern 3.12 syntax already in use in this repo (`list[dict]`, `dict[int, tuple[float, float]]`, `X | None`) — not `typing.List`/`Optional`.
- **`pathlib.Path`** for all filesystem paths in new code, not raw strings — match `track_video.py`/`detect_video.py`, which already do this for CLI args.
- **Don't mutate shared frame/detection state across pipeline stages.** `track_video.py`'s two-pass structure (track first, render second) exists so tracking decisions don't depend on rendering order — preserve that separation when extending the pipeline rather than collapsing it into a single mutating pass.
- **Bounding boxes are `[x1, y1, x2, y2]` pixel-space, top-left origin** — keep this convention consistent across every new function. Once Phase 4 (court calibration) introduces court-space coordinates (meters, court-relative origin), suffix variables/functions to disambiguate (`_px` for pixel space, `_court` for calibrated court space) — never mix the two without an explicit conversion step through the homography.
- Prefer pure, testable functions for geometry/state logic (like `box_center`, `interpolate_gaps`) that don't touch I/O or models — this is what makes the `tests/` fast-check loop possible without GPU/model overhead. When adding new tracking/geometry logic, ask whether it can be written and tested this way before reaching for an integration-style test.
- Confidence thresholds and tuning constants (`PLAYER_CONF`, `BALL_CONF`, `MAX_BALL_GAP_FRAMES`, ByteTrack's `minimum_matching_threshold`/`lost_track_buffer`) are empirically tuned — see the comments in `track_video.py` for why specific values were chosen. Don't change them without re-running the sanity checks in §2 and noting the new empirical basis in a comment, matching the existing style.
- No real-time constraint on this project (per `plan.md`) — prefer correctness and clarity over micro-optimizing hot loops.
