"""Ask Gemini to double-check frames the local YOLO detector was unsure about.

Not a replacement for the local detector - this is a spot-check tool for the specific
failure mode where detect_frame.py/detect_video.py report no ball (or low confidence)
and you want a second opinion before trusting that the ball was genuinely out of frame.

Gemini calls are cheap relative to the value of a correct label, so don't ration them for
cost reasons - batch-sweep a whole flagged set if that's what the task calls for. --frames-dir
sweeps run concurrently (paid-tier quota has real headroom); the only real constraint is API
rate limits, which this script handles with retry-with-backoff (respecting the server's
Retry-After header when present, exponential backoff otherwise) - if you hit sustained 429s
even after backoff, that's a signal to lower --concurrency, not a reason to avoid calling it
in the first place.

Usage:
    # single frame
    python scripts/gemini_verify.py --image data/frames/online_match_01/frame_0042.jpg

    # batch: every frame in a directory, run concurrently, results written to a JSON file
    python scripts/gemini_verify.py --frames-dir data/annotations/detect_review --out results.json

    python scripts/gemini_verify.py --frames-dir data/annotations/detect_review --concurrency 16 --out results.json
    python scripts/gemini_verify.py --image data/frames/online_match_01/frame_0042.jpg --model gemini-pro-latest
"""

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

# Dated model names (gemini-2.5-pro, gemini-2.5-flash, ...) get deprecated out from
# under callers with a 404 ("no longer available to new users") even though they
# still show up in client.models.list() - found 2026-07-19 when every tier-2 Gemini
# call in bootstrap_player_labels.py silently failed and fell through to tier 3
# (bootstrap_ball_labels.py's tier 2 was equally broken, just not noticed since its
# try/except degrades quietly to "leave for manual review"). The "-latest" alias
# floats to whatever the current stable model actually is, so it doesn't rot the
# same way - confirmed working via a live call before switching to it.
DEFAULT_MODEL = "gemini-flash-latest"
DEFAULT_CONCURRENCY = 8
MAX_RETRIES = 6
BASE_BACKOFF_SECONDS = 2.0

PROMPT = """You are assisting a volleyball computer-vision pipeline. A local YOLO \
detector analyzed this frame and did NOT confidently find the volleyball.

Look at the image and answer only with JSON matching this shape:
{"ball_visible": bool, "bbox": [x1, y1, x2, y2] or null, "confidence": "high"|"medium"|"low", "notes": str}

- bbox is in pixel coordinates of the image as given (x1,y1 = top-left, x2,y2 = bottom-right), or null if ball_visible is false.
- notes should be one short sentence explaining what you see (e.g. "ball is partially occluded by the net").
Return only the JSON object, no markdown fences."""

PLAYER_PROMPT = """You are assisting a volleyball computer-vision pipeline. A local YOLO \
detector found some player boxes in this frame, but frames like this one (players \
clustered/overlapping near the net or in a defensive scramble) are exactly where it is \
known to sometimes miss a player entirely.

Look at the image and identify every player on the court (not a referee, coach, or \
spectator). Answer only with JSON matching this shape:
{"player_count": int, "boxes": [[x1, y1, x2, y2], ...], "confidence": "high"|"medium"|"low", "notes": str}

- boxes are in pixel coordinates of the image as given (x1,y1 = top-left, x2,y2 = bottom-right), one per player.
- notes should be one short sentence about anything ambiguous (e.g. "one player mostly hidden behind another near the net").
Return only the JSON object, no markdown fences."""


def _get_client() -> genai.Client:
    load_dotenv()
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def _retry_after_seconds(exc: errors.ClientError) -> float | None:
    response = getattr(exc, "response", None)
    header = response.headers.get("Retry-After") if response is not None else None
    if header is None:
        return None
    try:
        return float(header)
    except ValueError:
        return None


def _call_with_backoff(fn, max_retries: int = MAX_RETRIES):
    """Retry on 429 (rate limit), honoring Retry-After when the server sends one and
    falling back to exponential backoff with jitter otherwise. Rate limits are the only
    real constraint on Gemini usage here - this exists so callers can batch freely
    without hand-rolling throttling per call site."""
    for attempt in range(max_retries):
        try:
            return fn()
        except errors.ClientError as exc:
            if exc.code != 429 or attempt == max_retries - 1:
                raise
            wait = _retry_after_seconds(exc)
            if wait is None:
                wait = BASE_BACKOFF_SECONDS * (2 ** attempt) + random.uniform(0, 1)
            print(f"  rate limited, retrying in {wait:.1f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)


def verify_ball_frame(image_path: str, model: str = DEFAULT_MODEL, client: genai.Client | None = None) -> dict:
    client = client or _get_client()
    image_bytes = Path(image_path).read_bytes()
    mime_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"

    response = _call_with_backoff(lambda: client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            PROMPT,
        ],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    ))
    return json.loads(response.text)


def verify_ball_frames(
    image_paths: list[str], model: str = DEFAULT_MODEL, concurrency: int = DEFAULT_CONCURRENCY,
) -> dict[str, dict]:
    """Batch sweep - safe to point this at a whole flagged set, not just one-off frames.
    Runs concurrently (paid-tier quota has real headroom); per-call backoff (see
    _call_with_backoff) is what keeps this within rate limits, not caller discipline or
    artificial sequential pacing.

    A single frame's failure (e.g. a malformed, non-JSON response for that one image)
    is recorded as {"error": str(exc)} rather than propagated - letting one bad frame
    raise out of as_completed() would discard every other already-completed result in
    the batch, which is a much worse outcome for a "verify a few hundred frames" sweep
    than just flagging the one frame as unresolved."""
    client = _get_client()
    results: dict[str, dict] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(verify_ball_frame, path, model, client): path for path in image_paths}
        for future in as_completed(futures):
            path = futures[future]
            try:
                results[path] = future.result()
            except Exception as exc:
                results[path] = {"error": str(exc)}
            done += 1
            if done % 25 == 0:
                print(f"  verified {done}/{len(image_paths)}")
    return results


def verify_player_frame(image_path: str, model: str = DEFAULT_MODEL, client: genai.Client | None = None) -> dict:
    client = client or _get_client()
    image_bytes = Path(image_path).read_bytes()
    mime_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"

    response = _call_with_backoff(lambda: client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            PLAYER_PROMPT,
        ],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    ))
    return json.loads(response.text)


def verify_player_frames(
    image_paths: list[str], model: str = DEFAULT_MODEL, concurrency: int = DEFAULT_CONCURRENCY,
) -> dict[str, dict]:
    """Batch sweep, same concurrency/backoff/error-isolation shape as verify_ball_frames -
    see its docstring for why a single frame's failure doesn't abort the whole batch."""
    client = _get_client()
    results: dict[str, dict] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(verify_player_frame, path, model, client): path for path in image_paths}
        for future in as_completed(futures):
            path = futures[future]
            try:
                results[path] = future.result()
            except Exception as exc:
                results[path] = {"error": str(exc)}
            done += 1
            if done % 25 == 0:
                print(f"  verified {done}/{len(image_paths)}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Verify a single frame")
    group.add_argument("--frames-dir", help="Verify every .jpg/.png frame in a directory")
    parser.add_argument("--out", help="Write batch results to this JSON file (only used with --frames-dir)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                         help="Parallel requests for --frames-dir sweeps (default: %(default)s)")
    args = parser.parse_args()

    if args.image:
        result = verify_ball_frame(args.image, model=args.model)
        print(json.dumps(result, indent=2))
        return

    frames_dir = Path(args.frames_dir)
    image_paths = sorted(str(p) for p in frames_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    print(f"verifying {len(image_paths)} frames from {frames_dir} (concurrency={args.concurrency})")
    results = verify_ball_frames(image_paths, model=args.model, concurrency=args.concurrency)

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"wrote results -> {args.out}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
