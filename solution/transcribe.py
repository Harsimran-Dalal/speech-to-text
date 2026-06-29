"""Reference contract for the builderr local-dictation challenge.

Entrants replace the body of transcribe() with their own local engine/router.
The CLI signature and the result.json shape are REQUIRED and checked by the harness:

    python -m solution.transcribe --input clip.wav --mode auto --output result.json

Rules: runs fully local; no outbound network during the scored run (loopback to a
local ASR server is fine); emit the JSON below; no hardcoded phrase fixes.
"""
from __future__ import annotations
import argparse
import json
import time

from solution.engine import run_pipeline

def transcribe(wav_path: str, mode: str = "auto") -> dict:
    t0 = time.time()
    try:
        final_text, lang_guess, timings, candidates, model_ids = run_pipeline(wav_path, mode=mode)
        total_ms = (time.time() - t0) * 1000
        timings["total"] = round(total_ms)
        return {
            "text": final_text,
            "mode_used": mode,
            "language_guess": lang_guess,
            "timings_ms": timings,
            "raw_candidates": candidates,
            "model_ids": model_ids,
            "local_only": True,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "text": "",
            "mode_used": mode,
            "language_guess": "unknown",
            "timings_ms": {"total": round((time.time() - t0) * 1000), "asr": 0, "postprocess": 0},
            "raw_candidates": [{"engine": "none", "text": "", "note": str(e)}],
            "model_ids": [],
            "local_only": True,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--mode", default="auto", choices=["auto", "fast", "hinglish", "verbatim"])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    result = transcribe(args.input, args.mode)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.output}  ({result['timings_ms']['total']}ms, local_only={result['local_only']})")


if __name__ == "__main__":
    main()
