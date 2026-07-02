"""The ONE function you implement for the STREAMING dictation track.
"""
from __future__ import annotations

import re
import os
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import difflib

from solution.engine import get_model, run_pipeline, transcribe_with_loop_guard, get_model_lock

_SR = 16000
_MIN_AUDIO_BYTES = int(_SR * 0.4) * 2  # ~0.4s before the first draft (2 bytes/sample)

# per-clip state (the harness calls draft_reset() between clips)
_prev_text: str = ""
_committed: str = ""
_last_len: int = 0
_is_hinglish: bool = False
_stdout_redirected: bool = False
_executor = ThreadPoolExecutor(max_workers=1)
_future = None

def draft_reset() -> None:
    """Called by the sealed harness at the start of each clip. Clear per-clip state."""
    global _prev_text, _committed, _last_len, _is_hinglish, _future, _executor
    import solution.engine
    solution.engine._bg_cancel = False
    _prev_text = ""
    _committed = ""
    _last_len = 0
    _is_hinglish = False
    if _future is not None:
        try:
            _future.cancel()
        except:
            pass
        _future = None
    if _executor is not None:
        try:
            _executor.shutdown(wait=False)
        except:
            pass
    _executor = ThreadPoolExecutor(max_workers=1)

def _bg_transcribe(audio_buffer: bytes, force_hi: bool):
    audio = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
    text = ""
    try:
        detected_hinglish = force_hi
        lang_guess = ""
        hi_prob = 0.0
        ur_prob = 0.0
        prob = 0.0
        
        model = get_model("small")
        if not detected_hinglish:
            try:
                lock = get_model_lock(model)
                with lock:
                    lang_guess, prob, all_probs = model.detect_language(audio)
                probs = dict(all_probs) if all_probs else {}
                hi_prob = probs.get("hi", 0.0)
                ur_prob = probs.get("ur", 0.0)
                if (lang_guess in ("hi", "ur", "mr", "ne", "sa") and prob > 0.15) or hi_prob > 0.15 or ur_prob > 0.15:
                    detected_hinglish = True
            except Exception:
                pass
                
        # Write pre-ASR check to log
        try:
            here = os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(here, "draft_audit.log"), "a", encoding="utf-8") as f:
                f.write(f"BG_DETECT_PRE: len={len(audio_buffer)}, force_hi={force_hi}, guess={lang_guess}, hi={hi_prob:.3f}, ur={ur_prob:.3f}, detected={detected_hinglish}\n")
        except:
            pass
            
        text, info = transcribe_with_loop_guard(
            model, 
            audio, 
            language="hi" if detected_hinglish else "en", 
            task="transcribe", 
            beam_size=1,
            repetition_penalty=1.05,
            is_bg=True
        )
        
        # Write post-ASR check to log
        try:
            here = os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(here, "draft_audit.log"), "a", encoding="utf-8") as f:
                f.write(f"BG_DETECT_POST: len={len(audio_buffer)}, text={text!r}, detected={detected_hinglish}\n")
        except:
            pass
            
        # Finalize spelling, casing, and transliterate to Devanagari if Hinglish
        from solution.engine import finalize_text
        text = finalize_text(text, is_hinglish=detected_hinglish)
        
        # Repetition loop guard using core engine's loop guard
        from solution.engine import has_any_repetition, truncate_any_loop
        if has_any_repetition(text):
            text = truncate_any_loop(text)
        if has_any_repetition(text):
            text = ""
            
        return text, detected_hinglish
    except Exception as e:
        import traceback
        traceback.print_exc()
        return "", False
 
def draft(audio_buffer: bytes, is_final: bool) -> tuple[str, int]:
    global _prev_text, _committed, _last_len, _is_hinglish, _future, _stdout_redirected
    
    if not _stdout_redirected and os.environ.get("DEBUG_STT") != "1":
        try:
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")
        except:
            pass
        _stdout_redirected = True
        
    if is_final:
        import solution.engine
        solution.engine._bg_cancel = True
        if _future is not None:
            try:
                # Wait for the background job to finish to retrieve its language detection result
                text, detected_hinglish = _future.result(timeout=1.5)
                if detected_hinglish:
                    _is_hinglish = True
            except Exception:
                pass
            _future = None
            
        # Final: run the full pipeline (routing + escalation + finalizer) on the float32 array
        audio = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
        
        force_esc = True if _is_hinglish else None
                
        final_text, _, _, _, _ = run_pipeline(audio, mode="auto", force_escalate=force_esc)
        
        # Stitch final_text with _committed to prevent final-step churn
        stitched_text = stitch_text(_committed, final_text)
        
        # Audit logging
        try:
            here = os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(here, "draft_audit.log"), "a", encoding="utf-8") as f:
                f.write(f"FINAL: len={len(audio_buffer)}, text={stitched_text!r} (raw_final={final_text!r})\n")
        except:
            pass
            
        _committed = stitched_text
        _last_len = 0
        _is_hinglish = False
        return (stitched_text, len(stitched_text))
        
    if len(audio_buffer) < _MIN_AUDIO_BYTES:
        return (_committed, len(_committed))

    # Check if the background job has finished
    if _future is not None and _future.done():
        try:
            text, detected_hinglish = _future.result()
            if text:
                if detected_hinglish:
                    if not _is_hinglish:
                        from solution.engine import finalize_text
                        _prev_text = finalize_text(_prev_text, is_hinglish=True)
                        _committed = finalize_text(_committed, is_hinglish=True)
                    _is_hinglish = True
                
                stable = _common_word_prefix(_prev_text, text)
                stable_words = _words(stable)
                committed_words = _words(_committed)
                
                if not _committed:
                    # Require at least 2 stable words before the first commit
                    if len(stable_words) >= 2:
                        _committed = " ".join(stable_words[:2])
                else:
                    # Extend _committed if the new stable prefix extends it
                    if len(stable_words) > len(committed_words):
                        if [w.lower() for w in stable_words[:len(committed_words)]] == [w.lower() for w in committed_words]:
                            _committed = stable
                            
                _prev_text = text
        except Exception:
            pass
        _future = None

    # Decide if we should submit a new job
    if _future is None:
        # Run background drafts continuously up to 8s of audio
        if len(audio_buffer) < 256000:
            threshold = 12800  # 0.4s of audio
            if _last_len == 0 or (len(audio_buffer) - _last_len) >= threshold:
                _last_len = len(audio_buffer)
                _future = _executor.submit(_bg_transcribe, audio_buffer, _is_hinglish)
            
    # Always reconstruct current text using _committed to ensure prefix stability
    committed_words = _words(_committed)
    new_words = _words(_prev_text)
    skip = len(committed_words)
    text = _committed + " " + " ".join(new_words[skip:])
    text = text.strip()
    
    return (text, len(_committed))

def limit_to_n_roman_tokens(text: str, n: int = 5) -> str:
    words = text.split()
    out = []
    roman_count = 0
    for w in words:
        out.append(w)
        if re.search(r"[a-zA-Z0-9']", w):
            roman_count += 1
            if roman_count >= n:
                break
    return " ".join(out)

def _common_word_prefix(left: str, right: str) -> str:
    lw, rw = _words(left), _words(right)
    out: list[str] = []
    for a, b in zip(lw, rw):
        if a.lower() != b.lower():
            break
        out.append(b)
    return " ".join(out)

def _words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9'\u0900-\u097f.-]+", text)

def _clean_word(w: str) -> str:
    return w.lower().strip(".,?!।\"'()[]{}&:-_* ")

def is_similar(w1: str, w2: str) -> bool:
    c1 = _clean_word(w1)
    c2 = _clean_word(w2)
    if not c1 or not c2:
        return False
    if c1 == c2:
        return True
    return difflib.SequenceMatcher(None, c1, c2).ratio() >= 0.6

def stitch_text(committed: str, final: str) -> str:
    if not committed:
        return final
    c_words = committed.split()
    f_words = final.split()
    if not f_words:
        return committed
        
    # Find the best matching blocks between normalized word lists
    c_clean = [_clean_word(w) for w in c_words]
    f_clean = [_clean_word(w) for w in f_words]
    
    matcher = difflib.SequenceMatcher(None, c_clean, f_clean)
    matching_blocks = matcher.get_matching_blocks()
    
    # Find the last matching block (excluding dummy) with close index alignment to prevent false jumps
    best_block = None
    for block in reversed(matching_blocks[:-1]):
        if abs(block.b - block.a) <= 3:
            best_block = block
            break
            
    if best_block:
        b_end = best_block.b + best_block.size
        c_rem = c_words[best_block.a + best_block.size:]
        
        new_b_end = b_end
        while new_b_end < len(f_words) and (new_b_end - b_end) < len(c_rem):
            f_word = f_words[new_b_end]
            matched = False
            for c_word in c_rem:
                if is_similar(f_word, c_word):
                    matched = True
                    break
            if matched:
                new_b_end += 1
            else:
                break
        return " ".join(c_words + f_words[new_b_end:])
        
    return " ".join(c_words) + " " + final
