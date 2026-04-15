from __future__ import annotations

import re
from collections import Counter


_FILLER_WORDS = {
    "um", "uh", "uhh", "umm", "hmm", "hm", "er", "ah",
    "euh", "hum", "heu", "ben"
}


def normalize_spaces(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;!?])", r"\1", text)
    return text


def tokenize_loose(text: str) -> list[str]:
    return re.findall(r"[A-Za-zÀ-ÿ0-9']+", text.lower())


def lexical_diversity(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def repeated_ngram_stats(tokens: list[str], min_n: int = 2, max_n: int = 8) -> dict:
    best = {"max_repeats": 1, "ngram_len": 0, "ngram": []}
    n_tokens = len(tokens)
    if n_tokens < 4:
        return best

    for n in range(min_n, min(max_n, n_tokens // 2) + 1):
        i = 0
        while i + 2 * n <= n_tokens:
            chunk = tokens[i:i + n]
            reps = 1
            j = i + n
            while j + n <= n_tokens and tokens[j:j + n] == chunk:
                reps += 1
                j += n
            if reps > best["max_repeats"]:
                best = {"max_repeats": reps, "ngram_len": n, "ngram": chunk}
            i += 1

    return best


def tail_repetition_score(tokens: list[str], max_n: int = 8) -> dict:
    best = {"tail_repeats": 1, "tail_ngram_len": 0, "tail_ngram": []}
    n_tokens = len(tokens)
    if n_tokens < 4:
        return best

    for n in range(2, min(max_n, n_tokens // 2) + 1):
        tail = tokens[-n:]
        reps = 1
        j = n_tokens - 2 * n
        while j >= 0 and tokens[j:j + n] == tail:
            reps += 1
            j -= n
        if reps > best["tail_repeats"]:
            best = {"tail_repeats": reps, "tail_ngram_len": n, "tail_ngram": tail}

    return best


def filler_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    fillers = sum(1 for t in tokens if t in _FILLER_WORDS)
    return fillers / len(tokens)


def compute_text_features(text: str) -> dict:
    text = normalize_spaces(text)
    tokens = tokenize_loose(text)

    rep = repeated_ngram_stats(tokens)
    tail = tail_repetition_score(tokens)
    counts = Counter(tokens)
    most_common_count = counts.most_common(1)[0][1] if counts else 0

    return {
        "text": text,
        "tokens": tokens,
        "num_tokens": len(tokens),
        "lexical_diversity": lexical_diversity(tokens),
        "filler_ratio": filler_ratio(tokens),
        "most_common_token_count": most_common_count,
        "repeated_ngram_max_repeats": rep["max_repeats"],
        "repeated_ngram_len": rep["ngram_len"],
        "repeated_ngram": rep["ngram"],
        "tail_repeats": tail["tail_repeats"],
        "tail_ngram_len": tail["tail_ngram_len"],
        "tail_ngram": tail["tail_ngram"],
    }


def is_probable_hallucination_text_only(text: str, duration_s: float) -> bool:
    feats = compute_text_features(text)
    n = feats["num_tokens"]

    if n == 0:
        return True

    words_per_sec = n / max(duration_s, 0.01)

    if words_per_sec > 6.5 and n >= 8:
        return True

    if feats["repeated_ngram_len"] >= 3 and feats["repeated_ngram_max_repeats"] >= 3:
        return True

    if feats["tail_ngram_len"] >= 3 and feats["tail_repeats"] >= 3:
        return True

    if n >= 12 and feats["lexical_diversity"] < 0.35 and feats["tail_repeats"] >= 2:
        return True

    if n >= 10 and feats["most_common_token_count"] / n > 0.55:
        if feats["filler_ratio"] < 0.5:
            return True

    return False


def clean_transcript(text: str) -> str:
    return normalize_spaces(text)


def join_cleaned_segments(texts: list[str]) -> str:
    kept: list[str] = []
    prev = None

    for t in texts:
        t = normalize_spaces(t)
        if not t:
            continue
        if prev is not None and t.lower() == prev.lower():
            continue
        kept.append(t)
        prev = t

    return normalize_spaces(" ".join(kept))
