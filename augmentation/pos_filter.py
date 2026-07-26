from typing import List, Sequence, Tuple


TaggedToken = Tuple[str, str]


def _dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _is_noun_tag(tag: str) -> bool:
    return tag.startswith("N")


def _extract_from_tagged(tagged: Sequence[TaggedToken], include_phrases: bool) -> List[str]:
    candidates = []
    noun_run = []

    def flush_run():
        if include_phrases and len(noun_run) >= 2:
            candidates.append(" ".join(noun_run))
        noun_run.clear()

    for token, tag in tagged:
        token = token.strip()
        if token and _is_noun_tag(tag):
            candidates.append(token)
            noun_run.append(token)
        else:
            flush_run()
    flush_run()

    return _dedupe_preserve_order(candidates)


def extract_nouns(text: str) -> List[str]:
    from underthesea import pos_tag

    return _extract_from_tagged(pos_tag(text), include_phrases=False)


def extract_noun_candidates(text: str) -> List[str]:
    from underthesea import pos_tag

    return _extract_from_tagged(pos_tag(text), include_phrases=True)
