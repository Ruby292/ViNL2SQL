import string
from collections import Counter
from typing import Dict, List, Optional, Sequence, Set, Tuple


TaggedToken = Tuple[str, str]
VIETNAMESE_STOPWORDS = {
    "hiển thị",
    "liệt kê",
    "cho biết",
    "vui lòng",
    "trả về",
    "tìm",
    "đếm",
    "sắp xếp",
    "số lượng",
    "tổng số",
    "số",
    "tổng",
    "tổng cộng",
    "giá trị",
    "mức",
    "lượt",
    "thứ tự",
    "khoảng",
    "lần",
    "tên",
    "loại",
    "bộ",
    "phần",
    "chiếc",
    "việc",
    "sức",
    "cấp",
    "con",
    "năm",
}
VIETNAMESE_DIACRITICS = set(
    "ăâđêôơư"
    "áàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệ"
    "íìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữự"
    "ýỳỷỹỵ"
    "ĂÂĐÊÔƠƯ"
    "ÁÀẢÃẠẮẰẲẴẶẤẦẨẪẬÉÈẺẼẸẾỀỂỄỆ"
    "ÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰ"
    "ÝỲỶỸỴ"
)


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


def _normalized_schema_name(name: str) -> str:
    return " ".join(name.replace("_", " ").lower().split())


def _has_vietnamese_diacritic(text: str) -> bool:
    return any(char in VIETNAMESE_DIACRITICS for char in text)


def _ascii_letter_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    ascii_letters = [char for char in letters if char in string.ascii_letters]
    return len(ascii_letters) / len(letters)


def is_stopword(noun: str) -> bool:
    """Return True for generic Vietnamese noun/action terms."""
    return noun.lower().strip() in VIETNAMESE_STOPWORDS


def _filter_reason(noun: str, schema_names: Set[str]) -> str:
    stripped = noun.strip()
    length = len(stripped)
    if length < 2:
        return "too_short_removed"
    if length > 30:
        return "too_long_removed"
    if is_stopword(stripped):
        return "vietnamese_stopwords_removed"

    normalized = _normalized_schema_name(stripped)
    schema_name_variants = {
        name.lower()
        for name in schema_names
    } | {
        _normalized_schema_name(name)
        for name in schema_names
    }
    if normalized in schema_name_variants or stripped.lower() in schema_name_variants:
        return "schema_name_duplicates_removed"

    if _ascii_letter_ratio(stripped) > 0.70 and not _has_vietnamese_diacritic(stripped):
        return "english_nouns_removed"

    return ""


def is_vietnamese_noun(noun: str, schema_names: Optional[Set[str]] = None) -> bool:
    """Return True when a noun candidate survives the augmentation filters."""
    return _filter_reason(noun, schema_names or set()) == ""


def filter_nouns_with_stats(nouns: List[str], schema_names: Set[str]) -> Tuple[List[str], Dict[str, int]]:
    """Filter noun candidates and return removal counters for stats."""
    stats = Counter(
        {
            "total_nouns_before_filter": len(nouns),
            "english_nouns_removed": 0,
            "vietnamese_stopwords_removed": 0,
            "schema_name_duplicates_removed": 0,
            "too_short_removed": 0,
            "too_long_removed": 0,
            "nouns_after_filter": 0,
        }
    )
    removed_stopwords = Counter()
    kept = []
    for noun in nouns:
        reason = _filter_reason(noun, schema_names)
        if reason:
            stats[reason] += 1
            if reason == "vietnamese_stopwords_removed":
                removed_stopwords[noun.lower().strip()] += 1
            continue
        kept.append(noun)

    kept = _dedupe_preserve_order(kept)
    stats["nouns_after_filter"] = len(kept)
    result = dict(stats)
    result["top_removed_stopwords"] = dict(removed_stopwords.most_common(10))
    return kept, result


def filter_nouns(nouns: List[str], schema_names: Set[str]) -> List[str]:
    """Lọc danh từ trước khi đưa vào embedding.

    Args:
        nouns: danh sách danh từ/cụm danh từ từ POS tagger
        schema_names: set tên bảng + tên cột (lowercase) của db hiện tại
    Returns:
        danh sách đã lọc, chỉ giữ danh từ tiếng Việt có giá trị
    """
    kept, _stats = filter_nouns_with_stats(nouns, schema_names)
    return kept
