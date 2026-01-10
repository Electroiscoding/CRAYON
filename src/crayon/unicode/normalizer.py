# `unicode_normalize_nfc_optimized`

def unicode_normalize_nfc_optimized(text: str) -> str:
    """Optimized NFC normalization."""
    import unicodedata
    return unicodedata.normalize('NFC', text)
