def text_to_speech(text: str) -> bytes:
    """
    Convert `text` to speech bytes.
    For now this is a stub that simply returns the UTF-8 encoding.
    """
    return text.encode("utf-8")
