from pathlib import Path


def identify_file_type(file_name):

    ext = Path(file_name).suffix.lower()

    text_types = [
        ".md",
        ".txt",
        ".html"
    ]

    image_types = [
        ".png",
        ".jpg",
        ".jpeg"
    ]

    if ext in text_types:
        return "TEXT"

    if ext in image_types:
        return "IMAGE"

    return "UNKNOWN"