import re
from pathlib import Path

from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).parent
ARTICLES_DIR = BASE_DIR / "articles"
ARTICLES_DIR.mkdir(exist_ok=True)

SUPPORTED = {".md", ".txt", ".html"}


def _clean_html(raw: str) -> str:
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text("\n", strip=True)


def extract_text(filename: str, data: bytes) -> str:
    """Turn an uploaded md / txt / html file (as bytes) into plain text."""
    raw = data.decode("utf-8", errors="replace")
    if Path(filename).suffix.lower() == ".html":
        return _clean_html(raw)
    return raw


def load_all_articles() -> dict:
    """Return {article_name: text} for every article saved on disk."""
    articles = {}
    for path in sorted(ARTICLES_DIR.iterdir()):
        if path.suffix.lower() in SUPPORTED:
            articles[path.stem] = extract_text(path.name, path.read_bytes())
    return articles


def save_article(name: str, text: str) -> str:
    """Save an article permanently (as .md). Same name = overwrite."""
    safe = re.sub(r"[^\w\- ]", "", Path(name).stem).strip().replace(" ", "_")
    safe = safe or "article"
    (ARTICLES_DIR / f"{safe}.md").write_text(text, encoding="utf-8")
    return safe


def delete_article(name: str) -> None:
    for ext in SUPPORTED:
        path = ARTICLES_DIR / f"{name}{ext}"
        if path.exists():
            path.unlink()


def load_article(article_name: str) -> str:
    """Kept for backward compatibility with the old loader."""
    return (ARTICLES_DIR / article_name).read_text(encoding="utf-8")