"""
Stage 2: Cleaning
------------------
Strip navigation, headers, footers, scripts, and repeated boilerplate from
raw HTML, leaving the substantive page content.

Primary method: trafilatura (built for exactly this: main-content extraction
across inconsistent page templates). Falls back to a density-based
BeautifulSoup heuristic if trafilatura is not installed, so the pipeline
still runs end-to-end.

Also de-duplicates boilerplate strings (e.g. cookie notices, site-wide
footers) that recur across many pages -- these dilute the embedding space
if left in.

Usage:
    python clean.py --raw data/raw --out data/clean
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup

try:
    import trafilatura
    HAVE_TRAFILATURA = True
except ImportError:
    HAVE_TRAFILATURA = False

# Site-wide widgets that appear near the end of nearly every GITAM page
# (an "Upcoming Events" block with per-page dates/rooms, and a search
# overlay) -- generic noise, not department content. Add more markers here
# if you spot other recurring trailing widgets on other page types.
TRAILING_WIDGET_MARKERS = [
    "Upcoming Events",
    "What are you searching for?",
]


def clean_with_trafilatura(html: str) -> str | None:
    # favor_recall (not favor_precision): template-heavy pages built from many
    # short list/heading blocks (course lists, faculty cards, stat tables --
    # as seen on GITAM's discipline pages) get misclassified as boilerplate
    # under precision mode, which silently drops most of the real content.
    # Recall mode keeps more, at the cost of occasionally letting a little
    # extra noise through -- the better trade-off for this corpus, where
    # under-extraction (losing real content) is the worse failure.
    return trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        favor_recall=True,
    )


def clean_with_bs4_fallback(html: str) -> str:
    """Strip obvious chrome, then keep ALL of <main>'s content (not just the
    single densest sub-region). GITAM's pages put course lists, faculty
    cards, and stat tables as siblings inside <main> alongside a small
    <article>-wrapped intro paragraph; picking only the densest single
    candidate silently dropped everything else. Taking the whole <main>
    (after chrome removal) keeps it all -- consistent with favouring
    recall over precision at this stage."""
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "nav", "header", "footer", "form",
                     "noscript", "svg", "aside"]):
        tag.decompose()

    # Drop common chrome by class/id keywords (menus, cookie banners, social widgets)
    noise_pattern = re.compile(
        r"(nav|menu|footer|header|sidebar|cookie|banner|social|breadcrumb|share)",
        re.IGNORECASE,
    )
    for tag in soup.find_all(attrs={"class": noise_pattern}):
        tag.decompose()
    for tag in soup.find_all(attrs={"id": noise_pattern}):
        tag.decompose()

    main = soup.find("main") or soup.body or soup
    return main.get_text(separator="\n", strip=True)


def normalise_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip broken markdown table separator rows (e.g. "|---|---|---|")
    # left behind by trafilatura's table-to-text conversion -- pure
    # formatting artifact, never real content.
    text = re.sub(r"^\|[-\s|]+\|$", "", text, flags=re.MULTILINE)
    return text.strip()


def find_repeated_boilerplate(all_texts: list[str], min_len: int = 40,
                               min_pages: int = 5) -> set[str]:
    """Lines that appear near-identically across many pages are almost
    certainly template chrome (cookie notices, contact blurbs, etc.), not
    genuine content -- flag them for removal regardless of which cleaner
    produced them."""
    line_counts = Counter()
    for text in all_texts:
        lines = {ln.strip() for ln in text.splitlines() if len(ln.strip()) >= min_len}
        line_counts.update(lines)
    return {line for line, count in line_counts.items() if count >= min_pages}


def main():
    ap = argparse.ArgumentParser(description="Clean raw HTML pages into plain text.")
    ap.add_argument("--raw", default="data/raw", help="Directory of raw page JSON files")
    ap.add_argument("--out", default="data/clean", help="Output directory for cleaned text")
    ap.add_argument("--min-boilerplate-pages", type=int, default=5,
                     help="A line appearing on at least this many pages is treated as boilerplate")
    args = ap.parse_args()

    raw_dir, out_dir = Path(args.raw), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cleaner backend: {'trafilatura' if HAVE_TRAFILATURA else 'BeautifulSoup fallback'}")

    records = []
    for f in sorted(raw_dir.glob("*.json")):
        if f.name == "_manifest.json":
            continue
        rec = json.loads(f.read_text(encoding="utf-8"))
        if not rec.get("html"):
            continue
        records.append((f, rec))

    # Pass 1: extract main content per page.
    # Run BOTH extractors and keep whichever yields more text. Trafilatura is
    # built for long-form article prose and under-extracts componentized
    # "app-like" pages (course chip lists, faculty cards, stat tables) --
    # the bs4 density fallback often captures those better since it just
    # keeps everything under <main>/<article> after stripping chrome. Since
    # this corpus favours recall over precision at this stage, take the max.
    extracted = []
    method_counts = {"trafilatura": 0, "bs4_fallback": 0}
    for f, rec in records:
        html = rec["html"]
        text_traf = clean_with_trafilatura(html) if HAVE_TRAFILATURA else None
        text_traf = normalise_whitespace(text_traf or "")
        text_bs4 = normalise_whitespace(clean_with_bs4_fallback(html))

        if len(text_bs4) > len(text_traf):
            text, method = text_bs4, "bs4_fallback"
        else:
            text, method = text_traf, "trafilatura"
        method_counts[method] += 1

        extracted.append((f, rec, text))

    print(f"Extraction method used: {method_counts['trafilatura']} pages via trafilatura, "
          f"{method_counts['bs4_fallback']} pages via bs4 fallback (longer result won)")

    # Pass 2: find and strip cross-page boilerplate lines
    boilerplate = find_repeated_boilerplate(
        [t for _, _, t in extracted], min_pages=args.min_boilerplate_pages
    )
    print(f"Flagged {len(boilerplate)} boilerplate lines shared across "
          f">= {args.min_boilerplate_pages} pages.")

    written = 0
    for f, rec, text in extracted:
        lines = [ln for ln in text.splitlines() if ln.strip() not in boilerplate]
        clean_text = normalise_whitespace("\n".join(lines))

        # Truncate trailing site-wide widgets (Upcoming Events, search box).
        # These recur on nearly every page but aren't caught by the exact-line
        # boilerplate dedup above, because the surrounding dates/room numbers
        # differ per page -- so the widget as a whole never matches verbatim
        # across pages even though it's clearly generic, not department content.
        for marker in TRAILING_WIDGET_MARKERS:
            idx = clean_text.find(marker)
            if idx != -1:
                clean_text = clean_text[:idx].rstrip()

        if len(clean_text) < 50:
            print(f"  [warn] near-empty after cleaning ({len(clean_text)} chars): {rec['url']}")
            continue

        out_rec = {
            "url": rec["url"],
            "fetched_at_utc": rec["fetched_at_utc"],
            "clean_text": clean_text,
            "clean_text_length": len(clean_text),
        }
        (out_dir / f.name).write_text(json.dumps(out_rec, ensure_ascii=False), encoding="utf-8")
        written += 1

    print(f"\nDone. Cleaned {written}/{len(records)} pages -> {out_dir}")


if __name__ == "__main__":
    main()
