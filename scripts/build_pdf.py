"""Build manuscript.pdf from manuscript.md using Chrome headless print."""

from __future__ import annotations

import base64
import re
import subprocess
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "manuscript.md"
HTML_PATH = ROOT / "manuscript.html"
PDF_PATH = ROOT / "manuscript.pdf"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


def embed_images(markdown_text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        alt, rel = match.group(1), match.group(2)
        image_path = (ROOT / rel).resolve()
        if not image_path.is_file():
            return match.group(0)
        payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
        css_class = "equation" if image_path.name.startswith("eq_") else "figure"
        return f'![{alt}](data:image/png;base64,{payload} "{css_class}")'

    return re.sub(r"!\[([^\]]*)\]\((figures/[^)]+)\)", replace, markdown_text)


def wrap_html(body: str) -> str:
    body = body.replace('title="equation"', 'class="equation"')
    body = body.replace('title="figure"', 'class="figure"')
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>An evidence-aware atlas of Hippo/RAM–actin candidate associations</title>
<style>
  @page {{ size: letter; margin: 0.75in 0.9in; }}
  html, body {{
    font-family: Palatino, "Palatino Linotype", "Times New Roman", Times, serif;
    font-size: 10.8pt;
    line-height: 1.42;
    color: #111;
    background: white;
  }}
  h1 {{
    font-size: 18pt;
    font-weight: 700;
    line-height: 1.25;
    margin: 0 0 0.6em 0;
    text-align: center;
  }}
  h2 {{
    font-size: 13pt;
    margin: 1.4em 0 0.45em 0;
    border-bottom: 0.4pt solid #444;
    padding-bottom: 0.15em;
  }}
  h3 {{
    font-size: 11.5pt;
    font-style: italic;
    font-weight: 600;
    margin: 1.1em 0 0.35em 0;
  }}
  p {{ margin: 0 0 0.7em 0; text-align: justify; }}
  img.figure {{
    max-width: 100%;
    max-height: 3.7in;
    width: auto;
    height: auto;
    display: block;
    margin: 0.7em auto;
    page-break-inside: avoid;
  }}
  img.equation {{
    max-width: 100%;
    height: auto;
    display: block;
    margin: 0.45em auto;
    page-break-inside: avoid;
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 9.5pt;
    margin: 0.7em 0 1em 0;
    page-break-inside: avoid;
  }}
  th, td {{
    border: 0.4pt solid #555;
    padding: 0.28em 0.45em;
    text-align: left;
  }}
  th {{ background: #f3f3f3; }}
  code {{
    font-family: Menlo, Consolas, monospace;
    font-size: 9pt;
  }}
  hr {{ border: none; border-top: 0.4pt solid #ccc; margin: 1.2em 0; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def main() -> None:
    source = MD_PATH.read_text()
    body = markdown.markdown(embed_images(source), extensions=["tables", "sane_lists", "nl2br"])
    HTML_PATH.write_text(wrap_html(body))
    subprocess.run(
        [
            str(CHROME),
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--virtual-time-budget=8000",
            f"--print-to-pdf={PDF_PATH}",
            HTML_PATH.as_uri(),
        ],
        check=True,
    )
    HTML_PATH.unlink(missing_ok=True)
    print(f"wrote {PDF_PATH} ({PDF_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
