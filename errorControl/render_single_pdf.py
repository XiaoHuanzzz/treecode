#!/usr/bin/env python3
"""Render single.md to HTML with vector mathematics for PDF conversion."""

from __future__ import annotations

import hashlib
import html
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown


ROOT = Path(__file__).resolve().parent
SOURCE = (
    Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "single.md"
)
OUTPUT = (
    Path(sys.argv[2]).resolve()
    if len(sys.argv) > 2
    else SOURCE.with_suffix(".html")
)
ASSET_DIR = OUTPUT.parent / f"_{OUTPUT.stem}_math"


def image_dimensions(image: Path, display: bool) -> tuple[int, int]:
    header = image.read_text(encoding="utf-8")[:500]
    match = re.search(
        r"width='([0-9.]+)pt' height='([0-9.]+)pt'", header
    )
    if match is None:
        return (560, 40) if display else (20, 16)
    width_pt = float(match.group(1))
    height_pt = float(match.group(2))
    if display:
        scale = min(4.0, 620.0 / max(width_pt, 1.0))
        return (
            max(1, round(width_pt * scale)),
            max(1, round(height_pt * scale)),
        )
    height_px = 16
    return (max(1, round(width_pt * height_px / height_pt)), height_px)


def render_math(expression: str, display: bool) -> Path:
    normalized = expression.strip()
    digest = hashlib.sha1(
        (("v4-display:" if display else "v4-inline:") + normalized).encode("utf-8")
    ).hexdigest()[:16]
    output = ASSET_DIR / f"math_{digest}.svg"
    if output.exists():
        return output

    math_body = (
        rf"\(\displaystyle {normalized}\)" if display else rf"\({normalized}\)"
    )
    latex_source = "\n".join(
        [
            r"\documentclass[preview,border=1pt]{standalone}",
            r"\usepackage{amsmath,amssymb,bm,mathtools}",
            r"\begin{document}",
            math_body,
            r"\end{document}",
        ]
    )
    with tempfile.TemporaryDirectory(prefix="markdown-math-", dir="/tmp") as tmp:
        temp_dir = Path(tmp)
        tex_file = temp_dir / "formula.tex"
        tex_file.write_text(latex_source, encoding="utf-8")
        latex_run = subprocess.run(
            [
                "latex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                tex_file.name,
            ],
            cwd=temp_dir,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if latex_run.returncode != 0:
            raise RuntimeError(
                "Failed to render formula:\n"
                f"{normalized}\n\n{latex_run.stdout[-4000:]}"
            )
        subprocess.run(
            [
                "dvisvgm",
                "--no-fonts",
                "--exact",
                "--bbox=min",
                "-o",
                str(output),
                "formula.dvi",
            ],
            cwd=temp_dir,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
    return output


def main() -> None:
    ASSET_DIR.mkdir(exist_ok=True)
    source = SOURCE.read_text(encoding="utf-8")
    replacements: dict[str, str] = {}
    counter = 0

    def stash_display(match: re.Match[str]) -> str:
        nonlocal counter
        token = f"MATHDISPLAYTOKEN{counter:04d}"
        counter += 1
        image = render_math(match.group(1), display=True)
        width, height = image_dimensions(image, display=True)
        replacements[token] = (
            '<div class="math-display">'
            f'<img src="{html.escape(image.name)}" '
            f'width="{width}" height="{height}" alt="'
            f'{html.escape(match.group(1).strip())}"></div>'
        )
        return f"\n\n<div>{token}</div>\n\n"

    def stash_inline(match: re.Match[str]) -> str:
        nonlocal counter
        token = f"MATHINLINETOKEN{counter:04d}"
        counter += 1
        image = render_math(match.group(1), display=False)
        width, height = image_dimensions(image, display=False)
        replacements[token] = (
            '<img class="math-inline" '
            f'src="{html.escape(image.name)}" '
            f'width="{width}" height="{height}" alt="'
            f'{html.escape(match.group(1).strip())}">'
        )
        return token

    source = re.sub(r"\\\[(.*?)\\\]", stash_display, source, flags=re.DOTALL)
    source = re.sub(r"\$\$(.*?)\$\$", stash_display, source, flags=re.DOTALL)
    source = re.sub(r"\\\((.*?)\\\)", stash_inline, source, flags=re.DOTALL)
    body = markdown.markdown(
        source,
        extensions=["extra", "tables", "fenced_code", "sane_lists"],
    )
    for token, replacement in replacements.items():
        body = body.replace(token, replacement)

    # The equation assets live beside the HTML after this relative base path.
    body = body.replace(
        'src="math_', f'src="{html.escape(ASSET_DIR.name)}/math_'
    )
    heading = re.search(r"^#\s+(.+)$", source, flags=re.MULTILINE)
    title = heading.group(1).strip() if heading else SOURCE.stem
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
  @page {{ size: A4; margin: 18mm 20mm; }}
  body {{
    font-family: "Noto Serif CJK SC", "Noto Serif CJK", serif;
    font-size: 10.5pt; line-height: 1.55; color: #202124;
    max-width: 170mm; margin: 0 auto;
  }}
  h1 {{ font-size: 21pt; text-align: center; margin: 0 0 18pt; }}
  h2 {{ font-size: 15pt; margin-top: 20pt; border-bottom: 1px solid #bbb; }}
  h3 {{ font-size: 12pt; margin-top: 15pt; }}
  p {{ margin: 6pt 0; }}
  blockquote {{
    margin: 10pt 0; padding: 5pt 12pt;
    border-left: 4px solid #6b8eaa; background: #f3f6f8;
  }}
  table {{ border-collapse: collapse; width: 100%; margin: 12pt 0; }}
  th, td {{ border: 1px solid #999; padding: 5pt 7pt; vertical-align: top; }}
  th {{ background: #edf1f4; font-weight: 600; }}
  code {{ font-family: "Noto Sans Mono CJK SC", monospace; }}
  pre {{ padding: 9pt; background: #f5f5f5; white-space: pre-wrap; }}
  hr {{ border: 0; border-top: 1px solid #aaa; margin: 18pt 0; }}
  .math-display {{ text-align: center; margin: 10pt auto; }}
  .math-display img {{ max-width: 96%; height: auto; }}
  .math-inline {{ height: 1.25em; width: auto; vertical-align: -0.28em; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""
    OUTPUT.write_text(document, encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(f"Rendered {counter} mathematical expressions into {ASSET_DIR}")


if __name__ == "__main__":
    main()
