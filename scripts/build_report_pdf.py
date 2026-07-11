"""Converts report/report.md to a styled report/report.html.

report.html is then printed to report/report.pdf separately (via a headless
Chromium container — see the accompanying instructions in this repo's README
or run scripts/build_report_pdf.sh which does both steps), since installing
a system PDF engine (pandoc/wkhtmltopdf) directly would need sudo in a
sandboxed dev environment.

Usage:
    python scripts/build_report_pdf.py
"""

from pathlib import Path

import markdown

REPORT_DIR = Path(__file__).resolve().parents[1] / "report"
MD_PATH = REPORT_DIR / "report.md"
HTML_PATH = REPORT_DIR / "report.html"

CSS = """
@page { size: A4; margin: 20mm 18mm; }
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
       color: #1a1a1a; line-height: 1.5; font-size: 11pt; max-width: 800px; margin: 0 auto; }
h1 { font-size: 20pt; border-bottom: 3px solid #2c3e50; padding-bottom: 6px; }
h2 { font-size: 15pt; margin-top: 28px; border-bottom: 1px solid #ccc; padding-bottom: 4px;
     page-break-after: avoid; }
h3 { font-size: 12.5pt; page-break-after: avoid; }
p, li { orphans: 3; widows: 3; }
img { max-width: 100%; display: block; margin: 12px auto; page-break-inside: avoid;
      border: 1px solid #ddd; }
pre { background: #f5f5f5; border: 1px solid #ddd; border-radius: 4px; padding: 10px;
      font-size: 9pt; overflow-x: auto; page-break-inside: avoid; }
code { background: #f0f0f0; padding: 1px 4px; border-radius: 3px; font-size: 9.5pt; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 9.5pt; }
th, td { border: 1px solid #bbb; padding: 5px 8px; text-align: left; }
th { background: #f0f0f0; }
hr { border: none; border-top: 1px solid #ccc; margin: 20px 0; }
blockquote { border-left: 3px solid #ccc; margin: 10px 0; padding: 4px 14px; color: #555; }
"""

HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Heart Disease Risk Classifier — MLOps Report</title>
<style>{css}</style>
</head>
<body>
{body}
</body>
</html>
"""


def main() -> None:
    md_text = MD_PATH.read_text()
    body_html = markdown.markdown(
        md_text, extensions=["tables", "fenced_code", "sane_lists", "toc"]
    )
    html = HTML_TEMPLATE.format(css=CSS, body=body_html)
    HTML_PATH.write_text(html)
    print(f"Wrote {HTML_PATH}")


if __name__ == "__main__":
    main()
