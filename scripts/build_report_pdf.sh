#!/usr/bin/env bash
# Builds report/report.pdf from report/report.md.
#
# Two steps: render Markdown -> styled HTML (scripts/build_report_pdf.py, pure
# Python), then print that HTML -> PDF using a headless-Chromium Docker
# container (mcr.microsoft.com/playwright/python). The container approach
# exists because installing a system PDF engine (pandoc/wkhtmltopdf/a browser)
# directly in this environment needs sudo; Docker itself doesn't.
#
# Usage: ./scripts/build_report_pdf.sh   (from repo root; needs Docker running)
set -euo pipefail
cd "$(dirname "$0")/.."

.venv/bin/pip show markdown >/dev/null 2>&1 || .venv/bin/pip install -q markdown
.venv/bin/python scripts/build_report_pdf.py

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT
cat > "$TMPDIR/make_pdf.py" <<'EOF'
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto("file:///report/report.html", wait_until="load", timeout=30000)
    page.wait_for_timeout(1000)
    page.pdf(
        path="/report/report.pdf",
        format="A4",
        margin={"top": "0mm", "bottom": "0mm", "left": "0mm", "right": "0mm"},
        print_background=True,
    )
print("Wrote report.pdf")
EOF

docker run --rm \
  -v "$TMPDIR/make_pdf.py:/make_pdf.py:ro" \
  -v "$(pwd)/report:/report" \
  -e PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
  mcr.microsoft.com/playwright/python:v1.61.0-noble \
  bash -c "pip install -q playwright==1.61.0 && python /make_pdf.py"
