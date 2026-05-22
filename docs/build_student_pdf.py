from pathlib import Path
import re
import markdown

ROOT = Path(__file__).resolve().parent
md_path = ROOT / "bayesian_optimization_guide.md"
html_path = ROOT / "bayesian_optimization_guide_student.html"

text = md_path.read_text(encoding="utf-8")

# Light cleanup for cleaner rendering.
text = re.sub(r"\n{3,}", "\n\n", text)

html_body = markdown.markdown(
    text,
    extensions=["fenced_code", "tables", "toc", "sane_lists"],
)

workflow_svg = """
<svg viewBox=\"0 0 980 140\" width=\"100%\" height=\"140\" role=\"img\" aria-label=\"Bayesian optimization loop\">
  <defs>
    <marker id=\"arrow\" markerWidth=\"10\" markerHeight=\"8\" refX=\"9\" refY=\"4\" orient=\"auto\">
      <polygon points=\"0 0, 10 4, 0 8\" fill=\"#29524a\"/>
    </marker>
  </defs>
  <rect x=\"20\" y=\"24\" rx=\"12\" ry=\"12\" width=\"200\" height=\"84\" fill=\"#edf6f2\" stroke=\"#6f8f85\"/>
  <text x=\"120\" y=\"62\" text-anchor=\"middle\" font-size=\"15\" fill=\"#1f3f39\" font-weight=\"600\">Ask Ax for trial</text>
  <text x=\"120\" y=\"84\" text-anchor=\"middle\" font-size=\"13\" fill=\"#335f56\">(suggest conditions)</text>

  <rect x=\"270\" y=\"24\" rx=\"12\" ry=\"12\" width=\"200\" height=\"84\" fill=\"#edf6f2\" stroke=\"#6f8f85\"/>
  <text x=\"370\" y=\"62\" text-anchor=\"middle\" font-size=\"15\" fill=\"#1f3f39\" font-weight=\"600\">Run experiment</text>
  <text x=\"370\" y=\"84\" text-anchor=\"middle\" font-size=\"13\" fill=\"#335f56\">(measure outcomes)</text>

  <rect x=\"520\" y=\"24\" rx=\"12\" ry=\"12\" width=\"200\" height=\"84\" fill=\"#edf6f2\" stroke=\"#6f8f85\"/>
  <text x=\"620\" y=\"62\" text-anchor=\"middle\" font-size=\"15\" fill=\"#1f3f39\" font-weight=\"600\">Tell Ax result</text>
  <text x=\"620\" y=\"84\" text-anchor=\"middle\" font-size=\"13\" fill=\"#335f56\">(update model)</text>

  <rect x=\"770\" y=\"24\" rx=\"12\" ry=\"12\" width=\"190\" height=\"84\" fill=\"#edf6f2\" stroke=\"#6f8f85\"/>
  <text x=\"865\" y=\"62\" text-anchor=\"middle\" font-size=\"15\" fill=\"#1f3f39\" font-weight=\"600\">Repeat until</text>
  <text x=\"865\" y=\"84\" text-anchor=\"middle\" font-size=\"13\" fill=\"#335f56\">stopping criterion</text>

  <line x1=\"220\" y1=\"66\" x2=\"268\" y2=\"66\" stroke=\"#29524a\" stroke-width=\"2.4\" marker-end=\"url(#arrow)\"/>
  <line x1=\"470\" y1=\"66\" x2=\"518\" y2=\"66\" stroke=\"#29524a\" stroke-width=\"2.4\" marker-end=\"url(#arrow)\"/>
  <line x1=\"720\" y1=\"66\" x2=\"768\" y2=\"66\" stroke=\"#29524a\" stroke-width=\"2.4\" marker-end=\"url(#arrow)\"/>
</svg>
"""

html = f"""
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Bayesian Optimization with Ax - Student Guide</title>
  <style>
    @page {{
      size: Letter;
      margin: 0.72in;
    }}
    html, body {{
      color: #13221f;
      background: #ffffff;
      font-family: "Segoe UI", "Aptos", "Calibri", sans-serif;
      line-height: 1.52;
      font-size: 11.2pt;
    }}
    .cover {{
      border: 2px solid #c8d8d4;
      border-radius: 12px;
      padding: 18px 20px 14px;
      margin: 0 0 16px;
      background: #f3f9f7;
    }}
    .cover h1 {{
      margin: 0 0 6px;
      color: #133f37;
      font-size: 21pt;
      line-height: 1.2;
    }}
    .cover p {{
      margin: 0;
      color: #2f5550;
      font-size: 11pt;
    }}
    .figure {{
      margin: 10px 0 4px;
      border: 1px solid #c8d8d4;
      border-radius: 10px;
      background: #f8fcfb;
      padding: 10px 12px 4px;
      page-break-inside: avoid;
    }}
    h1, h2, h3 {{
      page-break-after: avoid;
      break-after: avoid;
      margin-top: 1.08em;
      margin-bottom: 0.45em;
      line-height: 1.26;
      color: #113a33;
    }}
    h1 {{ font-size: 20pt; border-bottom: 1px solid #c8d8d4; padding-bottom: 8px; }}
    h2 {{ font-size: 15pt; border-left: 4px solid #9bc5bb; padding-left: 10px; }}
    h3 {{ font-size: 12.6pt; color: #1f564c; }}
    p {{ margin: 0.5em 0; }}
    ul, ol {{ margin: 0.45em 0 0.72em 1.2em; }}
    li {{ margin: 0.2em 0; }}
    pre, code {{ font-family: "Consolas", "Cascadia Code", monospace; }}
    pre {{
      background: #f4f7f7;
      border: 1px solid #d8e4e2;
      border-radius: 8px;
      padding: 10px 12px;
      overflow-wrap: anywhere;
      white-space: pre-wrap;
      page-break-inside: avoid;
      font-size: 9.7pt;
    }}
    code {{
      background: #f2f6f5;
      border: 1px solid #e1ebea;
      border-radius: 4px;
      padding: 1px 4px;
      font-size: 0.94em;
    }}
    pre code {{ border: none; background: transparent; padding: 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 0.6em 0 0.9em; font-size: 10.2pt; }}
    th, td {{ border: 1px solid #ccd9d6; padding: 6px 7px; text-align: left; vertical-align: top; }}
    th {{ background: #eef5f3; font-weight: 700; }}
    blockquote {{
      margin: 0.8em 0;
      border-left: 4px solid #9bc5bb;
      padding: 7px 12px;
      background: #f2faf6;
      color: #244944;
    }}
    hr {{ border: 0; border-top: 1px solid #c8d8d4; margin: 1.2em 0; }}
    a {{ color: #1f6f5f; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .emoji-note {{
      margin: 12px 0 8px;
      padding: 8px 10px;
      border: 1px dashed #aaccc4;
      border-radius: 8px;
      background: #f7fcfa;
      color: #2a4c46;
      font-size: 10.8pt;
      page-break-inside: avoid;
    }}
  </style>
</head>
<body>
  <div class=\"cover\">
    <h1>Bayesian Optimization with Ax: Student Guide</h1>
    <p>Readable practical guide for chemistry experiments with categorical and mixture parameters.</p>
    <div class=\"figure\">{workflow_svg}</div>
    <div class=\"emoji-note\">🧪 Tip: Read Section 6 and Section 14 first if you are designing your first campaign.</div>
  </div>
  {html_body}
</body>
</html>
"""

html_path.write_text(html, encoding="utf-8")
print(f"Wrote {html_path}")
