"""Extract structured data from HTML content."""

import re
import sys

SAMPLE_HTML = """<html>
<head><title>Product Catalog</title></head>
<body>
  <h1>Products</h1>
  <h2>Electronics</h2>
  <table>
    <tr><th>Product</th><th>Price</th><th>Stock</th></tr>
    <tr><td>Laptop</td><td>999.99</td><td>15</td></tr>
    <tr><td>Phone</td><td>699.99</td><td>42</td></tr>
    <tr><td>Tablet</td><td>449.99</td><td>28</td></tr>
  </table>
  <h2>Links</h2>
  <a href="/laptop">View Laptop</a>
  <a href="/phone">View Phone</a>
  <a href="/tablet">View Tablet</a>
</body>
</html>"""


def extract_links(html):
  pattern = r'<a\s+href="([^"]*)">(.*?)</a>'
  for href, text in re.findall(pattern, html):
    print(f"{text} -> {href}")


def extract_headings(html):
  pattern = r"<h[1-6]>(.*?)</h[1-6]>"
  for heading in re.findall(pattern, html):
    print(heading)


def extract_table(html):
  rows = re.findall(r"<tr>(.*?)</tr>", html, re.DOTALL)
  for row in rows:
    cells = re.findall(r"<t[hd]>(.*?)</t[hd]>", row)
    print(",".join(cells))


def extract_text(html):
  clean = re.sub(r"<[^>]+>", " ", html)
  clean = re.sub(r"\s+", " ", clean).strip()
  print(clean)


def main():
  target = "links"
  for arg in sys.argv[1:]:
    if arg.startswith("target="):
      target = arg.split("=", 1)[1]

  extractors = {
      "links": extract_links,
      "headings": extract_headings,
      "table": extract_table,
      "text": extract_text,
  }

  if target not in extractors:
    print(f"Error: unknown target '{target}'")
    print(f"Available: {', '.join(extractors)}")
    sys.exit(1)

  extractors[target](SAMPLE_HTML)


if __name__ == "__main__":
  main()
