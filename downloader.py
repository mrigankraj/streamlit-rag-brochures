# downloader.py
import os
import requests
import time
import mimetypes
from pathlib import Path
from urllib.parse import urlparse, urljoin
import pandas as pd
from tqdm import tqdm
from PIL import Image
from io import BytesIO

DOWNLOAD_DIR = Path(os.environ.get("DOWNLOAD_DIR", "./downloads"))
DOWNLOAD_DIR.mkdir(exist_ok=True)

HEADERS = {"User-Agent": "RAG-Brochure-Downloader/1.0"}

def safe_filename(url, prefix=None):
    parsed = urlparse(url)
    name = Path(parsed.path).name or parsed.netloc.replace(":", "_")
    if prefix:
        return f"{prefix}__{name}"
    return name

def download_file(url, dest: Path, max_retries=3, timeout=30):
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
            if resp.status_code == 200:
                content_type = resp.headers.get("Content-Type", "").lower()
                data = resp.content

                if "pdf" in content_type or dest.suffix.lower() == ".pdf":
                    out_path = dest.with_suffix(".pdf")
                    out_path.write_bytes(data)
                    return "pdf", str(out_path)

                if "image" in content_type:
                    try:
                        image = Image.open(BytesIO(data)).convert("RGB")
                        out_path = dest.with_suffix(".pdf")
                        image.save(out_path)
                        return "image->pdf", str(out_path)
                    except Exception:
                        dest.write_bytes(data)
                        return "image_failed_raw", str(dest)

                if "html" in content_type:
                    text = resp.text
                    import re
                    m = re.search(r'href=["\']([^"\']+\.pdf)["\']', text, re.I)
                    if m:
                        pdf_url = m.group(1)
                        if not pdf_url.startswith("http"):
                            pdf_url = urljoin(resp.url, pdf_url)
                        return download_file(pdf_url, dest, max_retries=max_retries, timeout=timeout)

                guessed = mimetypes.guess_extension(content_type.split(";")[0].strip())
                if guessed and guessed in [".jpg", ".jpeg", ".png"]:
                    image = Image.open(BytesIO(data)).convert("RGB")
                    out_path = dest.with_suffix(".pdf")
                    image.save(out_path)
                    return "image->pdf", str(out_path)

                dest.write_bytes(data)
                return "raw", str(dest)
            else:
                time.sleep(1 + attempt)
        except Exception:
            time.sleep(1 + attempt)
    return None, None

def download_from_excel(excel_path: str, url_col: str='Brochure_Link', id_col: str='PSM_ID', resume=True, limit=None):
    df = pd.read_excel(excel_path)
    if limit:
        df = df.head(limit)
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        url = row.get(url_col)
        pid = row.get(id_col)
        if not isinstance(url, str) or not url.strip():
            results.append({'PSM_ID': pid, 'url': url, 'status':'empty', 'file': None})
            continue
        fname = safe_filename(url, prefix=str(pid))
        dest = DOWNLOAD_DIR / fname
        pdf_dest = dest.with_suffix(".pdf")
        if resume and (pdf_dest.exists() or dest.exists()):
            file_path = str(pdf_dest if pdf_dest.exists() else dest)
            results.append({'PSM_ID': pid, 'url': url, 'status':'skipped', 'file': file_path})
            continue
        status, file_path = download_file(url, dest)
        if status:
            results.append({'PSM_ID': pid, 'url': url, 'status': status, 'file': file_path})
        else:
            results.append({'PSM_ID': pid, 'url': url, 'status':'failed', 'file': None})
    return pd.DataFrame(results)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--excel", required=True)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    report = download_from_excel(args.excel, limit=args.limit)
    report.to_csv("download_report.csv", index=False)
    print("Download complete. See download_report.csv")
