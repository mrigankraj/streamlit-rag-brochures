import os, requests, time, mimetypes
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd
from tqdm import tqdm
from PIL import Image
from io import BytesIO

DOWNLOAD_DIR = Path("downloads"); DOWNLOAD_DIR.mkdir(exist_ok=True)
HEADERS = {"User-Agent": "RAG-Brochure-Downloader/1.0"}

def safe_filename(url, prefix=None):
    parsed = urlparse(url)
    name = os.path.basename(parsed.path) or parsed.netloc
    return f"{prefix}__{name}" if prefix else name

def download_file(url, dest: Path, max_retries=3, timeout=30):
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            if resp.status_code == 200:
                ctype = resp.headers.get('Content-Type','')
                data = resp.content
                if 'pdf' in ctype or dest.suffix.lower()=='.pdf':
                    dest.write_bytes(data); return 'pdf'
                if 'image' in ctype:
                    Image.open(BytesIO(data)).convert('RGB').save(dest.with_suffix('.pdf'))
                    return 'image->pdf'
                guessed = mimetypes.guess_extension(ctype.split(';')[0].strip())
                if guessed in ['.jpg','.jpeg','.png']:
                    Image.open(BytesIO(data)).convert('RGB').save(dest.with_suffix('.pdf'))
                    return 'image->pdf'
                dest.write_bytes(data); return 'raw'
        except Exception: time.sleep(1+attempt)
    return None

def download_from_excel(excel_path, url_col='Brochure_Link', id_col='PSM_ID', limit=None):
    df = pd.read_excel(excel_path)
    if limit: df = df.head(limit)
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        url, pid = row.get(url_col), row.get(id_col)
        if not isinstance(url, str) or not url.strip():
            results.append({'PSM_ID': pid,'url':url,'status':'empty'}); continue
        fname = safe_filename(url, prefix=str(pid))
        dest = DOWNLOAD_DIR / fname
        if dest.exists() or dest.with_suffix('.pdf').exists():
            results.append({'PSM_ID': pid,'url':url,'status':'skipped','file':str(dest)}); continue
        status = download_file(url, dest)
        if status:
            file_path = dest if status in ['pdf','raw'] else dest.with_suffix('.pdf')
            results.append({'PSM_ID': pid,'url':url,'status':status,'file':str(file_path)})
        else:
            results.append({'PSM_ID': pid,'url':url,'status':'failed'})
    return pd.DataFrame(results)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--excel', required=True)
    p.add_argument('--limit', type=int, default=None)
    args = p.parse_args()
    res = download_from_excel(args.excel, limit=args.limit)
    res.to_csv('download_report.csv', index=False)
    print("✅ Done. Report → download_report.csv")
