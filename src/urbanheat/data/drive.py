"""Read-only access to a Google Drive folder shared as 'Anyone with the link'.

Listing uses the embedded folder view (no 50-item cap, no API key). Files are fetched through the
drive.usercontent direct URL, which works for files gdown fails on.
"""
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import requests

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
_ENTRY = re.compile(
    r'href="https://drive\.google\.com/(drive/folders|file/d)/([A-Za-z0-9_-]+)[^"]*".*?'
    r'flip-entry-title">([^<]*)<.*?flip-entry-last-modified"><div>([^<]*)<',
    re.S,
)


@dataclass(frozen=True)
class DriveEntry:
    id: str
    path: str          # relative to the listed root, "/"-separated
    is_folder: bool
    modified: str      # as Drive shows it, e.g. "Oct 24"

    @property
    def name(self):
        return self.path.rsplit("/", 1)[-1]


def _session():
    s = requests.Session()
    s.headers["User-Agent"] = _UA
    return s


def parse_folder_html(html):
    """(id, name, is_folder, modified) for every child in an embedded folder view page."""
    return [(i, n, k == "drive/folders", m) for k, i, n, m in _ENTRY.findall(html)]


def list_folder(folder_id, session=None):
    s = session or _session()
    r = s.get("https://drive.google.com/embeddedfolderview?id=" + folder_id, timeout=60)
    r.raise_for_status()
    return parse_folder_html(r.text)


def walk(folder_id, prefix="", skip=lambda path: False, workers=8, session=None):
    """All entries below `folder_id`, recursively. `skip(path)` prunes a folder and everything in it."""
    s = session or _session()
    out = []

    def visit(fid, base):
        sub = []
        for i, name, is_folder, modified in list_folder(fid, s):
            path = f"{base}/{name}" if base else name
            if skip(path):
                continue
            out.append(DriveEntry(i, path, is_folder, modified))
            if is_folder:
                sub.append((i, path))
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(lambda a: visit(*a), sub))

    visit(folder_id, prefix)
    return sorted(out, key=lambda e: e.path)


def download(file_id, dest, tries=5, session=None):
    """Download one file to `dest`. Returns 'skip' if it already exists, 'ok', or 'FAIL'."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    s = session or _session()
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    for k in range(tries):
        try:
            with s.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            with open(tmp, "rb") as fh:
                if b"<html" in fh.read(200).lower():
                    raise RuntimeError("got an HTML page instead of the file (Drive quota or permission)")
            tmp.replace(dest)
            return "ok"
        except Exception:
            if k == tries - 1:
                tmp.unlink(missing_ok=True)
                return "FAIL"
            time.sleep(10 * (k + 1))
