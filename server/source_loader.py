from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
import io
import os
import re
import shutil
import subprocess
import tempfile
from typing import Dict, Iterable, Optional
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
import zipfile

from server.code_workspace import DEFAULT_CODEBASE, parse_pasted_files, read_zip_codebase


ALLOWED_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".html",
    ".css",
}
MAX_FILES = 180
MAX_FILE_CHARS = 40000
MAX_SITE_PAGES = 8


@dataclass
class WorkspaceSpec:
    label: str
    source_type: str
    files: Dict[str, str]
    origin: str


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.links.append(value)


def trim_files(files: Dict[str, str], *, max_files: int = MAX_FILES) -> Dict[str, str]:
    trimmed: Dict[str, str] = {}
    for idx, path in enumerate(sorted(files.keys())):
        if idx >= max_files:
            break
        content = files[path]
        trimmed[path] = content[:MAX_FILE_CHARS]
    return trimmed or DEFAULT_CODEBASE


def read_directory_codebase(root: Path) -> Dict[str, str]:
    files: Dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if len(files) >= MAX_FILES:
            break
        if not path.is_file():
            continue
        if any(part.startswith(".git") or part == "__pycache__" for part in path.parts):
            continue
        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        try:
            rel = path.relative_to(root).as_posix()
            files[rel] = path.read_text(encoding="utf-8", errors="ignore")[:MAX_FILE_CHARS]
        except Exception:
            continue
    return trim_files(files)


def clone_git_repo(url: str) -> Dict[str, str]:
    with tempfile.TemporaryDirectory(prefix="bugginhell_clone_") as tmpdir:
        target = Path(tmpdir) / "repo"
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(target)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=90,
        )
        return read_directory_codebase(target)


def load_local_path(path_str: str) -> WorkspaceSpec:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    if path.is_dir():
        return WorkspaceSpec(
            label=path.name or str(path),
            source_type="local_directory",
            files=read_directory_codebase(path),
            origin=str(path),
        )
    if path.suffix.lower() == ".zip":
        return WorkspaceSpec(
            label=path.name,
            source_type="local_zip",
            files=read_zip_codebase(str(path)),
            origin=str(path),
        )
    raise ValueError("Local path must be a directory or .zip file.")


def fetch_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": "BugginHell/1.0"})
    with urlopen(req, timeout=20) as resp:
        raw = resp.read()
    return raw.decode("utf-8", errors="ignore")


def crawl_site(url: str, *, max_pages: int = MAX_SITE_PAGES) -> Dict[str, str]:
    start = url if url.startswith(("http://", "https://")) else f"https://{url}"
    origin = urlparse(start)
    queue = [start]
    seen = set()
    files: Dict[str, str] = {}
    manifest: list[str] = []

    while queue and len(seen) < max_pages and len(files) < MAX_FILES:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        try:
            html_text = fetch_text(current)
        except Exception as exc:
            manifest.append(f"{current} -> fetch failed: {exc}")
            continue

        parsed = urlparse(current)
        page_name = parsed.path.strip("/") or "index"
        safe_name = re.sub(r"[^a-zA-Z0-9/_-]+", "_", page_name)[:120]
        file_name = f"site/{safe_name}.html"
        files[file_name] = html_text[:MAX_FILE_CHARS]
        manifest.append(f"{current} -> {file_name}")

        parser = LinkParser()
        parser.feed(html_text)
        for link in parser.links:
            absolute = urljoin(current, link)
            linked = urlparse(absolute)
            if linked.scheme not in {"http", "https"}:
                continue
            if linked.netloc != origin.netloc:
                continue
            if absolute not in seen and absolute not in queue and len(queue) + len(seen) < max_pages * 2:
                queue.append(absolute)

    files["site/_crawl_manifest.txt"] = "\n".join(manifest)[:MAX_FILE_CHARS]
    return trim_files(files)


def load_source(
    source_input: str | None = None,
    zip_file: str | None = None,
    pasted_text: str | None = None,
) -> WorkspaceSpec:
    if zip_file:
        return WorkspaceSpec(
            label=Path(str(zip_file)).name,
            source_type="uploaded_zip",
            files=read_zip_codebase(zip_file),
            origin=str(zip_file),
        )

    if source_input:
        raw = source_input.strip()
        if raw:
            maybe_path = Path(raw).expanduser()
            if maybe_path.exists():
                return load_local_path(raw)
            parsed = urlparse(raw)
            if parsed.scheme in {"http", "https"}:
                if "github.com" in parsed.netloc or raw.endswith(".git"):
                    return WorkspaceSpec(
                        label=parsed.path.strip("/").split("/")[-1] or parsed.netloc,
                        source_type="git_repo",
                        files=clone_git_repo(raw),
                        origin=raw,
                    )
                return WorkspaceSpec(
                    label=parsed.netloc,
                    source_type="website",
                    files=crawl_site(raw),
                    origin=raw,
                )

    if pasted_text and pasted_text.strip():
        return WorkspaceSpec(
            label="pasted_codebase",
            source_type="pasted_files",
            files=parse_pasted_files(pasted_text),
            origin="inline",
        )

    return WorkspaceSpec(
        label="demo_codebase",
        source_type="default",
        files=DEFAULT_CODEBASE,
        origin="built_in",
    )
