from pathlib import Path
import zipfile
import tempfile


DEFAULT_CODEBASE = {
    "cart.py": '''def calculate_total(items):
    total = 0
    for i in range(len(items) - 1):
        total += items[i]["price"]
    return total
''',
    "tests/test_cart.py": '''from cart import calculate_total

def test_calculate_total_counts_all_items():
    items = [{"price": 10}, {"price": 20}, {"price": 30}]
    assert calculate_total(items) == 60
''',
    "README.md": "# Demo buggy cart codebase\n",
}


def parse_pasted_files(raw_text: str):
    """
    Format:
    === cart.py ===
    code here

    === tests/test_cart.py ===
    code here
    """
    files = {}
    current_name = None
    current_lines = []

    for line in raw_text.splitlines():
        if line.strip().startswith("===") and line.strip().endswith("==="):
            if current_name:
                files[current_name] = "\n".join(current_lines).strip() + "\n"

            current_name = line.strip().strip("=").strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_name:
        files[current_name] = "\n".join(current_lines).strip() + "\n"

    return files or DEFAULT_CODEBASE


def read_zip_codebase(zip_path):
    if not zip_path:
        return DEFAULT_CODEBASE

    files = {}

    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if name.endswith("/"):
                continue

            if not name.endswith((".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".json")):
                continue

            try:
                content = z.read(name).decode("utf-8", errors="ignore")
                files[name] = content
            except Exception:
                pass

    return files or DEFAULT_CODEBASE


def make_file_tree(files):
    return "\n".join(sorted(files.keys()))


def get_file_content(files, selected_file):
    if not selected_file:
        selected_file = sorted(files.keys())[0]

    return files.get(selected_file, "File not found.")


def format_codebase_for_display(files):
    return "\n\n".join(
        f"=== {name} ===\n{content}"
        for name, content in sorted(files.items())
    )
