import os
import json
import datetime
import chromadb

from embed_utils import embed
from github_sync import commit_file

DIARY_PATH = "data/notes/diary_2026_daily_paragraphs.txt"
MANIFEST_PATH = "./ingested_dates.json"
CHUNK_WORDS = 250


def chunk_text(text: str, chunk_size: int = CHUNK_WORDS):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])


def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return set(json.load(f))
    return set()


def save_manifest(dates):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(sorted(dates), f)


def append_to_diary_file(entry_date: datetime.date, entry_text: str):
    """Appends one entry to the diary .txt in the same MONTH header + 'Month Day — text' format."""
    month_name_upper = entry_date.strftime("%B").upper()
    header_line = f"{month_name_upper} {entry_date.year}"
    entry_line = f"{entry_date.strftime('%B')} {entry_date.day} — {entry_text.strip()}"

    header_exists = False

    if os.path.exists(DIARY_PATH):
        with open(DIARY_PATH, "r") as f:
            header_exists = any(line.strip() == header_line for line in f)

    with open(DIARY_PATH, "a") as f:
        if not header_exists:
            f.write(f"\n{header_line}\n\n{entry_line}\n")
        else:
            f.write(f"\n{entry_line}\n")


def ingest_single_entry(entry_date: datetime.date, entry_text: str):
    """Embeds just this one new entry and adds it to the vector store immediately."""

    date_str = entry_date.strftime("%Y-%m-%d")

    client = chromadb.PersistentClient(path="./chroma_store")
    collection = client.get_or_create_collection("notes")

    for i, chunk in enumerate(chunk_text(entry_text)):
        embedding = embed(chunk)

        chunk_id = f"{date_str}-{i}"

        collection.upsert(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[
                {
                    "date": date_str,
                    "source": os.path.basename(DIARY_PATH),
                }
            ],
        )

    manifest = load_manifest()
    manifest.add(date_str)
    save_manifest(manifest)


def save_diary_entry(entry_date: datetime.date, entry_text: str):
    """Writes the diary entry, embeds it, and commits it to GitHub."""

    append_to_diary_file(entry_date, entry_text)

    ingest_single_entry(entry_date, entry_text)

    with open(DIARY_PATH, "r") as f:
        full_content = f.read()

    commit_file(
        DIARY_PATH,
        full_content,
        f"Add entry for {entry_date}"
    )