import os
import re
import json
import chromadb
from embed_utils import embed

NOTES_DIR = "data/notes"
CHUNK_WORDS = 250
MANIFEST_PATH = "./ingested_dates.json"

MONTHS = {
    "JANUARY": "01",
    "FEBRUARY": "02",
    "MARCH": "03",
    "APRIL": "04",
    "MAY": "05",
    "JUNE": "06",
    "JULY": "07",
    "AUGUST": "08",
    "SEPTEMBER": "09",
    "OCTOBER": "10",
    "NOVEMBER": "11",
    "DECEMBER": "12",
}

MONTH_HEADER = re.compile(r"^([A-Z]+)\s+(\d{4})$")
ENTRY_LINE = re.compile(r"^([A-Za-z]+)\s+(\d{1,2})\s+—\s+(.*)$")


def parse_diary(text: str):
    current_month, current_year = None, None

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        header_match = MONTH_HEADER.match(line)
        if header_match:
            month_name, year = header_match.groups()

            if month_name in MONTHS:
                current_month = MONTHS[month_name]
                current_year = year

            continue

        entry_match = ENTRY_LINE.match(line)

        if entry_match and current_month:
            _, day, entry_text = entry_match.groups()

            date_str = f"{current_year}-{current_month}-{int(day):02d}"

            yield date_str, entry_text


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


def main():
    client = chromadb.PersistentClient(path="./chroma_store")
    collection = client.get_or_create_collection("notes")

    already_ingested = load_manifest()

    # Safety check: if the manifest claims entries exist but the actual
    # vector store is empty (e.g. a stale manifest got deployed onto a
    # fresh server), the manifest is lying — ignore it and rebuild from scratch.
    if collection.count() == 0:
        already_ingested = set()

    new_count = 0

    for filename in os.listdir(NOTES_DIR):

        if not filename.endswith(".txt"):
            continue

        with open(os.path.join(NOTES_DIR, filename), "r") as f:
            text = f.read()

        for date, entry_text in parse_diary(text):

            if date in already_ingested:
                continue

            for i, chunk in enumerate(chunk_text(entry_text)):

                embedding = embed(chunk)

                chunk_id = f"{date}-{i}"

                collection.upsert(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[
                        {
                            "date": date,
                            "source": filename,
                        }
                    ],
                )

                new_count += 1

            already_ingested.add(date)

    save_manifest(already_ingested)

    print(
        f"Ingested {new_count} new chunks. "
        f"Total dates tracked: {len(already_ingested)}"
    )


if __name__ == "__main__":
    main()