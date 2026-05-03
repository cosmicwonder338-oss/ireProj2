import json
import os


def load_all_wiki(folder_path, max_files=None):
    wiki = {}

    files = [f for f in os.listdir(folder_path) if f.endswith(".jsonl")]
    files = sorted(files)

    if max_files is not None:
        files = files[:max_files]

    print(f"Loading {len(files)} wiki files...")

    for i, file in enumerate(files):
        path = os.path.join(folder_path, file)

        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                    except:
                        continue  # skip bad JSON lines safely

                    page = item.get("id", "")
                    if not page:
                        continue

                    sentences = {}

                    for l in item.get("lines", "").split("\n"):
                        if not l.strip():
                            continue

                        parts = l.split("\t")

                        # validate format
                        if len(parts) < 2 or not parts[0].isdigit():
                            continue

                        idx = int(parts[0])
                        #text = parts[1].strip()
                        text = parts[1].strip().lower()

                        # ⚡ filter weak/noisy sentences
                        if len(text.split()) < 4:
                            continue

                        sentences[idx] = text

                    if sentences:
                        wiki[page] = sentences

        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue

        # progress log
        if (i + 1) % 50 == 0:
            print(f"  Loaded {i+1}/{len(files)} files, {len(wiki)} pages so far...")

    print(f"Done. Loaded {len(wiki)} wiki pages total.")
    return wiki