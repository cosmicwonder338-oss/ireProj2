from wiki_loader import load_all_wiki
import random

wiki = load_all_wiki("data/wiki/", max_files=200)

sentences = []

for page in wiki:
    for sent_id, text in wiki[page].items():
        # keep only clean, useful sentences
        if len(text.split()) >= 6:
            sentences.append((page, text))

print(f"Total candidate sentences: {len(sentences)}")

# pick some good ones
demo = random.sample(sentences, 20)

print("\n===== DEMO SENTENCES =====\n")
for i, (page, sent) in enumerate(demo):
    print(f"{i+1}. [{page}] {sent}")