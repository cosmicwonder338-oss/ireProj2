import json
import random

# --------------------------
# GLOBAL SENTENCE POOL (🔥 SPEED FIX)
# --------------------------
ALL_SENTENCES = []


# --------------------------
# LOAD FEVER
# --------------------------
def load_fever(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    return data


# --------------------------
# NORMALIZE TEXT
# --------------------------
def normalize(text):
    return text.lower().strip()


# --------------------------
# BUILD SENTENCE POOL (🔥 IMPORTANT)
# --------------------------
def build_sentence_pool(wiki):
    global ALL_SENTENCES

    ALL_SENTENCES = []

    for page in wiki:
        ALL_SENTENCES.extend(list(wiki[page].values()))

    print(f"Sentence pool size: {len(ALL_SENTENCES)}")


# --------------------------
# GET RANDOM SENTENCE (FAST)
# --------------------------
def get_random_sentence():
    return random.choice(ALL_SENTENCES)


# --------------------------
# GET EVIDENCE TEXT
# --------------------------
def get_evidence_text(evidence, wiki):
    texts = []

    for group in evidence:
        for ev in group:
            if len(ev) < 4:
                continue

            page = ev[2]
            sent_id = ev[3]

            if page is None or sent_id is None:
                continue

            if page in wiki:
                sentence = wiki[page].get(sent_id)
                if sentence and len(sentence.split()) >= 4:
                    texts.append(normalize(sentence))

    return texts


# --------------------------
# LABEL MAP
# --------------------------
def label_map(label):
    if label == "SUPPORTS":
        return 0
    elif label == "REFUTES":
        return 1
    else:
        return 2


# --------------------------
# PREPARE DATA (IMPROVED)
# --------------------------
def prepare_data(data, wiki, limit=4000):

    dataset = []

    for item in data:

        claim = normalize(item.get("claim", ""))
        label = label_map(item.get("label", "NOT ENOUGH INFO"))

        evidence_list = get_evidence_text(item.get("evidence", []), wiki)

        # --------------------------
        # 1. POSITIVE SAMPLE
        # --------------------------
        if len(evidence_list) == 0:
            evidence = "no evidence"
        else:
            evidence = max(evidence_list, key=len)

        dataset.append((claim, evidence, label))

        # --------------------------
        # 2. HARD NEGATIVE (REFUTES)
        # --------------------------
        if label == 0:  # SUPPORTS only

            wrong_evidence = get_random_sentence()

            if wrong_evidence != evidence:
                dataset.append((claim, normalize(wrong_evidence), 1))

        # --------------------------
        # 3. NEI NOISE
        # --------------------------
        if random.random() < 0.3:

            random_ev = get_random_sentence()

            dataset.append((claim, normalize(random_ev), 2))

        # --------------------------
        # LIMIT CONTROL
        # --------------------------
        if len(dataset) >= limit:
            break

    random.shuffle(dataset)

    return dataset