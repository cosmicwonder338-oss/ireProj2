import json
import random


def load_fever(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    return data


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
                    texts.append(sentence)

    return texts


def label_map(label):
    if label == "SUPPORTS":
        return 0
    elif label == "REFUTES":
        return 1
    else:
        return 2


def prepare_data(data, wiki, limit=4000):
    dataset = []

    for item in data:
        claim = item.get("claim", "")
        label = label_map(item.get("label", "NOT ENOUGH INFO"))

        evidence_list = get_evidence_text(item.get("evidence", []), wiki)

        # --------------------------
        # BETTER EVIDENCE SELECTION
        # --------------------------
        if len(evidence_list) == 0:
            evidence = "no evidence"
        else:
            # ⚡ randomly pick one evidence (better generalization)
            evidence = random.choice(evidence_list)

        dataset.append((claim, evidence, label))

        if len(dataset) >= limit:
            break

    # --------------------------
    # OPTIONAL: SHUFFLE
    # --------------------------
    random.shuffle(dataset)

    return dataset