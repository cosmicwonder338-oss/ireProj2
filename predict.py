import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification ,DistilBertForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("saved_model")
#model = BertForSequenceClassification.from_pretrained("saved_model").to(device)
model = DistilBertForSequenceClassification.from_pretrained("saved_model").to(device)

labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
model.eval()


# --------------------------
# SINGLE PAIR PREDICTION
# --------------------------
def predict(claim, evidence):

    if not evidence or not evidence.strip():
        evidence = "no evidence"

    inputs = tokenizer(
        claim,
        evidence,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = F.softmax(logits, dim=1)[0]

    return {
        "label": labels[torch.argmax(probs).item()],
        "scores": {
            "SUPPORTS": float(probs[0] * 100),
            "REFUTES": float(probs[1] * 100),
            "NOT ENOUGH INFO": float(probs[2] * 100),
        },
        "logits": logits[0]  # 🔥 important for aggregation
    }


# --------------------------
# FINAL AGGREGATION (FIXED)
# --------------------------
def predict_with_evidence_list(claim, evidence_list):

    if not evidence_list:
        r = predict(claim, "no evidence")
        r["used_evidence"] = "no evidence"
        return r

    best_result = None
    best_conf = -1

    # aggregate scores (optional reporting only)
    sum_scores = {
        "SUPPORTS": 0,
        "REFUTES": 0,
        "NOT ENOUGH INFO": 0
    }

    for ev in evidence_list:
        result = predict(claim, ev)
        scores = result["scores"]

        # accumulate for UI only (NOT decision)
        for k in sum_scores:
            sum_scores[k] += scores[k]

        # 🔥 core fix: use max confidence evidence (NOT averaging)
        top_label = max(scores, key=scores.get)
        if scores[top_label] > best_conf:
            best_conf = scores[top_label]
            best_result = {
                "label": top_label,
                "scores": scores,
                "used_evidence": ev
            }

    # normalize UI scores (optional display only)
    n = len(evidence_list)
    avg_scores = {k: v / n for k, v in sum_scores.items()}

    return {
        "label": best_result["label"],
        "scores": avg_scores,
        "used_evidence": best_result["used_evidence"]
    }

# import torch
# import torch.nn.functional as F
# from transformers import DistilBertTokenizer, DistilBertForSequenceClassification  # ✅ FIXED

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ✅ FIX: correct tokenizer
# tokenizer = DistilBertTokenizer.from_pretrained("saved_model")
# model = DistilBertForSequenceClassification.from_pretrained("saved_model").to(device)

# labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
# model.eval()


# def predict(claim, evidence):

#     if not evidence or not evidence.strip():
#         evidence = "no evidence"

#     inputs = tokenizer(
#         claim,
#         evidence,
#         return_tensors="pt",
#         truncation=True,
#         padding=True,
#         max_length=512
#     ).to(device)

#     with torch.no_grad():
#         logits = model(**inputs).logits

#     probs = F.softmax(logits, dim=1)[0]

#     return {
#         "label": labels[torch.argmax(probs).item()],
#         "scores": {
#             "SUPPORTS": float(probs[0] * 100),
#             "REFUTES": float(probs[1] * 100),
#             "NOT ENOUGH INFO": float(probs[2] * 100),
#         },
#         "logits": logits[0]
#     }


# def predict_with_evidence_list(claim, evidence_list):

#     if not evidence_list:
#         r = predict(claim, "no evidence")
#         r["used_evidence"] = "no evidence"
#         return r

#     best_result = None
#     best_conf = -1

#     sum_scores = {
#         "SUPPORTS": 0,
#         "REFUTES": 0,
#         "NOT ENOUGH INFO": 0
#     }

#     for ev in evidence_list:
#         result = predict(claim, ev)
#         scores = result["scores"]

#         for k in sum_scores:
#             sum_scores[k] += scores[k]

#         top_label = max(scores, key=scores.get)
#         if scores[top_label] > best_conf:
#             best_conf = scores[top_label]
#             best_result = {
#                 "label": top_label,
#                 "scores": scores,
#                 "used_evidence": ev
#             }

#     n = len(evidence_list)
#     avg_scores = {k: v / n for k, v in sum_scores.items()}

#     return {
#         "label": best_result["label"],
#         "scores": avg_scores,
#         "used_evidence": best_result["used_evidence"]
#     }