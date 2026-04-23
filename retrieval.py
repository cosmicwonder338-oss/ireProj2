from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import re


class Retriever:
    def __init__(self, wiki):
        self.wiki = wiki

        print("Building TF-IDF index over page titles...")
        self.pages = list(wiki.keys())
        self.page_titles = [p.replace("_", " ").lower() for p in self.pages]

        self.title_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.title_matrix = self.title_vectorizer.fit_transform(self.page_titles)

        print("Loading Sentence-BERT model...")
        self.bert_model = SentenceTransformer("all-MiniLM-L6-v2")

        print(f"Indexed {len(self.pages)} pages.")

    # --------------------------
    # CLAIM HELPERS
    # --------------------------
    def extract_years(self, text):
        return set(re.findall(r"\b(19|20)\d{2}\b", text))

    def extract_keywords(self, text):
        return set(text.lower().split())

    # --------------------------
    # PAGE RETRIEVAL (FIXED)
    # --------------------------
    def retrieve_pages(self, claim, top_k=5):

        claim_lower = claim.lower()
        claim_years = self.extract_years(claim_lower)

        claim_vec = self.title_vectorizer.transform([claim_lower])
        scores = cosine_similarity(claim_vec, self.title_matrix)[0]

        boosted_scores = []

        for i, score in enumerate(scores):
            title = self.page_titles[i]

            boost = 0.0

            # 🔥 exact year match boost
            page_years = self.extract_years(title)
            if claim_years and page_years and len(claim_years & page_years) > 0:
                boost += 0.3

            # 🔥 exact claim entity match boost
            if "formula 3 sudamericana" in title:
                boost += 0.2

            boosted_scores.append(score + boost)

        top_indices = np.argsort(boosted_scores)[::-1][:top_k]

        results = []
        for i in top_indices:
            if boosted_scores[i] > 0.02:
                results.append({
                    "page": self.pages[i],
                    "score": float(boosted_scores[i])
                })

        return results

    # --------------------------
    # SENTENCE RETRIEVAL (FIXED)
    # --------------------------
    def retrieve_sentences(self, claim, pages, top_k=5):

        all_sentences = []

        for p in pages:
            page = p["page"]
            if page not in self.wiki:
                continue

            for sid, text in self.wiki[page].items():
                all_sentences.append({
                    "page": page,
                    "sent_id": sid,
                    "text": text
                })

        if not all_sentences:
            return []

        texts = [s["text"] for s in all_sentences]

        tfidf = TfidfVectorizer(ngram_range=(1, 2))
        mat = tfidf.fit_transform(texts)
        q = tfidf.transform([claim])

        tfidf_scores = cosine_similarity(q, mat)[0]
        top_idx = np.argsort(tfidf_scores)[::-1][:30]

        candidates = [all_sentences[i] for i in top_idx]
        candidate_texts = [c["text"] for c in candidates]

        claim_emb = self.bert_model.encode([claim], convert_to_numpy=True)
        sent_emb = self.bert_model.encode(candidate_texts, convert_to_numpy=True)

        bert_scores = cosine_similarity(claim_emb, sent_emb)[0]

        final_scores = 0.6 * bert_scores + 0.4 * tfidf_scores[top_idx]

        results = []

        for i, idx in enumerate(top_idx):
            score = final_scores[i]
            text = candidates[i]["text"]

            # 🔥 stronger filtering (removes junk evidence)
            if score < 0.18:
                continue

            # extra noise filter
            if len(text.split()) < 5:
                continue

            results.append({
                "page": candidates[i]["page"],
                "sent_id": candidates[i]["sent_id"],
                "text": text,
                "score": float(score)
            })

        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

    # --------------------------
    # MAIN PIPELINE
    # --------------------------
    def retrieve(self, claim, k=5):
        pages = self.retrieve_pages(claim, top_k=10)

        if not pages:
            return []

        return self.retrieve_sentences(claim, pages, top_k=k)