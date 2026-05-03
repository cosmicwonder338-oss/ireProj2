# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import re


# class Retriever:
#     def __init__(self, wiki):
#         self.wiki = wiki

#         print("Building TF-IDF index over page titles...")
#         self.pages = list(wiki.keys())
#         self.page_titles = [p.replace("_", " ").lower() for p in self.pages]

#         self.title_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
#         self.title_matrix = self.title_vectorizer.fit_transform(self.page_titles)

#         print("Loading Sentence-BERT model...")
#         self.bert_model = SentenceTransformer("all-MiniLM-L6-v2")

#         print(f"Indexed {len(self.pages)} pages.")

#     # --------------------------
#     # CLAIM HELPERS
#     # --------------------------
#     def extract_years(self, text):
#         return set(re.findall(r"\b(19|20)\d{2}\b", text))

#     def extract_keywords(self, text):
#         return set(text.lower().split())

#     # --------------------------
#     # PAGE RETRIEVAL (FIXED)
#     # --------------------------
#     def retrieve_pages(self, claim, top_k=5):

#         claim_lower = claim.lower()
#         claim_years = self.extract_years(claim_lower)

#         claim_vec = self.title_vectorizer.transform([claim_lower])
#         scores = cosine_similarity(claim_vec, self.title_matrix)[0]

#         boosted_scores = []

#         for i, score in enumerate(scores):
#             title = self.page_titles[i]

#             boost = 0.0

#             # 🔥 exact year match boost
#             page_years = self.extract_years(title)
#             if claim_years and page_years and len(claim_years & page_years) > 0:
#                 boost += 0.3

#             # 🔥 exact claim entity match boost
#             if "formula 3 sudamericana" in title:
#                 boost += 0.2

#             boosted_scores.append(score + boost)

#         top_indices = np.argsort(boosted_scores)[::-1][:top_k]

#         results = []
#         for i in top_indices:
#             if boosted_scores[i] > 0.02:
#                 results.append({
#                     "page": self.pages[i],
#                     "score": float(boosted_scores[i])
#                 })

#         return results

#     # --------------------------
#     # SENTENCE RETRIEVAL (FIXED)
#     # --------------------------
#     def retrieve_sentences(self, claim, pages, top_k=5):

#         all_sentences = []

#         for p in pages:
#             page = p["page"]
#             if page not in self.wiki:
#                 continue

#             for sid, text in self.wiki[page].items():
#                 all_sentences.append({
#                     "page": page,
#                     "sent_id": sid,
#                     "text": text
#                 })

#         if not all_sentences:
#             return []

#         texts = [s["text"] for s in all_sentences]

#         tfidf = TfidfVectorizer(ngram_range=(1, 2))
#         mat = tfidf.fit_transform(texts)
#         q = tfidf.transform([claim])

#         tfidf_scores = cosine_similarity(q, mat)[0]
#         top_idx = np.argsort(tfidf_scores)[::-1][:30]

#         candidates = [all_sentences[i] for i in top_idx]
#         candidate_texts = [c["text"] for c in candidates]

#         claim_emb = self.bert_model.encode([claim], convert_to_numpy=True)
#         sent_emb = self.bert_model.encode(candidate_texts, convert_to_numpy=True)

#         bert_scores = cosine_similarity(claim_emb, sent_emb)[0]

#         final_scores = 0.6 * bert_scores + 0.4 * tfidf_scores[top_idx]

#         results = []

#         for i, idx in enumerate(top_idx):
#             score = final_scores[i]
#             text = candidates[i]["text"]

#             # 🔥 stronger filtering (removes junk evidence)
#             if score < 0.18:
#                 continue

#             # extra noise filter
#             if len(text.split()) < 5:
#                 continue

#             results.append({
#                 "page": candidates[i]["page"],
#                 "sent_id": candidates[i]["sent_id"],
#                 "text": text,
#                 "score": float(score)
#             })

#         return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

#     # --------------------------
#     # MAIN PIPELINE
#     # --------------------------
#     def retrieve(self, claim, k=5):
#         pages = self.retrieve_pages(claim, top_k=10)

#         if not pages:
#             return []

#         return self.retrieve_sentences(claim, pages, top_k=k)



## ----------------------------------------------------------------------------------------
## Working one 


# import re
# from collections import Counter

# class Retriever:
#     def __init__(self, wiki):
#         self.wiki = wiki
#         self.docs = []

#         for page, sentences in wiki.items():
#             for idx, text in sentences.items():
#                 self.docs.append({
#                     "page": page,
#                     "text": text
#                 })

#     # --------------------------
#     # SIMPLE TOKENIZER
#     # --------------------------
#     def tokenize(self, text):
#         text = text.lower()
#         text = re.sub(r'[^a-z0-9 ]', '', text)
#         return text.split()

#     # --------------------------
#     # SIMPLE SIMILARITY (WORD OVERLAP)
#     # --------------------------
#     def score(self, query_tokens, doc_tokens):
#         query_set = set(query_tokens)
#         doc_set = set(doc_tokens)

#         overlap = query_set.intersection(doc_set)
#         return len(overlap)

#     # --------------------------
#     # RETRIEVE
#     # --------------------------
#     def retrieve(self, query, k=5):

#         query_tokens = self.tokenize(query)

#         results = []

#         for doc in self.docs:
#             doc_tokens = self.tokenize(doc["text"])
#             s = self.score(query_tokens, doc_tokens)

#             if s > 0:  # 🔥 ignore useless matches
#                 results.append({
#                     "page": doc["page"],
#                     "text": doc["text"],
#                     "score": float(s)
#                 })

#         # sort by score
#         results = sorted(results, key=lambda x: x["score"], reverse=True)

#         return results[:k]


#-------------------------------
# import re

# STOPWORDS = {"is", "a", "the", "on", "in", "of", "and", "to"}

# class Retriever:
#     def __init__(self, wiki):
#         self.wiki = wiki
#         self.docs = []

#         for page, sentences in wiki.items():
#             for idx, text in sentences.items():
#                 self.docs.append({
#                     "page": page,
#                     "text": text
#                 })

#     def tokenize(self, text):
#         text = text.lower()
#         text = re.sub(r'[^a-z0-9 ]', '', text)
#         return [w for w in text.split() if w not in STOPWORDS]

#     def score(self, query_tokens, doc_tokens):
#         query_set = set(query_tokens)
#         doc_set = set(doc_tokens)

#         overlap = query_set.intersection(doc_set)

#         if len(overlap) == 0:
#             return 0

#         # ✅ improved scoring
#         return len(overlap) / (len(doc_set) ** 0.5)

#     def retrieve(self, query, k=5):

#         query_tokens = self.tokenize(query)

#         results = []

#         for doc in self.docs:
#             doc_tokens = self.tokenize(doc["text"])
#             s = self.score(query_tokens, doc_tokens)

#             if s > 0:
#                 results.append({
#                     "page": doc["page"],
#                     "text": doc["text"],
#                     "score": float(s)
#                 })

#         results = sorted(results, key=lambda x: x["score"], reverse=True)

#         return results[:k]


#--------------------------------------------------------------------------------------

# import re

# class Retriever:
#     def __init__(self, wiki):
#         self.wiki = wiki
#         self.docs = []

#         for page, sentences in wiki.items():
#             for idx, text in sentences.items():
#                 self.docs.append({
#                     "page": page,
#                     "text": text,
#                     "text_lower": text.lower()
#                 })

#         # optional speed cap (important for performance)
#         self.docs = self.docs[:200000]

#     # --------------------------
#     # TOKENIZER
#     # --------------------------
#     def tokenize(self, text):
#         text = text.lower()
#         text = re.sub(r'[^a-z0-9 ]', '', text)
#         return text.split()

#     # --------------------------
#     # IMPROVED SCORING
#     # --------------------------
#     def score(self, query_tokens, doc):

#         doc_tokens = self.tokenize(doc["text"])

#         # basic overlap
#         overlap = len(set(query_tokens).intersection(set(doc_tokens)))

#         # 🔥 BIG FIX: phrase + entity boost
#         query_text = " ".join(query_tokens)
#         bonus = 0

#         if query_text in doc["text_lower"]:
#             bonus += 3   # strong boost for full phrase

#         # boost if multiple keywords appear
#         keyword_hits = sum(1 for w in query_tokens if w in doc["text_lower"])
#         bonus += 0.5 * keyword_hits

#         return overlap + bonus

#     # --------------------------
#     # RETRIEVE
#     # --------------------------
#     def retrieve(self, query, k=5):

#         query_tokens = self.tokenize(query)

#         results = []

#         for doc in self.docs:
#             s = self.score(query_tokens, doc)

#             # 🔥 stricter filtering (important)
#             if s >= 2:
#                 results.append({
#                     "page": doc["page"],
#                     "text": doc["text"],
#                     "score": float(s)
#                 })

#         results = sorted(results, key=lambda x: x["score"], reverse=True)

#         return results[:k]

##--------------------------------------------------------------------------------------------


# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# class Retriever:
#     def __init__(self, wiki):

#         self.wiki = wiki
#         self.docs = []

#         print("Building document corpus...")

#         for page, sentences in wiki.items():
#             for idx, text in sentences.items():
#                 if len(text.split()) < 5:
#                     continue

#                 self.docs.append({
#                     "page": page,
#                     "text": text
#                 })

#         # ⚡ speed control (important)
#         self.docs = self.docs[:100000]

#         self.texts = [d["text"] for d in self.docs]

#         print("Loading semantic model...")
#         self.model = SentenceTransformer("all-MiniLM-L6-v2")

#         print("Encoding documents (this takes time once)...")
#         self.embeddings = self.model.encode(
#             self.texts,
#             batch_size=64,
#             show_progress_bar=True,
#             convert_to_numpy=True
#         )

#         print(f"✅ Indexed {len(self.docs)} sentences.")

#     # --------------------------
#     # RETRIEVE
#     # --------------------------
#     def retrieve(self, query, k=5):

#         query_emb = self.model.encode([query], convert_to_numpy=True)

#         scores = cosine_similarity(query_emb, self.embeddings)[0]

#         top_idx = np.argsort(scores)[::-1][:k * 5]

#         results = []

#         for i in top_idx:
#             score = scores[i]

#             # 🔥 filter weak matches
#             if score < 0.35:
#                 continue

#             results.append({
#                 "page": self.docs[i]["page"],
#                 "text": self.docs[i]["text"],
#                 "score": float(score)
#             })

#         return results[:k]


##-=============================================================================================
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np


# class Retriever:
#     def __init__(self, wiki):

#         self.wiki = wiki
#         self.docs = []

#         print("Building document corpus...")

#         for page, sentences in wiki.items():
#             for idx, text in sentences.items():

#                 # skip useless sentences
#                 if len(text.split()) < 6:
#                     continue

#                 self.docs.append({
#                     "page": page,
#                     "text": text
#                 })

#         # ⚡ SPEED CONTROL (you can increase later)
#         self.docs = self.docs[:150000]

#         self.texts = [d["text"] for d in self.docs]

#         print("Loading semantic model...")
#         self.model = SentenceTransformer("all-MiniLM-L6-v2")

#         print("Encoding documents (one-time cost)...")
#         self.embeddings = self.model.encode(
#             self.texts,
#             batch_size=64,
#             show_progress_bar=True,
#             convert_to_numpy=True,
#             normalize_embeddings=True   # 🔥 IMPORTANT
#         )

#         print(f"✅ Indexed {len(self.docs)} sentences.")

#     # --------------------------
#     # RETRIEVE
#     # --------------------------
#     def retrieve(self, query, k=5):

#         query_emb = self.model.encode(
#             [query],
#             convert_to_numpy=True,
#             normalize_embeddings=True  # 🔥 IMPORTANT
#         )

#         scores = cosine_similarity(query_emb, self.embeddings)[0]

#         # take larger pool first
#         top_idx = np.argsort(scores)[::-1][:k * 10]

#         results = []

#         for i in top_idx:
#             score = scores[i]

#             # 🔥 LOWER threshold (fixes your Dhoni issue)
#             if score < 0.25:
#                 continue

#             results.append({
#                 "page": self.docs[i]["page"],
#                 "text": self.docs[i]["text"],
#                 "score": float(score)
#             })

#         return results[:k]

## =======================================================

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os


class Retriever:
    def __init__(self, wiki):

        self.wiki = wiki
        self.docs = []

        print("Building document corpus...")

        for page, sentences in wiki.items():
            for idx, text in sentences.items():

                # skip weak sentences
                if len(text.split()) < 6:
                    continue

                self.docs.append({
                    "page": page,
                    "text": text,
                    "text_lower": text.lower()
                })

        # ⚡ speed control
        # self.docs = self.docs[:150000]
        self.docs = self.docs[:200000]

        self.texts = [d["text"] for d in self.docs]

        print("Loading semantic model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # --------------------------
        # 🔥 CACHE (VERY IMPORTANT)
        # --------------------------
        if os.path.exists("embeddings.npy"):
            print("Loading cached embeddings...")
            self.embeddings = np.load("embeddings.npy")
        else:
            print("Encoding documents (one-time cost)...")
            self.embeddings = self.model.encode(
                self.texts,
                batch_size=64,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            np.save("embeddings.npy", self.embeddings)

        print(f"✅ Indexed {len(self.docs)} sentences.")

    # --------------------------
    # 🔥 ENTITY EXTRACTION (SIMPLE BUT EFFECTIVE)
    # --------------------------
    def extract_entities(self, text):
        words = text.split()
        return [w.lower() for w in words if w[0].isupper()]

    # --------------------------
    # RETRIEVE
    # --------------------------
    def retrieve(self, query, k=5):

        query_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores = cosine_similarity(query_emb, self.embeddings)[0]

        query_entities = self.extract_entities(query)

        results = []

        # take larger pool first
        top_idx = np.argsort(scores)[::-1][:k * 15]

        for i in top_idx:
            base_score = scores[i]
            text_lower = self.docs[i]["text_lower"]

            bonus = 0

            # 🔥 ENTITY BOOST (CRITICAL FIX)
            for ent in query_entities:
                if ent in text_lower:
                    bonus += 0.4

            # 🔥 EXTRA: keyword density boost
            keyword_hits = sum(1 for w in query.lower().split() if w in text_lower)
            bonus += 0.1 * keyword_hits

            final_score = base_score + bonus

            # 🔥 relaxed threshold (important)
            if final_score < 0.30:
                continue

            results.append({
                "page": self.docs[i]["page"],
                "text": self.docs[i]["text"],
                "score": float(final_score)
            })

        # sort final results
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        return results[:k]