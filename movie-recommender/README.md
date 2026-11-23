# Movie Recommender (MovieLens ml-latest-small, runs entirely on Kaggle)

A single Kaggle notebook that builds and compares eight recommenders, from a popularity baseline to deep
retrieval and sequential models, all scored on the same train/test split.

- **Notebook:** https://www.kaggle.com/code/atomicboyy/movie-recommender (private)
- **Dataset:** [`shubhammehta21/movie-lens-small-latest-dataset`](https://www.kaggle.com/datasets/shubhammehta21/movie-lens-small-latest-dataset):
  100,836 ratings, 610 users, 9,742 movies, 3,683 tags (98.3% sparse)
- **Nothing runs locally.** This folder holds only the notebook, the Kaggle metadata, this README and the small outputs
  fetched from the last run (`kaggle_out/`, under 300 KB).

## Models

| Model | Idea |
|---|---|
| Popularity | IMDb weighted rating `v/(v+m)·R + m/(v+m)·C`, with m = 90th percentile of vote counts |
| Content | TF-IDF over genres + user tags, cosine similarity; `similar_movies()` with fuzzy title matching |
| SVD | Surprise biased MF, 3-fold GridSearchCV on train only |
| Item-KNN | Surprise `KNNBaseline`, item-based, `pearson_baseline` similarity, k=40 |
| Hybrid | Top-200 content candidates reranked by `α·norm(SVD) + (1-α)·norm(content)`, α=0.7, with "Because you liked X" explanations |
| NeuMF | GMF + MLP (dim 64), explicit ratings, MSE, AdamW, dropout, AMP, early stopping on val RMSE |
| Two-tower | User tower (ID→MLP), item tower (ID + multi-hot genres + 32-d tag SVD → MLP), in-batch sampled softmax with logQ correction; τ picked on validation |
| SASRec | 2-block causal transformer over each user's last 50 rated movies, full-softmax next-item loss |

**Protocol.** One 80/20 random split (`random_state=42`) shared by every model. The deep models carve a 10% validation
split out of train for early stopping and tuning. Ranking metrics are computed over the **full catalogue**, with the
user's train items excluded. A test item counts as relevant if its rating is ≥ 4. Coverage is the share of the catalogue
that appears in at least one user's top 10. SASRec is also scored on the standard leave-last-out protocol (LLO).

## Results (Kaggle run, version 4)

| Model | RMSE | MAE | P@10 | R@10 | NDCG@10 | Coverage | LLO HR@10 | LLO NDCG@10 |
|---|---|---|---|---|---|---|---|---|
| **Two-tower** | – | – | **0.177** | **0.170** | **0.237** | 0.071 | – | – |
| Popularity | 0.985 | 0.769 | 0.092 | 0.073 | 0.117 | 0.005 | 0.039 | 0.018 |
| SASRec | – | – | 0.053 | 0.066 | 0.073 | 0.153 | **0.102** | **0.051** |
| Item-KNN | **0.859** | **0.653** | 0.025 | 0.017 | 0.028 | 0.166 | – | – |
| SVD | 0.878 | 0.670 | 0.020 | 0.016 | 0.022 | 0.021 | – | – |
| Hybrid | – | – | 0.009 | 0.008 | 0.011 | 0.122 | – | – |
| NeuMF | 0.867 | 0.662 | 0.011 | 0.007 | 0.011 | 0.010 | – | – |
| Content | – | – | 0.004 | 0.004 | 0.004 | **0.214** | – | – |

Total runtime: **25.2 min**. The run that produced these numbers used Kaggle's **CPU**, because the GPU queue was stuck
for over an hour. An earlier GPU run (Tesla T4) finished in 1.2 min. In that run SASRec and the two-tower model were
still mis-trained, so its ranking numbers are not comparable.

### Takeaways
- **The two-tower retrieval model wins on ranking by a wide margin** (NDCG@10 0.237, about 2× popularity). It is trained
  directly for top-N retrieval, and the logQ correction stops in-batch negatives from penalising popular movies.
- **Low RMSE does not mean good top-10 lists.** Item-KNN has the best RMSE and NeuMF beats SVD on RMSE, yet all three rating
  models rank far below the popularity baseline. Over the full catalogue they push obscure movies with a few high ratings to
  the top, while test positives cluster on popular movies.
- **Popularity is a strong baseline** on this skewed, small dataset, but it recommends almost the same handful of movies to
  everyone (coverage 0.5%).
- **SASRec** beats popularity about 2.6× on leave-last-out, where order matters. On the random split it lands between
  popularity and the CF models.
- **Hybrid underperforms plain SVD.** Gating candidates by content similarity discards the popular movies users actually
  rate highly. A better hybrid would take candidates from CF or two-tower and use content for reranking and explanations.
- **Diversity and cold start:** content-based scoring has the highest coverage and works for unrated movies, as does the
  two-tower item tower through genre and tag features. Pure CF cannot score new items, and unknown users fall back to popularity.
- **Next steps:** MovieLens 25M (ALS/`implicit`, FAISS for two-tower retrieval), a temporal split, history features and
  hard negatives for the two-tower model, tuning α on validation, and EASE, LightGCN or BERT4Rec.

## Re-running on Kaggle

Requires the Kaggle CLI, authenticated. On this machine it is invoked as `python -m kaggle`.

```powershell
python -m kaggle kernels push -p movie-recommender -t 3000
```

```powershell
python -m kaggle kernels status atomicboyy/movie-recommender
```

```powershell
python -m kaggle kernels output atomicboyy/movie-recommender -p movie-recommender/kaggle_out
```

- `kernel-metadata.json` currently has `"enable_gpu": false` (the CPU run). To use a GPU, set it to `true` and optionally
  add `--accelerator NvidiaTeslaT4` to the push command. The notebook checks that CUDA really works and falls back to CPU
  with a warning if it doesn't.
- `-t 3000` caps the run at 50 minutes. SASRec also stops itself if the notebook nears its 45-minute budget.
- Outputs written to `/kaggle/working`: `results.csv`, `eda.png`, `neumf_loss.png`, `two_tower_loss.png` and
  `model_comparison.png`. No model weights or similarity matrices are saved.
