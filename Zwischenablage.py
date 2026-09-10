import time

def run_for(searcher) -> dict:
    out = {}
    for q in queries:
        scored = searcher(q.text)
        out[q.qid] = {d: s for d, s in R.collapse_to_docs(scored)}
    return out


setups = {
    "BM25":         lambda q: bm25.search(q, 10),
    "Dense":        lambda q: dense.search(q, 10),
    "Hybrid-RRF":   lambda q: hybrid_search(q, 10),
    "Hybrid+Rerank": lambda q: hybrid_rerank(q, 10),
}

zeilen = []
laeufe = {}
for name, fn in setups.items():
    t0 = time.perf_counter()
    run = run_for(fn)
    dt = time.perf_counter() - t0
    laeufe[name] = run
    m = evaluate(gold, Run(run, name=name), ["ndcg@10", "recall@10", "mrr"])
    zeilen.append({"setup": name, **{k: round(float(v), 3) for k, v in m.items()},
                   "sek/Query": round(dt / len(queries), 3)})

df = pd.DataFrame(zeilen).set_index("setup")
print(df)
