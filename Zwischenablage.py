def ndcg_pro_query(run, qid):
    return float(evaluate(Qrels({qid: qrels()[qid]}), Run({qid: run[qid]}), ["ndcg@10"]))


print(f"{'qid':4s} {'BM25':>6s} {'Dense':>6s}  Frage")
for q in queries:
    nb = ndcg_pro_query(laeufe["BM25"], q.qid)
    nd = ndcg_pro_query(laeufe["Dense"], q.qid)
    if abs(nb - nd) > 0.1:
        print(f"{q.qid:4s} {nb:6.2f} {nd:6.2f}  {q.text[:46]}")


def hybrid_k(query, kk, pool=20):
    bm = [c.chunk_id for c, _ in bm25.search(query, pool)]
    dn = [c.chunk_id for c, _ in dense.search(query, pool)]
    top = sorted(rrf([bm, dn], k=kk).items(), key=lambda x: -x[1])[:10]
    return [(by_id[cid], s) for cid, s in top]


for kk in (10, 60, 200):
    run = run_for(lambda q, kk=kk: hybrid_k(q, kk))
    ndcg = float(evaluate(gold, Run(run), ["ndcg@10"]))
    print(f"rrf k={kk:3d}: nDCG@10={ndcg:.3f}")
