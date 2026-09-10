def ndcg_pro_query(run, qid):
    return float(evaluate(Qrels({qid: qrels()[qid]}), Run({qid: run[qid]}), ["ndcg@10"]))


print(f"{'qid':4s} {'BM25':>6s} {'Dense':>6s}  Frage")
for q in queries:
    nb = ndcg_pro_query(laeufe["BM25"], q.qid)
    nd = ndcg_pro_query(laeufe["Dense"], q.qid)
    if abs(nb - nd) > 0.1:
        print(f"{q.qid:4s} {nb:6.2f} {nd:6.2f}  {q.text[:46]}")
