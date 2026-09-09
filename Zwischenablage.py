import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

labels = [f"S{i+1}" for i in range(len(sentences))]
xy = PCA(n_components=2).fit_transform(vektoren)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Links: Ähnlichkeitsmatrix als Heatmap
ax1.imshow(sim, cmap="Blues", vmin=0, vmax=1)
ax1.set_xticks(range(len(sentences)), labels)
ax1.set_yticks(range(len(sentences)), labels)
for i in range(len(sentences)):
    for j in range(len(sentences)):
        ax1.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center",
                 color="white" if sim[i, j] > 0.6 else "black")
ax1.set_title("Cosine-Ähnlichkeit")

# Rechts: PCA-Projektion, jeder Punkt ein Satz.
ax2.scatter(xy[:, 0], xy[:, 1], s=80)
for (x, y), lab, satz in zip(xy, labels, sentences):
    ax2.annotate(f"{lab}: {satz[:28]}", (x, y), xytext=(6, 4),
                 textcoords="offset points", fontsize=9)
ax2.set_title("PCA-Projektion der Vektoren")
ax2.margins(0.25)

plt.tight_layout()
plt.show()




import re
from collections import Counter

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text):
    return _TOKEN.findall(text.lower())


doc_tokens = [Counter(tokenize(f"{d.title}. {d.text}")) for d in docs]
N = len(docs)
# Dokumentfrequenz je Wort: in wie vielen Dokumenten kommt es vor?
df = Counter(w for tokens in doc_tokens for w in tokens)


def sparse_suche(frage, k=3):
    """Gewichtete Wortüberlappung: häufig im Dokument, selten im Korpus = viele Punkte."""
    scores = []
    for tokens in doc_tokens:
        s = 0.0
        for w in tokenize(frage):
            if w in tokens:
                idf = np.log(N / df[w])       # seltenes Wort, hoher Wert
                s += tokens[w] * idf
        scores.append(s)
    top = np.argsort(scores)[::-1][:k]
    # Score 0 heißt: kein einziges Wort der Frage kommt vor. Solche Dokumente
    # sind keine Treffer, sondern nur Rauschen, und fliegen raus.
    return [(doc_ids[i], float(scores[i])) for i in top if scores[i] > 0]


print(sparse_suche("P12-2007"))
