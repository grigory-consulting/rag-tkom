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
