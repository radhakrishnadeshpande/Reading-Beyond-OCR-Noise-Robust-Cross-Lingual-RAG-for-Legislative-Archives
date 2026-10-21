# # # import torch
# # # import numpy as np
# # # from sentence_transformers import SentenceTransformer
# # # from sklearn.metrics.pairwise import cosine_similarity

# # # MODEL = "BAAI/bge-m3"

# # # device = "cuda" if torch.cuda.is_available() else "cpu"

# # # model = SentenceTransformer(MODEL, device=device)

# # # documents = [
# # #     "विद्यालय में लगभग पाँच सौ विद्यार्थी अध्ययन करते हैं।",
# # #     "भारत सरकार ने शिक्षा के लिए नई योजना शुरू की।",
# # #     "न्यायालय ने मामले की सुनवाई की।",
# # #     "शहर में आज भारी बारिश हुई।",
# # #     "कृषि विभाग ने किसानों के लिए नई योजना घोषित की।"
# # # ]

# # # queries = [
# # #     "How many students study in the school?",
# # #     "What new education scheme did the Indian government introduce?",
# # #     "What did the court hear?"
# # # ]

# # # doc_embeddings = model.encode(
# # #     documents,
# # #     normalize_embeddings=True,
# # #     convert_to_numpy=True
# # # )

# # # query_embeddings = model.encode(
# # #     queries,
# # #     normalize_embeddings=True,
# # #     convert_to_numpy=True
# # # )

# # # scores = cosine_similarity(
# # #     query_embeddings,
# # #     doc_embeddings
# # # )

# # # for q_idx, query in enumerate(queries):

# # #     ranking = np.argsort(scores[q_idx])[::-1]

# # #     print("\n========================================")
# # #     print("QUERY:", query)
# # #     print("========================================")

# # #     for rank, doc_idx in enumerate(ranking, start=1):

# # #         print(
# # #             f"\nRank {rank}"
# # #             f"\nScore: {scores[q_idx][doc_idx]:.4f}"
# # #             f"\nDocument: {documents[doc_idx]}"
# # #         )

# # import torch
# # import numpy as np
# # from sentence_transformers import SentenceTransformer
# # from sklearn.metrics.pairwise import cosine_similarity

# # MODEL = "BAAI/bge-m3"

# # device = "cuda" if torch.cuda.is_available() else "cpu"

# # print("Device:", device)
# # print("Loading:", MODEL)

# # model = SentenceTransformer(MODEL, device=device)

# # pairs = [
# #     ("विद्यालय", "school"),
# #     ("शिक्षक", "teacher"),
# #     ("छात्र", "student"),
# #     ("पुस्तक", "book"),
# #     ("सरकार", "government"),
# #     ("कानून", "law"),
# #     ("न्यायालय", "court"),
# #     ("शिक्षा", "education"),
# #     ("घर", "house"),
# #     ("पानी", "water"),
# # ]

# # hindi = [x[0] for x in pairs]
# # english = [x[1] for x in pairs]

# # hindi_embeddings = model.encode(
# #     hindi,
# #     normalize_embeddings=True,
# #     convert_to_numpy=True
# # )

# # english_embeddings = model.encode(
# #     english,
# #     normalize_embeddings=True,
# #     convert_to_numpy=True
# # )

# # similarities = cosine_similarity(
# #     hindi_embeddings,
# #     english_embeddings
# # )

# # print("\n=== CROSS-LINGUAL SIMILARITY ===\n")

# # for i, (hi, en) in enumerate(pairs):
# #     score = similarities[i][i]

# #     print(
# #         f"{hi:15s} <-> {en:15s} "
# #         f"similarity = {score:.4f}"
# #     )

# # print("\n=== FULL SIMILARITY MATRIX ===\n")

# # print("             " + " ".join(f"{x:>12}" for x in english))

# # for i, hi in enumerate(hindi):
# #     row = " ".join(f"{similarities[i][j]:12.4f}" for j in range(len(english)))
# #     print(f"{hi:10s} {row}")
# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm
# import seaborn as sns

# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity


# # ============================================================
# # 1. FONT CONFIGURATION
# # ============================================================

# # Hindi / Devanagari font
# DEVANAGARI_FONT_PATH = os.path.expanduser(
#     "~/.local/share/fonts/NotoSansDevanagari-Regular.ttf"
# )

# # Check that Hindi font exists
# if not os.path.exists(DEVANAGARI_FONT_PATH):
#     raise FileNotFoundError(
#         f"\nHindi font not found:\n{DEVANAGARI_FONT_PATH}\n\n"
#         "Please check that the TTF file is installed correctly."
#     )

# # English font
# DEJAVU_FONT_PATH = fm.findfont(
#     fm.FontProperties(family="DejaVu Sans")
# )

# print("=" * 70)
# print("FONT CONFIGURATION")
# print("=" * 70)

# print("Hindi font   :", DEVANAGARI_FONT_PATH)
# print("English font :", DEJAVU_FONT_PATH)

# # Register fonts
# fm.fontManager.addfont(DEVANAGARI_FONT_PATH)
# fm.fontManager.addfont(DEJAVU_FONT_PATH)

# # Create font properties
# hindi_font = fm.FontProperties(
#     fname=DEVANAGARI_FONT_PATH
# )

# english_font = fm.FontProperties(
#     fname=DEJAVU_FONT_PATH
# )

# print("Hindi font name   :", hindi_font.get_name())
# print("English font name :", english_font.get_name())


# # ============================================================
# # 2. MODEL
# # ============================================================

# MODEL = "BAAI/bge-m3"

# device = "cuda" if torch.cuda.is_available() else "cpu"

# print("\n" + "=" * 70)
# print("MODEL")
# print("=" * 70)

# print("Device :", device)
# print("Model  :", MODEL)

# model = SentenceTransformer(
#     MODEL,
#     device=device
# )


# # ============================================================
# # 3. ONLY 10 HINDI - ENGLISH WORD PAIRS
# # ============================================================

# pairs = [

#     ("विद्यालय", "school"),
#     ("शिक्षक", "teacher"),
#     ("छात्र", "student"),
#     ("पुस्तक", "book"),
#     ("शिक्षा", "education"),
#     ("सरकार", "government"),
#     ("मंत्री", "minister"),
#     ("विधानसभा", "assembly"),
#     ("संसद", "parliament"),
#     ("जिला", "district"),

# ]

# hindi = [x[0] for x in pairs]
# english = [x[1] for x in pairs]

# print("\n" + "=" * 70)
# print("WORD PAIRS")
# print("=" * 70)

# print("Number of word pairs:", len(pairs))

# for hi, en in pairs:
#     print(f"{hi:15s} <-> {en}")


# # ============================================================
# # 4. GENERATE HINDI EMBEDDINGS
# # ============================================================

# print("\n" + "=" * 70)
# print("GENERATING HINDI EMBEDDINGS")
# print("=" * 70)

# hindi_embeddings = model.encode(
#     hindi,
#     normalize_embeddings=True,
#     convert_to_numpy=True,
#     show_progress_bar=True
# )


# # ============================================================
# # 5. GENERATE ENGLISH EMBEDDINGS
# # ============================================================

# print("\n" + "=" * 70)
# print("GENERATING ENGLISH EMBEDDINGS")
# print("=" * 70)

# english_embeddings = model.encode(
#     english,
#     normalize_embeddings=True,
#     convert_to_numpy=True,
#     show_progress_bar=True
# )


# # ============================================================
# # 6. COSINE SIMILARITY MATRIX
# # ============================================================

# similarities = cosine_similarity(
#     hindi_embeddings,
#     english_embeddings
# )

# print("\nSimilarity matrix shape:")
# print(similarities.shape)


# # ============================================================
# # 7. PRINT CROSS-LINGUAL SIMILARITY
# # ============================================================

# print("\n" + "=" * 70)
# print("CROSS-LINGUAL SIMILARITY")
# print("=" * 70)

# for i, (hi, en) in enumerate(pairs):

#     score = similarities[i][i]

#     print(
#         f"{hi:15s} <-> "
#         f"{en:15s} "
#         f"similarity = {score:.4f}"
#     )


# # ============================================================
# # 8. DIAGONAL SIMILARITY STATISTICS
# # ============================================================

# diagonal_scores = np.diag(similarities)

# mean_score = np.mean(diagonal_scores)
# min_score = np.min(diagonal_scores)
# max_score = np.max(diagonal_scores)

# min_index = np.argmin(diagonal_scores)
# max_index = np.argmax(diagonal_scores)


# print("\n" + "=" * 70)
# print("SUMMARY STATISTICS")
# print("=" * 70)

# print(
#     f"Number of word pairs : {len(pairs)}"
# )

# print(
#     f"Average similarity   : {mean_score:.4f}"
# )

# print(
#     f"Minimum similarity   : {min_score:.4f}"
# )

# print(
#     f"Maximum similarity   : {max_score:.4f}"
# )

# print("\nLowest pair:")

# print(
#     f"{pairs[min_index][0]} <-> "
#     f"{pairs[min_index][1]} = "
#     f"{min_score:.4f}"
# )

# print("\nHighest pair:")

# print(
#     f"{pairs[max_index][0]} <-> "
#     f"{pairs[max_index][1]} = "
#     f"{max_score:.4f}"
# )


# # ============================================================
# # 9. FULL SIMILARITY MATRIX
# # ============================================================

# print("\n" + "=" * 70)
# print("FULL SIMILARITY MATRIX")
# print("=" * 70)

# print()

# # Print English column names
# print(
#     " " * 15 +
#     " ".join(
#         f"{word:>12}"
#         for word in english
#     )
# )

# for i, hi in enumerate(hindi):

#     row = " ".join(
#         f"{similarities[i][j]:12.4f}"
#         for j in range(len(english))
#     )

#     print(
#         f"{hi:15s} {row}"
#     )


# # ============================================================
# # 10. CREATE RED HEATMAP
# # ============================================================

# print("\n" + "=" * 70)
# print("CREATING RED HEATMAP")
# print("=" * 70)

# fig, ax = plt.subplots(
#     figsize=(16, 13)
# )


# sns.heatmap(
#     similarities,

#     # --------------------------------------------------------
#     # Axis labels
#     # --------------------------------------------------------

#     xticklabels=english,
#     yticklabels=hindi,

#     # --------------------------------------------------------
#     # Display values inside cells
#     # --------------------------------------------------------

#     annot=True,
#     fmt=".2f",

#     # --------------------------------------------------------
#     # Similarity range
#     # --------------------------------------------------------

#     vmin=0.0,
#     vmax=1.0,

#     # --------------------------------------------------------
#     # RED COLOR MAP
#     # --------------------------------------------------------

#     cmap="Reds",

#     # --------------------------------------------------------
#     # Cell borders
#     # --------------------------------------------------------

#     linewidths=0.5,
#     linecolor="white",

#     # --------------------------------------------------------
#     # Color bar
#     # --------------------------------------------------------

#     cbar=True,

#     cbar_kws={
#         "label": "Cosine Similarity",
#         "shrink": 0.80
#     },

#     ax=ax
# )


# # ============================================================
# # 11. TITLE
# # ============================================================

# ax.set_title(
#     "BGE-M3 Cross-Lingual Semantic Similarity\n"
#     "Hindi ↔ English",

#     fontsize=22,
#     fontweight="bold",

#     fontproperties=english_font,

#     pad=25
# )


# # ============================================================
# # 12. X-AXIS LABEL
# # ============================================================

# ax.set_xlabel(
#     "English Terms",

#     fontsize=16,
#     fontweight="bold",

#     fontproperties=english_font,

#     labelpad=15
# )


# # ============================================================
# # 13. Y-AXIS LABEL
# # ============================================================

# ax.set_ylabel(
#     "Hindi Terms",

#     fontsize=16,
#     fontweight="bold",

#     fontproperties=hindi_font,

#     labelpad=15
# )


# # ============================================================
# # 14. ENGLISH X-AXIS TICK LABELS
# # ============================================================

# for label in ax.get_xticklabels():

#     label.set_fontproperties(
#         english_font
#     )

#     label.set_fontsize(11)

#     label.set_rotation(45)

#     label.set_horizontalalignment(
#         "right"
#     )


# # ============================================================
# # 15. HINDI Y-AXIS TICK LABELS
# # ============================================================

# for label in ax.get_yticklabels():

#     label.set_fontproperties(
#         hindi_font
#     )

#     label.set_fontsize(12)

#     label.set_rotation(0)


# # ============================================================
# # 16. NUMBERS INSIDE HEATMAP
# # ============================================================

# for text in ax.texts:

#     text.set_fontproperties(
#         english_font
#     )

#     text.set_fontsize(9)


# # ============================================================
# # 17. COLORBAR
# # ============================================================

# cbar = ax.collections[0].colorbar

# cbar.set_label(
#     "Cosine Similarity",

#     fontsize=13,

#     fontproperties=english_font
# )

# for label in cbar.ax.get_yticklabels():

#     label.set_fontproperties(
#         english_font
#     )

#     label.set_fontsize(10)


# # ============================================================
# # 18. LAYOUT
# # ============================================================

# plt.tight_layout()


# # ============================================================
# # 19. SAVE FIGURE
# # ============================================================

# output_file = (
#     "/home/radhakrishna/ws/adapt_dr/primary_code/"
#     "bge_m3_hindi_english_10words_red_heatmap.png"
# )

# plt.savefig(
#     output_file,

#     dpi=300,

#     bbox_inches="tight"
# )


# print("\n" + "=" * 70)
# print("HEATMAP SAVED")
# print("=" * 70)

# print(output_file)


# # ============================================================
# # 20. DISPLAY
# # ============================================================

# plt.show()


import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. FONT CONFIGURATION
# ============================================================

# Hindi / Devanagari font
DEVANAGARI_FONT_PATH = os.path.expanduser(
    "~/.local/share/fonts/NotoSansDevanagari-Regular.ttf"
)

# Punjabi / Gurmukhi font
GURMUKHI_FONT_PATH = os.path.expanduser(
    "~/.local/share/fonts/NotoSansGurmukhi-Regular.ttf"
)

# Check fonts
for font_path in [DEVANAGARI_FONT_PATH, GURMUKHI_FONT_PATH]:
    if not os.path.exists(font_path):
        raise FileNotFoundError(
            f"\nFont not found: {font_path}\n"
            "Please check that the TTF file is installed correctly."
        )

# English font for numbers and title
DEJAVU_FONT_PATH = fm.findfont(
    fm.FontProperties(family="DejaVu Sans")
)

print("=" * 70)
print("FONT CONFIGURATION")
print("=" * 70)

print("Hindi font   :", DEVANAGARI_FONT_PATH)
print("Punjabi font :", GURMUKHI_FONT_PATH)
print("English font :", DEJAVU_FONT_PATH)

# Register fonts
fm.fontManager.addfont(DEVANAGARI_FONT_PATH)
fm.fontManager.addfont(GURMUKHI_FONT_PATH)
fm.fontManager.addfont(DEJAVU_FONT_PATH)

# Create font properties
hindi_font = fm.FontProperties(
    fname=DEVANAGARI_FONT_PATH
)

punjabi_font = fm.FontProperties(
    fname=GURMUKHI_FONT_PATH
)

english_font = fm.FontProperties(
    fname=DEJAVU_FONT_PATH
)

print("Hindi font name   :", hindi_font.get_name())
print("Punjabi font name :", punjabi_font.get_name())


# ============================================================
# 2. MODEL
# ============================================================

MODEL = "BAAI/bge-m3"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("\n" + "=" * 70)
print("MODEL")
print("=" * 70)

print("Device :", device)
print("Model  :", MODEL)

model = SentenceTransformer(
    MODEL,
    device=device
)


# ============================================================
# 3. ONLY 10 HINDI - PUNJABI WORD PAIRS
# ============================================================

pairs = [
    ("विद्यालय", "ਸਕੂਲ"),          # School
    ("शिक्षक", "ਅਧਿਆਪਕ"),          # Teacher
    ("छात्र", "ਵਿਦਿਆਰਥੀ"),         # Student
    ("पुस्तक", "ਕਿਤਾਬ"),           # Book
    ("शिक्षा", "ਸਿੱਖਿਆ"),          # Education
    ("सरकार", "ਸਰਕਾਰ"),            # Government
    ("मंत्री", "ਮੰਤਰੀ"),            # Minister
    ("विधानसभा", "ਵਿਧਾਨ ਸਭਾ"),     # Legislative assembly
    ("संसद", "ਸੰਸਦ"),              # Parliament
    ("जिला", "ਜ਼ਿਲ੍ਹਾ"),            # District
]

hindi = [x[0] for x in pairs]
punjabi = [x[1] for x in pairs]

print("\n" + "=" * 70)
print("WORD PAIRS")
print("=" * 70)

print("Number of word pairs:", len(pairs))

for hi, pa in pairs:
    print(f"{hi:15s} <-> {pa}")


# ============================================================
# 4. GENERATE HINDI EMBEDDINGS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING HINDI EMBEDDINGS")
print("=" * 70)

hindi_embeddings = model.encode(
    hindi,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True
)


# ============================================================
# 5. GENERATE PUNJABI EMBEDDINGS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING PUNJABI EMBEDDINGS")
print("=" * 70)

punjabi_embeddings = model.encode(
    punjabi,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True
)


# ============================================================
# 6. COSINE SIMILARITY MATRIX
# ============================================================

similarities = cosine_similarity(
    hindi_embeddings,
    punjabi_embeddings
)

print("\nSimilarity matrix shape:")
print(similarities.shape)


# ============================================================
# 7. PRINT CROSS-LINGUAL SIMILARITY
# ============================================================

print("\n" + "=" * 70)
print("HINDI-PUNJABI CROSS-LINGUAL SIMILARITY")
print("=" * 70)

for i, (hi, pa) in enumerate(pairs):
    score = similarities[i][i]

    print(
        f"{hi:15s} <-> "
        f"{pa:15s} "
        f"similarity = {score:.4f}"
    )


# ============================================================
# 8. DIAGONAL SIMILARITY STATISTICS
# ============================================================

diagonal_scores = np.diag(similarities)

mean_score = np.mean(diagonal_scores)
min_score = np.min(diagonal_scores)
max_score = np.max(diagonal_scores)

min_index = np.argmin(diagonal_scores)
max_index = np.argmax(diagonal_scores)

print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

print(f"Number of word pairs : {len(pairs)}")
print(f"Average similarity   : {mean_score:.4f}")
print(f"Minimum similarity   : {min_score:.4f}")
print(f"Maximum similarity   : {max_score:.4f}")

print("\nLowest pair:")
print(
    f"{pairs[min_index][0]} <-> "
    f"{pairs[min_index][1]} = "
    f"{min_score:.4f}"
)

print("\nHighest pair:")
print(
    f"{pairs[max_index][0]} <-> "
    f"{pairs[max_index][1]} = "
    f"{max_score:.4f}"
)


# ============================================================
# 9. FULL SIMILARITY MATRIX
# ============================================================

print("\n" + "=" * 70)
print("FULL SIMILARITY MATRIX")
print("=" * 70)

print(
    " " * 15 +
    " ".join(f"{word:>12}" for word in punjabi)
)

for i, hi in enumerate(hindi):
    row = " ".join(
        f"{similarities[i][j]:12.4f}"
        for j in range(len(punjabi))
    )
    print(f"{hi:15s} {row}")


# ============================================================
# 10. CREATE RED HEATMAP
# ============================================================

print("\n" + "=" * 70)
print("CREATING RED HEATMAP")
print("=" * 70)

fig, ax = plt.subplots(figsize=(16, 13))

sns.heatmap(
    similarities,

    # Axis labels
    xticklabels=punjabi,
    yticklabels=hindi,

    # Display values
    annot=True,
    fmt=".2f",

    # Similarity range
    vmin=0.0,
    vmax=1.0,

    # Red color map
    cmap="Reds",

    # Cell borders
    linewidths=0.5,
    linecolor="white",

    # Color bar
    cbar=True,
    cbar_kws={
        "label": "Cosine Similarity",
        "shrink": 0.80
    },

    ax=ax
)


# ============================================================
# 11. TITLE
# ============================================================

ax.set_title(
    "BGE-M3 Cross-Lingual Semantic Similarity\n"
    "Hindi ↔ Punjabi",

    fontsize=22,
    fontweight="bold",
    fontproperties=english_font,
    pad=25
)


# ============================================================
# 12. X-AXIS LABEL
# ============================================================

ax.set_xlabel(
    "Punjabi Terms (Gurmukhi)",
    fontsize=16,
    fontweight="bold",
    fontproperties=punjabi_font,
    labelpad=15
)


# ============================================================
# 13. Y-AXIS LABEL
# ============================================================

ax.set_ylabel(
    "Hindi Terms (Devanagari)",
    fontsize=16,
    fontweight="bold",
    fontproperties=hindi_font,
    labelpad=15
)


# ============================================================
# 14. PUNJABI X-AXIS TICK LABELS
# ============================================================

for label in ax.get_xticklabels():
    label.set_fontproperties(punjabi_font)
    label.set_fontsize(12)
    label.set_rotation(45)
    label.set_horizontalalignment("right")


# ============================================================
# 15. HINDI Y-AXIS TICK LABELS
# ============================================================

for label in ax.get_yticklabels():
    label.set_fontproperties(hindi_font)
    label.set_fontsize(12)
    label.set_rotation(0)


# ============================================================
# 16. NUMBERS INSIDE HEATMAP
# ============================================================

for text in ax.texts:
    text.set_fontproperties(english_font)
    text.set_fontsize(9)


# ============================================================
# 17. COLORBAR
# ============================================================

cbar = ax.collections[0].colorbar

cbar.set_label(
    "Cosine Similarity",
    fontsize=13,
    fontproperties=english_font
)

for label in cbar.ax.get_yticklabels():
    label.set_fontproperties(english_font)
    label.set_fontsize(10)


# ============================================================
# 18. LAYOUT
# ============================================================

plt.tight_layout()


# ============================================================
# 19. SAVE FIGURE
# ============================================================

output_file = (
    "/home/radhakrishna/ws/adapt_dr/primary_code/"
    "bge_m3_hindi_punjabi_10words_red_heatmap.png"
)

os.makedirs(os.path.dirname(output_file), exist_ok=True)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

print("\n" + "=" * 70)
print("HEATMAP SAVED")
print("=" * 70)

print(output_file)


# ============================================================
# 20. DISPLAY
# ============================================================

plt.show()
