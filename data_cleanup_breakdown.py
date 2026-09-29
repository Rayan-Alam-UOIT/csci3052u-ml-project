from URIELPlus.urielplus import urielplus

import numpy as np
import matplotlib.pyplot as plt


def load_data(file):
    npz_file = np.load(
        f'URIELPlus/urielplus/database/{file}',
        allow_pickle=True
    )
    return npz_file['data']


def check_missing(data, label):
    missing = np.sum(data == -1)
    total = data.size
    percentage = (missing / total) * 100

    print(f"\n{label}")
    print(f"Shape: {data.shape}")
    print(f"Missing values (-1): {missing:,}")
    print(f"Total values: {total:,}")
    print(f"Missing percentage: {percentage:.2f}%")


# Initialize URIEL+

u = urielplus.URIELPlus()
u.reset()

# Configuration
u.set_cache(True)
u.set_aggregation('U')

# Integrate databases
u.integrate_custom_databases(
    "UPDATED_SAPHON",
    "BDPROTO",
    "GRAMBANK",
    "APICS",
    "EWAVE"
)


# 1. Aggregation WITHOUT base-language filling

u.set_fill_with_base_lang(False)
u.aggregate()

data_post_aggregation = load_data("features_union.npz")

check_missing(
    data_post_aggregation,
    "After aggregation (before base-language filling)"
)


# Values that were known after union aggregation
originally_known = data_post_aggregation != -1


# 2. Aggregation WITH base-language filling

u.set_fill_with_base_lang(True)
u.aggregate()

data_post_base_lang = load_data("features_union.npz")

check_missing(
    data_post_base_lang,
    "After base-language filling"
)


# Values that were missing after aggregation, but were filled using a related/base language
filled_by_base_lang = (
    (data_post_aggregation == -1) &
    (data_post_base_lang != -1)
)


# 3. SoftImpute

u.softimpute_imputation()

data_post_imputation = load_data("features.npz")

check_missing(
    data_post_imputation,
    "After SoftImpute"
)


# Values that were still missing after base-language filling, but were filled by SoftImpute
filled_by_softimpute = (
    (data_post_base_lang == -1) &
    (data_post_imputation != -1)
)


# 4. Values still missing after SoftImpute

still_missing = data_post_imputation == -1


# Count categories

originally_known_count = np.sum(originally_known)
base_lang_count = np.sum(filled_by_base_lang)
softimpute_count = np.sum(filled_by_softimpute)
still_missing_count = np.sum(still_missing)

total = data_post_imputation.size


# Final breakdown

print("\n----------------------------------------")
print("Final breakdown")
print("----------------------------------------")

print(
    f"Known after aggregation:       "
    f"{originally_known_count:,} "
    f"({originally_known_count / total * 100:.2f}%)"
)

print(
    f"Filled by related language:     "
    f"{base_lang_count:,} "
    f"({base_lang_count / total * 100:.2f}%)"
)

print(
    f"Filled by SoftImpute:           "
    f"{softimpute_count:,} "
    f"({softimpute_count / total * 100:.2f}%)"
)

print(
    f"Still missing:                  "
    f"{still_missing_count:,} "
    f"({still_missing_count / total * 100:.2f}%)"
)

print(f"Total data points:              {total:,}")


# Validate all data points are accounted for

classified = (
    originally_known_count
    + base_lang_count
    + softimpute_count
    + still_missing_count
)

print(f"\nClassified data points:         {classified:,}")

if classified == total:
    print("✓ All data points accounted for.")
else:
    print("WARNING: Data point counts do not add up!")


# Pie chart

labels = [
    "Known after aggregation",
    "Filled by related language",
    "Filled by SoftImpute",
    "Still missing"
]

sizes = [
    originally_known_count,
    base_lang_count,
    softimpute_count,
    still_missing_count
]

percentages = [
    value / total * 100
    for value in sizes
]


def autopct_with_count(pct):
    count = int(round(pct * total / 100))
    return f"{pct:.1f}%\n({count:,})"


fig, ax = plt.subplots(figsize=(10, 8))

wedges, texts, autotexts = ax.pie(
    sizes,
    labels=None,
    autopct=autopct_with_count,
    startangle=90,
    counterclock=False,
    pctdistance=0.78,
    wedgeprops=dict(width=0.42, edgecolor="white")
)


# Improve percentage/count text
for autotext in autotexts:
    autotext.set_fontsize(10)
    autotext.set_fontweight("bold")


# Center text
ax.text(
    0,
    0,
    f"{total:,}\nTotal Data Points",
    ha="center",
    va="center",
    fontsize=13,
    fontweight="bold"
)


# Legend
legend_labels = [
    f"{label} — {count:,} ({percentage:.2f}%)"
    for label, count, percentage
    in zip(labels, sizes, percentages)
]

ax.legend(
    wedges,
    legend_labels,
    title="Data point classification",
    loc="center left",
    bbox_to_anchor=(1.0, 0.5),
    frameon=False
)


ax.set_title(
    "URIEL+ Feature Coverage After Aggregation and Imputation",
    fontsize=15,
    fontweight="bold",
    pad=20
)

ax.set_aspect("equal")

plt.tight_layout()
plt.show()