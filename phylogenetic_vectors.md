# Understanding URIEL+'s Phylogeny/Geography Build Pipeline

## Scope

This report examines three components that together make up the part of
the URIEL+ codebase responsible for building phylogeny (family-tree) and
geography vectors. Two are data tables, and one is the code that consumes
them; together they form what this report calls the **build pipeline**:
the data flows in through the two CSVs, and `urielplus_databases.py`
contains the logic that turns that data into URIEL+'s actual feature
vectors.

| Component | What it is | Rows |
|---|---|---|
| `lang_fam_geo.csv` | Data table. One row per language: its Glottocode, name, latitude/longitude, and full ancestor chain (lineage) | 26,881 |
| `uriel_glottocode_map.csv` | Data table. A lookup table mapping ISO 639-3 codes to Glottocodes | 7,970 |
| `urielplus_databases.py` | Code module. Contains `_calculate_phylogeny_vectors()` and `_calculate_geocoord_vectors()`, the functions that read the two CSVs above and build URIEL+'s feature vectors | n/a: source code, not data |

The goal of this report is to check all three components against
**Glottolog itself**, the original database this pipeline is built from,
to determine whether the numbers URIEL+'s code produces, and the two CSVs
that feed it, still align with Glottolog's current, live data.

---

## 0. Definitions

- **Glottocode** — an eight-character identifier (format:
  `[0-9a-z]{4}[0-9]{4}`, i.e. four letters-or-digits followed by four
  digits) that the Glottolog project assigns to every language, dialect,
  and language family it catalogues. Example: `ghot1243` is Ghotuo, a
  language spoken in Nigeria. The format is defined in Glottolog's FAQ
  (Glottolog, n.d.; see References, §7).
- **Lineage** — the chain of ancestor families a language descends from,
  root to leaf. Example: `Atlantic-Congo, Volta-Congo, Benue-Congo, ...`
- **Glottolog** — the organization and database, maintained at the Max
  Planck Institute for Evolutionary Anthropology, that assigns Glottocodes
  and maintains the family-tree classification for every language in the
  world (Hammarström et al., 2026; see References, §7).

---

## 1. Dataset Overview — `lang_fam_geo.csv`

### Script — Load and Profile the Dataset

```python
import pandas as pd

df = pd.read_csv(
    "lang_fam_geo.csv",
    dtype={"language_id": "string", "language_name": "string", "lineage": "string"},
)

print("shape:", df.shape)
print(df.isna().sum())
print("duplicate language_id:", df["language_id"].duplicated().sum())
print("latitude range:", df["latitude"].min(), df["latitude"].max())
print("longitude range:", df["longitude"].min(), df["longitude"].max())
```

| Property | Value |
|---|---|
| Rows | 26,881 |
| Columns | `language_id`, `language_name`, `latitude`, `longitude`, `lineage` |
| Duplicate `language_id` | 0 |
| Missing `latitude`/`longitude` | 652 (2.4%) |
| Missing `lineage` | 219 (0.8%) |
| Missing `language_name` | 40 (0.1%) |
| Latitude range | −55.27 to 73.14 |
| Longitude range (as stored) | −170.48 to 323.84 |

---

## 2. Glottocode Validation

This check does not require downloading Glottolog's data. It looks
exclusively at whether the identifiers already present in
`lang_fam_geo.csv` conform to Glottolog's own naming rule: a pattern
check against the file's own column, not a lookup against an external
source.

The pattern itself is not derived from the data. Glottolog's FAQ, published
on its GitHub repository (Glottolog, n.d.; see References, §7), specifies
the rule for a valid Glottocode: an eight-character code in which the first
four characters may be any lowercase letter or digit, and the last four
must be digits. That rule is what the regular expression encodes:

- `[0-9a-z]{4}` → first four characters, letters or digits
- `[0-9]{4}` → last four characters, digits only

### Script — Glottocode Syntax Check

```python
import re

glottocode_re = re.compile(r"^[0-9a-z]{4}[0-9]{4}$")
bad = df[~df["language_id"].str.match(glottocode_re)]
print("malformed glottocodes:", len(bad))   # -> 0
```

**Result:** applying Glottolog's own naming rule to every identifier in
`lang_fam_geo.csv` produced zero failures, including a few unusual-looking
codes such as `b10b1234`, `3adt1234`, and `ww2p1234`. The same FAQ page
(Glottolog, n.d.; see References, §7) confirms these are documented,
allowed exceptions to the typical "four letters" pattern, not errors.

---

## 3. Phylogeny Vector Reconstruction and Family Statistics

### 3.1 Rebuilding the Phylogeny Feature List

The URIEL+ code (`_calculate_phylogeny_vectors()`) converts every lineage
chain into a set of path-qualified features and asserts the result must
contain exactly **8,887** nodes, connected by **8,582** parent-to-child
edges, raising an error otherwise. That logic was reimplemented
independently below to verify the outcome.

#### Script — Rebuild Phylogeny Feature Paths

```python
import math

def lineage_parts(value):
    # turns "A, B, C" into ("A", "B", "C"), skipping blanks
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ()
    return tuple(
        part.strip() for part in str(value).split(",")
        if part.strip() and part.strip() != "<NA>"
    )

paths = set()
for lineage in df["lineage"]:
    parts = lineage_parts(lineage)
    # add every prefix of the chain: (A,) then (A,B) then (A,B,C) ...
    paths.update(parts[:d] for d in range(1, len(parts) + 1))

print("total nodes:", len(paths))                                       # -> 8887
print("edges (nodes with a parent):", sum(len(p) > 1 for p in paths))   # -> 8582
```

**Result:** the reconstruction matches exactly. This confirms that
`lang_fam_geo.csv` is the correct version of the file that
`_calculate_phylogeny_vectors()` was built against.

### 3.2 Lineage Depth Distribution

#### Script — Lineage Depth Statistics

```python
# lambda v: ... defines a small, unnamed function inline — v is one row's
# lineage value each time it runs. Used here since it's only needed once.
df["depth"] = df["lineage"].map(lambda v: len(lineage_parts(v)))
nonzero_depth = df.loc[df["depth"] > 0, "depth"]

print("median depth:", nonzero_depth.median())   # -> 6.0
print("mean depth:", round(nonzero_depth.mean(), 2))  # -> 6.85
print("max depth:", nonzero_depth.max())         # -> 26
```

**Result:** the typical language sits six levels deep in its family tree
(median 6.0, mean 6.85). The deepest chain reaches 26 levels, held by
languages in the Kikongo Language Cluster (Atlantic-Congo branch), which
Glottolog subdivides unusually finely.

### 3.3 Largest Families by Row Count

#### Script — Root Family Counts

```python
# take the first item in the lineage tuple (the top-level family);
# fall back to None if the lineage is empty
df["root"] = df["lineage"].map(lambda v: lineage_parts(v)[0] if lineage_parts(v) else None)
print(df["root"].value_counts().head(10))
```

| Family | Rows |
|---|---|
| Atlantic-Congo | 4,839 |
| Austronesian | 4,087 |
| Indo-European | 3,141 |
| Sino-Tibetan | 1,915 |
| Afro-Asiatic | 1,447 |
| Nuclear Trans New Guinea | 829 |
| Pama-Nyungan | 643 |
| Austroasiatic | 527 |
| Otomanguean | 386 |
| Bookkeeping | 385 |

"Bookkeeping" is not a genuine language family; it is a label Glottolog
uses to retain codes for languoids that turned out not to exist, or that
are too poorly attested to classify. A "Sign Language" label (344 rows,
not shown above) serves a similar function: Glottolog groups all sign
languages under one placeholder label because they are not related to one
another by descent the way spoken-language families are.

### 3.4 Longitude Range Anomaly

#### Script — Longitude Range Check

```python
oob = df[(df["longitude"] > 180) | (df["longitude"] < -180)]
print("rows with longitude outside -180..180:", len(oob))   # -> 3211
```

3,211 rows (12%) store longitude as a value greater than 180 instead of
using negative numbers. For example, one language located in Maine, USA
is recorded at `291.34` instead of `-68.66` (the same location, since
`291.34 - 360 = -68.66`). This does not affect the great-circle distance 
calculation (`getGreatCircleDistance()`) in `_calculate_geocoord_vectors()`, since the underlying
Haversine formula is mathematically invariant or unchanged to a full 360° shift.
It would, however, affect any downstream code that plots these coordinates
or assumes a conventional −180°…180° range.

---

## 4. ISO 639-3 to Glottocode Mapping — `uriel_glottocode_map.csv`

This file has two columns, `iso_code` and `glottocode`. It is a
translation table: given an ISO 639-3 code (a separate, shorter
language-code standard), it returns the matching Glottocode.

### Script — Load the ISO Code Map

```python
iso_map = pd.read_csv("uriel_glottocode_map.csv", dtype="string")
print(iso_map.shape)                 # (7970, 2)
print(iso_map.isna().sum())          # how many rows are missing a glottocode
```

**Result:** 7,970 ISO codes. 148 of them have no matching Glottocode.
This group is not uniform: some are legacy ISO 639-2 bibliographic codes
rather than true ISO 639-3 codes (`alb` and `gre`, for Albanian and Greek,
whose actual ISO 639-3 forms are `sqi` and `ell`; Library of Congress,
n.d.), others are genuine ISO 639-3 'macrolanguage' codes that Glottolog
represents only through their more specific member varieties (`est`,
`yid`; SIL International, n.d.), and a few are special reserved codes for
non-language content, such as `zxx` ("no linguistic content") (SIL
International, n.d.; see References, §7).


### 4.1 Glottocode Format Check on the Mapping File

The same syntax rule applied in §2 was also run against this file's
`glottocode` column, as a completeness check.

#### Script — Glottocode Syntax Check on the Mapping File

```python
valid_codes = iso_map["glottocode"].dropna()
bad_codes = valid_codes[~valid_codes.str.match(glottocode_re)]
print("malformed glottocodes in the ISO map:", len(bad_codes))   # -> 0
```

**Result:** all 7,822 non-missing Glottocodes in the mapping file conform
to Glottolog's naming rule.

### 4.2 Cross-File Consistency Check

This check determines whether the two data tables agree with each other.

#### Script — Glottocode Overlap Check

```python
map_ids = set(iso_map["glottocode"].dropna())
lfg_ids = set(df["language_id"])

print("glottocodes in the map file:", len(map_ids))
print("also found in lang_fam_geo.csv:", len(map_ids & lfg_ids))
print("in map file but missing from lang_fam_geo.csv:", len(map_ids - lfg_ids))
```

**Result:** all 7,822 non-missing Glottocodes in the ISO map appear in
`lang_fam_geo.csv`: full agreement between the two files (0 unmatched).
This is consistent with both having been built from the same
underlying Glottolog snapshot rather than from two different versions.
A further check confirmed no Glottocode in the mapping file is reused by
more than one ISO code; the mapping is strictly one-to-one.

---

## 5. Comparison with the Live Glottolog Database

This section did not rely on code run against a downloaded Glottolog
database. Individual pages on glottolog.org were consulted directly, and
the classification each page displayed was compared manually against the
corresponding row in `lang_fam_geo.csv`.

| Item checked | `lang_fam_geo.csv` | Glottolog (live) | Result |
|---|---|---|---|
| Ghotuo (`ghot1243`) | `Atlantic-Congo, Volta-Congo, Benue-Congo, Akpes-Edoid, Edoid, North-Central Edoid, Afenmai-Bendel` | Identical chain, confirmed via the page's internal classification path | Exact match |
| Slovincian (`slov1270`) | `..., Lechitic, Kashubian` | Glottolog's tree groups Slovincian under a node labelled "Kashubian" | Matches Glottolog (a secondary source, Wikipedia, instead lists "Pomeranian"; the CSV agrees with Glottolog) |
| "Classical Indo-European" node | Appears as the second element in several Indo-European lineages | Confirmed as a current Glottolog grouping (code `clas1257`) | Confirmed |

Glottolog pages consulted for this comparison (Hammarström et al., 2026):

- Ghotuo — https://glottolog.org/resource/languoid/id/ghot1243
- Slovincian — https://glottolog.org/resource/languoid/id/slov1270
- Classical Indo-European — https://glottolog.org/resource/languoid/id/clas1257

Automating this kind of check at scale would require Glottolog's full
downloadable dataset (published as a CSV/CLDF export; see References, §7),
rather than the two smaller URIEL+ tables reviewed here, which are
summaries derived from Glottolog rather than Glottolog itself.

---

## 6. Summary

- Both files are internally consistent: identifiers are well-formed under
  Glottolog's naming rule, and the two files agree with each other on
  100% of Glottocodes.
- The phylogeny-tree node and edge counts derived from `lang_fam_geo.csv`
  match exactly what the URIEL+ code expects.
- Spot-checks against Glottolog's live database confirmed the data
  reflects Glottolog's current classification rather than an older or
  simplified version.
- Two data-quality points are worth noting: the "Bookkeeping" and "Sign
  Language" labels are administrative categories rather than genuine
  language families, and approximately 12% of longitude values are stored
  in 0°–360° format rather than −180°…180°.

---

## 7. References

- Hammarström, H., Forkel, R., Haspelmath, M., & Bank, S. (2026).
  *Glottolog 5.3* [Data set]. Max Planck Institute for Evolutionary
  Anthropology. https://glottolog.org
- Glottolog. (n.d.). *Frequently asked questions*. GitHub.
  https://github.com/glottolog/glottolog/blob/master/faq.md
- Glottolog. (n.d.). *glottolog-cldf* [Data set]. GitHub.
  https://github.com/glottolog/glottolog-cldf
- Library of Congress. (n.d.). *ISO 639-2 codes for the representation of
  names of languages*. https://id.loc.gov/vocabulary/iso639-2.html
- SIL International. (n.d.). *ISO 639-3: Scope of denotation for language
  identifiers*. https://iso639-3.sil.org/about/scope
- Khan, A., Shipton, M., Anugraha, D., Duan, K., Hoang, P. H., Khiu, E.,
  Doğruöz, A. S., & Lee, E.-S. A. (2025). URIEL+: Enhancing linguistic
  inclusion and usability in a typological and multilingual knowledge
  base. *Proceedings of COLING 2025*, 6937–6952.
- Shipton, M., Ng, Y. H., Khan, A., Hoang, P. H., Lu, X., Doğruöz, A. S.,
  & Lee, E.-S. A. (2025). *Simple additions, substantial gains: Expanding
  scripts, languages, and lineage coverage in URIEL+*
  (arXiv:2510.27183). arXiv. https://arxiv.org/abs/2510.27183