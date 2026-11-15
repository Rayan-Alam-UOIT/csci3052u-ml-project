# Calculates linguistic distances between language pairs from datasets (dep, el, mt, pos) using URIEL+, outputs results to CSV files.

# Import URIEL+ for calculating language distances and the csv module for writing output files
from urielplus import urielplus as uriel
import pandas as pd
import csv
import time
from pathlib import Path
import zipfile

# Timing setup: logs how long each step takes, and the running total, to both the console and a log file.
CURRENT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIRECTORY.parent

TIMING_LOG_PATH = CURRENT_DIRECTORY / "calculate_distances_timing_log.txt"
start_time = time.time()
last_checkpoint = start_time

# Start with a fresh log file for this run
with open(TIMING_LOG_PATH, "w", encoding="utf-8") as f:
    f.write("Timing log\n")
    f.write("=" * 40 + "\n")


def log_step_time(step_name):
    global last_checkpoint
    now = time.time()
    step_duration = now - last_checkpoint
    total_duration = now - start_time
    last_checkpoint = now

    message = (
        f"[{step_name}] step time: {step_duration:.2f}s | "
        f"total elapsed: {total_duration:.2f}s"
    )
    print(message)
    with open(TIMING_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(message + "\n")


# Initialize the URIEL+ system and enable caching for efficiency
u = uriel.URIELPlus()
log_step_time("Initialize URIELPlus")
# u.reset()
# log_step_time("Reset")
# u.set_cache(True)
# log_step_time("Set cache")
# u.integrate_databases()        # Integrate all linguistic sources
# log_step_time("Integrate databases")
# u.softimpute_imputation()      # Fill missing values using soft
# log_step_time("Softimpute imputation")


# Dictionary mapping the language codes (2-letter) from dep.csv to their corresponding ISO 639-3 (3-digit) codes
ISO_639_3_MAPPING = {
    "cs": "ces",  # Czech
    "ru": "rus",  # Russian
    "no": "nor",  # Norwegian
    "es": "spa",  # Spanish
    "ko": "kor",  # Korean
    "et": "ekk",  # Estonian
    "pl": "pol",  # Polish
    "nl": "nld",  # Dutch
    "pt": "por",  # Portuguese
    "la": "lat",  # Latin
    "fr": "fra",  # French
    "de": "deu",  # German
    "hi": "hin",  # Hindi
    "ca": "cat",  # Catalan
    "it": "ita",  # Italian
    "en": "eng",  # English
    "fi": "fin",  # Finnish
    "bg": "bul",  # Bulgarian
    "sl": "slv",  # Slovenian
    "sk": "slk",  # Slovak
    "ro": "ron",  # Romanian
    "hr": "hrv",  # Croatian
    "ar": "ara",  # Arabic
    "lv": "lav",  # Latvian
    "he": "heb",  # Hebrew
    "uk": "ukr",  # Ukrainian
    "id": "ind",  # Indonesian
    "da": "dan",  # Danish
    "sv": "swe",  # Swedish
    "zh": "zho",  # Chinese (Mandarin)
    "kk": "kaz",  # Kazakh
    "hy": "hye",  # Armenian
    "lt": "lit",  # Lithuanian
    "be": "bel",  # Belarusian
    "mr": "mar",  # Marathi
    "ta": "tam",  # Tamil
    "ga": "gle",  # Irish
    "hu": "hun",  # Hungarian
    "te": "tel",  # Telugu
    "af": "afr",  # Afrikaans
    "vi": "vie",  # Vietnamese
    "ug": "uig",  # Uighur
    "el": "ell",  # Greek
    "gl": "glg",  # Galician
    "sr": "srp",  # Serbian
    "tr": "tur",  # Turkish
    "ur": "urd",  # Urdu
    "cu": "chu",  # Church Slavik
    "fa": "pes",  # Persian
    "eu": "eus",  # Basque
    "ja": "jpn",  # Japanese
    "am": "amh",  # Amharic
    "th": "tha",  # Thai
    "yo": "yor",  # Yoruba
    "tl": "tgl",  # Tagalog
    "br": "bre",  # Breton
    "sa": "san",  # Sanskrit
    "fo": "fao",  # Faroese
}

# Dictionary mapping the ISO 639-3 codes in URIEL but without equivalent glottocodes in URIEL+, and the glottocodes that were used as replacements
MANUAL_CODE_FIXES = {
    "alb": "alba1267",  # Albanian
    "ara": "stan1318",  # Arabic
    "aze": "nort2697",  # Azerbaijani
    "zho": "mand1415",  # Chinese (Mandarin)
    "est": "esto1258",  # Estonian
    "msa": "stan1306",  # Malay
    "orm": "east2652",  # Oromo
    "fas": "west2369",  # Persian
    "swa": "swah1253"   # Swahili
}

map_df = pd.read_csv(
    CURRENT_DIRECTORY / "urielplus" / "database" / "urielplus_csvs" / "uriel_glottocode_map.csv"
)
iso_to_glottocode = dict(zip(map_df['iso_code'], map_df['glottocode']))
iso_to_glottocode.update(MANUAL_CODE_FIXES)

# All eight distances, preserving the original output order
DISTANCES = ["genetic", "syntactic", "featural", "phonological", "inventory", "geographic", "morphological", "script"]
# Read these from fixed_distances.zip
FIXED_DISTANCES = ["genetic", "geographic", "script"]
# Calculate only these five distances with URIEL+
VARIABLE_DISTANCES = ["syntactic", "featural", "phonological", "inventory", "morphological"]

FIXED_ZIP_PATH = CURRENT_DIRECTORY / "fixed_distances.zip"

# Column names and original language-code conversion rules
TASK_SPECS = {
    "dep": ("Target lang", "Transfer lang", True),
    "el": ("Target lang", "Transfer lang", False),
    "mt": ("Source lang", "Transfer lang", False),
    "pos": ("Task lang", "Aux lang", True),
}

assert FIXED_ZIP_PATH.exists(), f"Fixed-distance archive not found: {FIXED_ZIP_PATH}"

def load_fixed_distances(archive, task, col1, col2):
    possible_names = {f"{task}_distances.csv", f"{task}_fixed_distances.csv", f"fixed_{task}_distances.csv", f"{task}.csv"}

    matches = [member for member in archive.namelist() if member.replace("\\", "/").split("/")[-1].lower() in possible_names]

    if len(matches) != 1:
        raise ValueError(f"Expected one cached CSV for {task}, but found {matches}. ZIP contents: {archive.namelist()}")

    # Read directly from the ZIP, without extracting files
    with archive.open(matches[0]) as f:
        fixed_df = pd.read_csv(f, sep=None, engine="python", encoding="utf-8-sig")

    fixed_df.columns = [str(column).strip() for column in fixed_df.columns]

    required = ([col1, col2] + [d.upper() for d in FIXED_DISTANCES])

    missing_columns = [column for column in required if column not in fixed_df.columns]

    if missing_columns:
        raise ValueError(f"Missing columns in cached {task} CSV: {missing_columns}")

    lookup = {}

    for row in fixed_df[required].itertuples(index=False, name=None):
        lang1, lang2, *raw_values = row

        if pd.isna(lang1) or pd.isna(lang2):
            raise ValueError(f"Missing language code in cached {task} CSV")

        key = (str(lang1).strip(), str(lang2).strip())

        # Preserve missing numeric values as empty CSV cells
        values = {
            distance: (
                None if pd.isna(value) else float(value)
            )
            for distance, value in zip(FIXED_DISTANCES, raw_values)
        }

        if key in lookup and lookup[key] != values:
            raise ValueError(f"Conflicting cached distances for {key} in {task}")

        lookup[key] = values

    print(f"Loaded {len(lookup)} cached pairs for {task} from {matches[0]}")

    return lookup

def convert_pair(lang1, lang2, use_iso_mapping):
    lang1 = str(lang1).strip()
    lang2 = str(lang2).strip()

    if use_iso_mapping:
        lang1 = ISO_639_3_MAPPING.get(lang1, lang1)
        lang2 = ISO_639_3_MAPPING.get(lang2, lang2)

    lang1 = iso_to_glottocode.get(lang1, lang1)
    lang2 = iso_to_glottocode.get(lang2, lang2)

    return (lang1, lang2)

variable_cache = {}

def calculate_variable_distances(lang_pair):
    key = tuple(lang_pair)

    if key not in variable_cache:
        result = list(u.new_distance(VARIABLE_DISTANCES, list(lang_pair)))

        if len(result) != len(VARIABLE_DISTANCES):
            raise ValueError(f"Unexpected distance count for {lang_pair}")

        variable_cache[key] = dict(zip(VARIABLE_DISTANCES, result))

    return variable_cache[key]

log_step_time("Load mappings")

# Process all four tasks
with zipfile.ZipFile(FIXED_ZIP_PATH, "r") as archive:
    for task, (col1, col2, use_iso_mapping) in TASK_SPECS.items():

        print(f"\nProcessing {task}...")

        input_path = PROJECT_ROOT / "experiment_csvs" / "URIEL" / f"{task}.csv"

        output_path = CURRENT_DIRECTORY / f"{task}_distances.csv"

        # Load the experiment's fixed distances
        fixed_lookup = load_fixed_distances(archive, task, col1, col2)

        # Load the input language pairs
        df = pd.read_csv(input_path)

        lang_pairs = [convert_pair(lang1, lang2, use_iso_mapping) for lang1, lang2 in df[[col1, col2]].itertuples(index=False, name=None)]

        # Validate every pair BEFORE calculating distances
        missing_pairs = [pair for pair in lang_pairs if pair not in fixed_lookup]

        if missing_pairs:
            raise ValueError(f"{task}: {len(missing_pairs)} pairs are missing from fixed_distances.zip. Examples: {missing_pairs[:5]}. Check that cached language codes are Glottocodes and match the input experiments.")

        # Create the complete distance CSV
        with open(output_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)

            header = ([col1, col2] + [d.upper() for d in DISTANCES])
            writer.writerow(header)

            for lang_pair in lang_pairs:
                # Three distances from the ZIP
                fixed = fixed_lookup[lang_pair]

                # Five distances from URIEL+
                variable = calculate_variable_distances(lang_pair)

                # Merge into one dictionary
                combined = {**fixed, **variable}

                # Preserve the original eight-column order
                distance_values = [combined[d] for d in DISTANCES]

                # Keep the original console output format
                distance_str = ",".join("" if value is None or pd.isna(value) else f"{float(value):.4f}" for value in distance_values)
                print(distance_str)

                writer.writerow(list(lang_pair) + distance_values)

        log_step_time(f"Calculate distances for {task}")

print(f"\nUnique pairs recalculated: {len(variable_cache)}")

total_time = time.time() - start_time

print(f"\nAll done. Total time: {total_time:.2f}s")

with open(TIMING_LOG_PATH, "a", encoding="utf-8") as f:
    f.write(f"Unique pairs recalculated: {len(variable_cache)}\n")
    f.write(f"\nAll done. Total time: {total_time:.2f}s\n")