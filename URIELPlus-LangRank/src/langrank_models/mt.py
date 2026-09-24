import argparse
import pandas as pd
import numpy as np
import time
from lightgbm import LGBMRanker
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import ndcg_score

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run LANGRANK with different feature sets.")
parser.add_argument('--mode', choices=['lang', 'all'], default='all',
                    help="Choose feature set: 'lang' for language vectors only, 'all' for vectors plus dataset features (default: all).")
args = parser.parse_args()

# Timing setup: logs how long each step takes, and the running total, to both the console and a log file.
# The log filename reflects which feature-set mode is being run.
TIMING_LOG_PATH = f"langrank_mt_timing_log_{args.mode}.txt"
start_time = time.time()
last_checkpoint = start_time

# Start with a fresh log file for this run
with open(TIMING_LOG_PATH, "w", encoding="utf-8") as f:
    f.write(f"Timing log (mode={args.mode})\n")
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


# Load the data
data = pd.read_csv('src\\csv_datasets\\mt.csv') 
groups = data['Source lang']
logo = LeaveOneGroupOut()

log_step_time("Load data")

# Define feature sets
features_all = [
    'Overlap word-level', 'Overlap subword-level', 'Transfer lang dataset size',
    'Target lang dataset size', 'Transfer over target size ratio', 'Transfer lang TTR',
    'Target lang TTR', 'Transfer target TTR distance', 'GENETIC','SYNTACTIC','FEATURAL','PHONOLOGICAL','INVENTORY','GEOGRAPHIC','MORPHOLOGICAL','SCRIPT'
]

features_lang = [
    'GENETIC', 'SYNTACTIC', 'FEATURAL', 'PHONOLOGICAL', 'INVENTORY', 'GEOGRAPHIC','MORPHOLOGICAL','SCRIPT'
]

# Choose feature list based on mode
if args.mode == 'all':
    features = features_all
else:
    features = features_lang

data['relevance'] = 0

# assign relevances from 10 to 0
for source_lang in data['Source lang'].unique():
    source_lang_data = data[data['Source lang'] == source_lang].copy()
    source_lang_data['rank'] = source_lang_data['BLEU'].rank(method='min', ascending=False)
    top_indices = source_lang_data[source_lang_data['rank'] <= 10].index
    data.loc[top_indices, 'relevance'] = 11 - source_lang_data.loc[top_indices, 'rank']

log_step_time("Assign relevances")

groups = data['Source lang']
ndcg_scores = []

# Parameters for LightGBM
ranker = LGBMRanker(
    boosting_type='gbdt',
    objective='lambdarank',
    n_estimators=100,
    metric='lambdarank',
    num_leaves=16,
    min_data_in_leaf=5,
    random_state=50,
    feature_fraction=0.8,
    verbose=-1
)

#leave one language out
for fold_num, (train_idx, test_idx) in enumerate(logo.split(data, groups=groups), start=1):
    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]

    train_X = train_data[features]
    train_y = train_data['relevance']
    test_X = test_data[features]
    test_y = test_data['relevance']

    train_group_sizes = train_data.groupby('Transfer lang').size().tolist()

    # Train the model
    ranker.fit(train_X, train_y, group=train_group_sizes)

    # Predict and evaluate NDCG@3
    y_pred = ranker.predict(test_X)
    ndcg = ndcg_score([test_y], [y_pred], k=3)
    ndcg_scores.append(ndcg)

    log_step_time(f"Leave-one-out fold {fold_num}")

# Output all NDCG@3 score for determining statistical significance
print([round(float(x), 4) for x in ndcg_scores])
# Calculate the average NDCG@3 score
average_ndcg = np.mean(ndcg_scores)
print(f'Average NDCG@3: {round(average_ndcg*100,1)}')

log_step_time("Compute average NDCG@3")

print(f"\nAll done. Total time: {time.time() - start_time:.2f}s")
with open(TIMING_LOG_PATH, "a", encoding="utf-8") as f:
    f.write(f"\nAll done. Total time: {time.time() - start_time:.2f}s\n")