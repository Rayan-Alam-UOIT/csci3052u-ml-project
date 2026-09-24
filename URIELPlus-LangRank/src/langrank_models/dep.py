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
TIMING_LOG_PATH = f"langrank_dep_timing_log_{args.mode}.txt"
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
data = pd.read_csv('src\\csv_datasets\\dep.csv')
logo = LeaveOneGroupOut()

log_step_time("Load data")

# Define feature sets
features_all = [
    'Word overlap', 'Transfer lang dataset size', 'Target lang dataset size',
    'Transfer over target size ratio', 'Transfer lang TTR', 'Target lang TTR',
    'Transfer target TTR distance', 'GENETIC', 'SYNTACTIC', 'FEATURAL',
    'PHONOLOGICAL', 'INVENTORY', 'GEOGRAPHIC', 'MORPHOLOGICAL', 'SCRIPT'
]

features_lang = [
    'GENETIC', 'SYNTACTIC', 'FEATURAL', 'PHONOLOGICAL', 'INVENTORY', 'GEOGRAPHIC', 'MORPHOLOGICAL', 'SCRIPT'
]

# Choose feature list based on mode
if args.mode == 'all':
    features = features_all
else:
    features = features_lang

data['relevance'] = 0

def ndcg1(rel_true, rel_pred):

    int_rel = rel_true.astype(int)
    rel_index = np.argsort(rel_pred)[::-1]
    true_indexed = int_rel[rel_index]
    int_rel[::-1].sort()
    dcg = (2**true_indexed[0] - 1)/np.log2(2) + (2**true_indexed[1] - 1)/np.log2(3) + (2**true_indexed[2] - 1)/np.log2(4)
    idcg = (2**int_rel[0] - 1)/np.log2(2) + (2**int_rel[1] - 1)/np.log2(3) + (2**int_rel[2] - 1)/np.log2(4)
    return dcg / idcg

# assign relevances from 10 to 0
for source_lang in data['Target lang'].unique():
    source_lang_data = data[data['Target lang'] == source_lang].copy()
    source_lang_data['rank'] = source_lang_data['Accuracy'].rank(method='dense', ascending=False)
    top_indices = source_lang_data[source_lang_data['rank'] <= 10].index
    data.loc[top_indices, 'relevance'] = 11 - source_lang_data.loc[top_indices, 'rank']

log_step_time("Assign relevances")

groups = data['Target lang']
ndcg_scores = []

# Parameters for ranker
ranker = LGBMRanker(
    boosting_type='gbdt',
    objective ='lambdarank',
    n_estimators=100,
    metric='lambdarank',
    num_leaves=16,
    min_data_in_leaf=5,
    verbose=-1
)

query = [29] * 29

#leave one language out
for fold_num, (train_idx, test_idx) in enumerate(logo.split(data, groups=groups), start=1):
    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]

    train_X = train_data[features]
    train_y = train_data['relevance']
    test_X = test_data[features]
    test_y = test_data['relevance']

    # Train the model
    ranker.fit(train_X, train_y, group=query, eval_set=[(test_X, test_y)], eval_group=[[29]], eval_at=[3])

    # Predict and evaluate NDCG@3
    y_pred = ranker.predict(test_X)
    score = ndcg_score([test_y],[y_pred],k=3)
    ndcg_scores.append(score)

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