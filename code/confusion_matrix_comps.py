import numpy as np
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

test_comp1_path = 'data/comp1_tagged.wtag'
predictions_comp1_path = 'comp_m1_341241297_206134867.wtag'
test_comp2_path = 'data/comp2_tagged.wtag'
predictions_comp2_path = 'comp_m2_341241297_206134867.wtag'

# Ensure alignment by removing mismatched lines from both files
aligned_true_lines = []
aligned_pred_lines = []
with open(test_comp1_path, 'r') as f_true, open(predictions_comp1_path, 'r') as f_pred:
    true_lines = f_true.readlines()
    pred_lines = f_pred.readlines()

    for i, (true_line, pred_line) in enumerate(zip(true_lines, pred_lines)):
        true_tokens = true_line.strip().split()
        pred_tokens = pred_line.strip().split()
        if len(true_tokens) == len(pred_tokens):
            aligned_true_lines.append(true_line)
            aligned_pred_lines.append(pred_line)
        else:
            print(f"Removing mismatched line {i + 1}: true_tokens={len(true_tokens)}, pred_tokens={len(pred_tokens)}")

# Overwrite the files with only aligned lines
with open(test_comp1_path, 'w') as f_true:
    f_true.writelines(aligned_true_lines)

with open(predictions_comp1_path, 'w') as f_pred:
    f_pred.writelines(aligned_pred_lines)

# Recompute true_labels and predicted_labels after alignment
true_labels = []
predicted_labels = []
for true_line, pred_line in zip(aligned_true_lines, aligned_pred_lines):
    true_labels.extend([pair.split('_')[1] for pair in true_line.strip().split()])
    predicted_labels.extend([pair.split('_')[1] for pair in pred_line.strip().split()])

labels = sorted(list(set(true_labels)))
cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
# export cm as csv file
np.savetxt('confusion_matrix_comp1.csv', cm, delimiter=',', fmt='%d', header=','.join(labels), comments='')

# Plot and save the confusion matrix as an image
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Comp 1')
plt.tight_layout()
plt.savefig('confusion_matrix_comp1.png')
plt.close()
print('Confusion matrix saved as confusion_matrix_comp1.png')

# Analyze label errors for Comp 1
false_positives = cm.sum(axis=0) - np.diag(cm)  # Predicted as label but wrong
true_counts = cm.sum(axis=1)  # True occurrences of each label
false_to_true_ratio = np.divide(false_positives, true_counts, out=np.zeros_like(false_positives, dtype=float), where=true_counts!=0)

label_stats = list(zip(labels, false_positives, true_counts, false_to_true_ratio))
label_stats_sorted = sorted(label_stats, key=lambda x: (-x[1], -x[3]))

print('\nLabel error analysis for Comp 1:')
print(f"{'Label':<10} {'FalsePred':<10} {'TrueCount':<10} {'False/True Ratio':<15}")
for label, fp, tc, ratio in label_stats_sorted:
    print(f"{label:<10} {fp:<10} {tc:<10} {ratio:<15.2f}")

# Analyze top mislabelings for Comp 1
N = 5  # Top N mislabelings to show
mislabel_stats = []
for i, true_label in enumerate(labels):
    row = cm[i]
    row_no_diag = row.copy()
    row_no_diag[i] = 0  # Zero out the diagonal
    if row_no_diag.sum() > 0:
        max_mislabel_idx = np.argmax(row_no_diag)
        max_mislabel_count = row_no_diag[max_mislabel_idx]
        pred_label = labels[max_mislabel_idx]
        percent = max_mislabel_count / true_counts[i] * 100 if true_counts[i] > 0 else 0
        mislabel_stats.append((true_label, pred_label, max_mislabel_count, percent))

mislabel_stats_sorted = sorted(mislabel_stats, key=lambda x: (-x[2], -x[3]))
print(f"\nTop {N} most frequent mislabelings for Comp 1:")
print(f"{'TrueLabel':<12} {'PredAs':<12} {'Count':<8} {'Percent':<10}")
for true_label, pred_label, count, percent in mislabel_stats_sorted[:N]:
    print(f"{true_label:<12} {pred_label:<12} {count:<8} {percent:<10.2f}")

# Ensure alignment by removing mismatched lines from both files
aligned_true_lines = []
aligned_pred_lines = []
with open(test_comp2_path, 'r') as f_true, open(predictions_comp2_path, 'r') as f_pred:
    true_lines = f_true.readlines()
    pred_lines = f_pred.readlines()

    for i, (true_line, pred_line) in enumerate(zip(true_lines, pred_lines)):
        true_tokens = true_line.strip().split()
        pred_tokens = pred_line.strip().split()
        if len(true_tokens) == len(pred_tokens):
            aligned_true_lines.append(true_line)
            aligned_pred_lines.append(pred_line)
        else:
            print(f"Removing mismatched line {i + 1}: true_tokens={len(true_tokens)}, pred_tokens={len(pred_tokens)}")

# Overwrite the files with only aligned lines
with open(test_comp2_path, 'w') as f_true:
    f_true.writelines(aligned_true_lines)

with open(predictions_comp2_path, 'w') as f_pred:
    f_pred.writelines(aligned_pred_lines)

# Recompute true_labels and predicted_labels after alignment
true_labels = []
predicted_labels = []
for true_line, pred_line in zip(aligned_true_lines, aligned_pred_lines):
    true_labels.extend([pair.split('_')[1] for pair in true_line.strip().split()])
    predicted_labels.extend([pair.split('_')[1] for pair in pred_line.strip().split()])

labels = sorted(list(set(true_labels)))
cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
# export cm as csv file
np.savetxt('confusion_matrix_comp2.csv', cm, delimiter=',', fmt='%d', header=','.join(labels), comments='')

# Plot and save the confusion matrix as an image
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Comp 2')
plt.tight_layout()
plt.savefig('confusion_matrix_comp2.png')
plt.close()
print('Confusion matrix saved as confusion_matrix_comp2.png')

# Analyze label errors for Comp 2
false_positives = cm.sum(axis=0) - np.diag(cm)
true_counts = cm.sum(axis=1)
false_to_true_ratio = np.divide(false_positives, true_counts, out=np.zeros_like(false_positives, dtype=float), where=true_counts!=0)

label_stats = list(zip(labels, false_positives, true_counts, false_to_true_ratio))
label_stats_sorted = sorted(label_stats, key=lambda x: (-x[1], -x[3]))

print('\nLabel error analysis for Comp 2:')
print(f"{'Label':<10} {'FalsePred':<10} {'TrueCount':<10} {'False/True Ratio':<15}")
for label, fp, tc, ratio in label_stats_sorted:
    print(f"{label:<10} {fp:<10} {tc:<10} {ratio:<15.2f}")

# Analyze top mislabelings for Comp 2
N = 5
mislabel_stats = []
for i, true_label in enumerate(labels):
    row = cm[i]
    row_no_diag = row.copy()
    row_no_diag[i] = 0
    if row_no_diag.sum() > 0:
        max_mislabel_idx = np.argmax(row_no_diag)
        max_mislabel_count = row_no_diag[max_mislabel_idx]
        pred_label = labels[max_mislabel_idx]
        percent = max_mislabel_count / true_counts[i] * 100 if true_counts[i] > 0 else 0
        mislabel_stats.append((true_label, pred_label, max_mislabel_count, percent))

mislabel_stats_sorted = sorted(mislabel_stats, key=lambda x: (-x[2], -x[3]))
print(f"\nTop {N} most frequent mislabelings for Comp 2:")
print(f"{'TrueLabel':<12} {'PredAs':<12} {'Count':<8} {'Percent':<10}")
for true_label, pred_label, count, percent in mislabel_stats_sorted[:N]:
    print(f"{true_label:<12} {pred_label:<12} {count:<8} {percent:<10.2f}")