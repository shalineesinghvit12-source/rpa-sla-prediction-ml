"""
03_cluster_classify_evaluate.py
Clusters RPA process runs, trains classifiers to predict SLA outcome,
and evaluates the better model with confusion matrix, precision/recall,
and ROC/AUC.
Converted from R (03_cluster_classify_evaluate.R) to Python.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, auc

BASE = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, "data")
PLOTS_DIR = os.path.join(BASE, "plots")
OUTPUTS_DIR = os.path.join(BASE, "outputs")
for d in (DATA_DIR, PLOTS_DIR, OUTPUTS_DIR):
    os.makedirs(d, exist_ok=True)

np.random.seed(441)

log_path = os.path.join(OUTPUTS_DIR, "03_log.txt")
log_file = open(log_path, "w")


def log(*args):
    msg = " ".join(str(a) for a in args)
    print(msg)
    log_file.write(msg + "\n")


df = pd.read_csv(os.path.join(DATA_DIR, "rpa_process_runs_preprocessed.csv"))

################################################################
log("================ e. CLUSTERING ================")

z_cols = [c for c in df.columns if c.startswith("z_")]
cluster_data = df[z_cols].values
log("Clustering on standardized numeric features:", ", ".join(z_cols), "\n")

# Choose k via elbow (within-cluster SS) and silhouette score
wss = []
for k in range(1, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=441).fit(cluster_data)
    wss.append(km.inertia_)

plt.figure(figsize=(8.5, 6))
plt.plot(range(1, 9), wss, marker="o", color="#4C72B0")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Total within-cluster SS")
plt.title("Elbow Method for Choosing k")
plt.savefig(os.path.join(PLOTS_DIR, "elbow_plot.png"), dpi=130)
plt.close()

sil_scores = {}
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=10, random_state=441).fit(cluster_data)
    sil_scores[k] = silhouette_score(cluster_data, km.labels_)

log("Silhouette scores for k=2..6:")
log({k: round(v, 3) for k, v in sil_scores.items()})
best_k = max(sil_scores, key=sil_scores.get)
log("\nChosen k (max avg silhouette):", best_k)

km_final = KMeans(n_clusters=best_k, n_init=25, random_state=441).fit(cluster_data)
df["cluster"] = km_final.labels_ + 1  # 1-indexed to mirror R
log("\nCluster sizes:")
log(df["cluster"].value_counts().sort_index().to_string())

# Compare clusters to known labels
log("\nCluster vs Department:")
log(pd.crosstab(df["cluster"], df["department"]).to_string())
log("\nCluster vs SLA outcome:")
log(pd.crosstab(df["cluster"], df["sla_met"]).to_string())
log("\nCluster centers (standardized units):")
centers_df = pd.DataFrame(km_final.cluster_centers_, columns=z_cols, index=range(1, best_k + 1))
log(centers_df.round(2).to_string())

# PCA projection for visualization
pca = PCA()
pcs = pca.fit_transform(cluster_data)
var_exp = np.round(100 * pca.explained_variance_ratio_[:2], 1)
log(f"\nPCA variance explained - PC1: {var_exp[0]}% PC2: {var_exp[1]}%")

colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
plt.figure(figsize=(9.5, 7))
for c in range(1, best_k + 1):
    mask = df["cluster"] == c
    plt.scatter(pcs[mask, 0], pcs[mask, 1], color=colors[(c - 1) % len(colors)], label=f"Cluster {c}", s=18)
plt.xlabel(f"PC1 ({var_exp[0]}%)")
plt.ylabel(f"PC2 ({var_exp[1]}%)")
plt.title(f"PCA Projection Colored by K-means Cluster (k={best_k})")
plt.legend(loc="upper right", fontsize=8)
plt.savefig(os.path.join(PLOTS_DIR, "pca_clusters.png"), dpi=130)
plt.close()

plt.figure(figsize=(9.5, 7))
for lbl, color in [("Yes", "#55A868"), ("No", "#C44E52")]:
    mask = df["sla_met"] == lbl
    plt.scatter(pcs[mask.values, 0], pcs[mask.values, 1], color=color,
                label=("SLA Met" if lbl == "Yes" else "SLA Breached"), s=18)
plt.xlabel(f"PC1 ({var_exp[0]}%)")
plt.ylabel(f"PC2 ({var_exp[1]}%)")
plt.title("PCA Projection Colored by Actual SLA Outcome")
plt.legend(loc="upper right", fontsize=8)
plt.savefig(os.path.join(PLOTS_DIR, "pca_sla.png"), dpi=130)
plt.close()

df.to_csv(os.path.join(DATA_DIR, "rpa_process_runs_clustered.csv"), index=False)

################################################################
log("\n\n================ f. CLASSIFICATION ================")

sla_map = {"No": 0, "Yes": 1}
df["sla_met_bin"] = df["sla_met"].map(sla_map)

dummy_prefixes = ("department_", "region_", "process_type_", "day_of_week_")
feat_cols = z_cols + ["complexity_num"] + [c for c in df.columns if c.startswith(dummy_prefixes)]
model_df = df[feat_cols + ["sla_met_bin"]].copy()

X = model_df[feat_cols].astype(float)
y = model_df["sla_met_bin"].astype(int)

# Train / test split (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=441)
log("Train size:", len(X_train), " Test size:", len(X_test))
log("Train class balance:")
log(y_train.value_counts().to_string())

## --- Classifier 1: Logistic Regression (with simple backward stepwise-style tuning) ---
# Backward elimination based on p-values, using statsmodels for coefficient significance.
import statsmodels.api as sm

X_train_sm = sm.add_constant(X_train)
remaining = list(X_train_sm.columns)
while True:
    model = sm.Logit(y_train, X_train_sm[remaining]).fit(disp=0)
    pvals = model.pvalues.drop("const", errors="ignore")
    worst_p = pvals.max() if len(pvals) else 0
    if worst_p > 0.05 and len(pvals) > 1:
        worst_feat = pvals.idxmax()
        remaining.remove(worst_feat)
    else:
        break

glm_tuned = sm.Logit(y_train, X_train_sm[remaining]).fit(disp=0)
log("\nLogistic regression: features retained after backward elimination:")
log([c for c in remaining if c != "const"])

X_test_sm = sm.add_constant(X_test, has_constant="add")[remaining]
glm_prob = glm_tuned.predict(X_test_sm)
glm_pred = (glm_prob > 0.5).astype(int)
glm_acc = (glm_pred.values == y_test.values).mean()
log(f"\nLogistic Regression test accuracy: {glm_acc:.4f}")

## --- Classifier 2: Decision Tree, tuned via cross-validated cost-complexity pruning (ccp_alpha) ---
base_tree = DecisionTreeClassifier(random_state=441)
path = base_tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

cv_scores = []
for alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=441, ccp_alpha=alpha)
    scores = cross_val_score(clf, X_train, y_train, cv=10)
    cv_scores.append(scores.mean())

best_alpha = ccp_alphas[int(np.argmax(cv_scores))]
log(f"\nDecision tree: best ccp_alpha from 10-fold cross-validation: {best_alpha:.5f}")

tree_tuned = DecisionTreeClassifier(random_state=441, ccp_alpha=best_alpha).fit(X_train, y_train)

plt.figure(figsize=(8.5, 6))
plt.plot(ccp_alphas, cv_scores, marker="o", color="#4C72B0")
plt.xlabel("ccp_alpha")
plt.ylabel("10-fold CV accuracy")
plt.title("Cross-validated Accuracy vs Tree Complexity")
plt.savefig(os.path.join(PLOTS_DIR, "tree_cv_plot.png"), dpi=130)
plt.close()

plt.figure(figsize=(11, 8))
plot_tree(tree_tuned, feature_names=feat_cols, class_names=["No", "Yes"],
          filled=True, fontsize=6)
plt.title("Pruned Decision Tree for SLA Outcome")
plt.savefig(os.path.join(PLOTS_DIR, "decision_tree.png"), dpi=130)
plt.close()

tree_pred = tree_tuned.predict(X_test)
tree_prob = tree_tuned.predict_proba(X_test)[:, 1]
tree_acc = (tree_pred == y_test.values).mean()
log(f"Decision Tree test accuracy: {tree_acc:.4f}")

log("\n---- Accuracy comparison ----")
log(f"Logistic Regression: {glm_acc:.4f}")
log(f"Decision Tree:       {tree_acc:.4f}")

# pick the better one for full evaluation
if glm_acc >= tree_acc:
    better, best_pred, best_prob = "Logistic Regression", glm_pred.values, glm_prob.values
else:
    better, best_pred, best_prob = "Decision Tree", tree_pred, tree_prob

log("\nBetter model selected for evaluation stage:", better)

################################################################
log("\n\n================ g. EVALUATION ================")

# 1. 2x2 confusion matrix
cm = confusion_matrix(y_test, best_pred, labels=[1, 0])  # rows/cols ordered Yes(1), No(0)
TP, FP = cm[0, 0], cm[1, 0]
FN, TN = cm[0, 1], cm[1, 1]
log("Confusion Matrix (rows=Actual, cols=Predicted) [Yes, No]:")
log(cm)
log(f"\nTP={TP}  FP={FP}  FN={FN}  TN={TN}")

# 2. Precision & Recall (manual)
precision = TP / (TP + FP) if (TP + FP) else float("nan")
recall = TP / (TP + FN) if (TP + FN) else float("nan")
accuracy = (TP + TN) / cm.sum()
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
log(f"\nAccuracy:  {accuracy:.4f}")
log(f"Precision (SLA Met): {precision:.4f}")
log(f"Recall (SLA Met):    {recall:.4f}")
log(f"F1-score:  {f1:.4f}")

# 3. ROC curve + AUC
fpr, tpr, _ = roc_curve(y_test, best_prob)
roc_auc = auc(fpr, tpr)
log(f"\nAUC: {roc_auc:.4f}")

plt.figure(figsize=(8.5, 7.5))
plt.plot(fpr, tpr, lw=2, color="#4C72B0")
plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - {better} (AUC = {roc_auc:.3f})")
plt.savefig(os.path.join(PLOTS_DIR, "roc_curve.png"), dpi=130)
plt.close()

log("\n---- Interpretation ----")
log("Accuracy alone can be misleading when classes are imbalanced (train balance shown above).")
log("Precision tells us: of runs predicted to meet SLA, what fraction actually did.")
log("Recall tells us: of runs that actually met SLA, what fraction the model correctly caught.")
log("The ROC/AUC summarizes performance across ALL thresholds, not just 0.5, showing the")
log("true trade-off between catching SLA-met runs (TPR) vs. false alarms (FPR).")

# Save key metrics to a small results file for report/PPT use
res = pd.DataFrame({
    "metric": ["k_chosen_clustering", "glm_accuracy", "tree_accuracy", "better_model",
               "confusion_TP", "confusion_FP", "confusion_FN", "confusion_TN",
               "precision", "recall", "f1", "auc"],
    "value": [best_k, round(glm_acc, 4), round(tree_acc, 4), better,
              TP, FP, FN, TN, round(precision, 4), round(recall, 4), round(f1, 4), round(roc_auc, 4)],
})
res.to_csv(os.path.join(OUTPUTS_DIR, "key_results.csv"), index=False)

log_file.close()
print(f"Done. See {log_path} and {os.path.join(OUTPUTS_DIR, 'key_results.csv')}")
