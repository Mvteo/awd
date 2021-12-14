import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from six import StringIO
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn import tree
import pydotplus
from IPython.display import Image
from sklearn.tree import export_graphviz
from sklearn.metrics import precision_score
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

cols = [
    "NAME",
    "MODEL",
    "MYCT",
    "MMIN",
    "MMAX",
    "CACH",
    "CHMIN",
    "CHMAX",
    "PRP",
    "ERP"
]
cpu = pd.read_csv("machine.csv", header=0)
feature_cols = ["MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"]

cpuNames = cpu[['NAME', 'MODEL']].copy()

cpuNormalizedData = cpu.copy()

#Standaryzacja  z-score
for column in cpu[feature_cols]:
    cpuNormalizedData[column] = (cpuNormalizedData[column] - cpuNormalizedData[column].mean()) / cpuNormalizedData[column].std()

dtc = tree.DecisionTreeClassifier()

x = cpuNormalizedData[feature_cols]
y = cpuNormalizedData["NAME"]
parameters = {'max_depth': range(3, 10)}

score = ["precision"]

k_fold = KFold(n_splits=7, random_state=10, shuffle=True)
clf = GridSearchCV(dtc, parameters, n_jobs=4, cv=k_fold)
clf.fit(X=x, y=y)
print(clf.best_score_, clf.best_params_)
tree_model = clf.best_estimator_
dot_data = StringIO()
export_graphviz(tree_model, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=feature_cols)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('cpu.png')
Image(graph.create_png())



# col_names = ['category', 'age', 'sex', 'alb', 'alp', 'alt', 'ast', 'bil', 'che', 'chol', 'crea', 'ggt', 'prot']
# pima = pd.read_csv("hcv.csv", header=None, names=col_names)
# feature_cols = ['age', 'sex', 'alb', 'alp', 'alt', 'ast', 'bil', 'che', 'chol', 'crea', 'ggt', 'prot']
# X = pima[feature_cols]
# y = pima.category
# KFold(n_splits=5, random_state=None, shuffle=False)
#
# k_fold = KFold(n_splits=5)
# parameters = {'max_depth': range(3, 60)}
# clf = GridSearchCV(tree.DecisionTreeClassifier(), parameters, n_jobs=4, return_train_score=True)
#
# clf.fit(X=X, y=y)
# tree_model = clf.best_estimator_
# print(clf.best_score_, clf.best_params_)
# dot_data = StringIO()
# export_graphviz(tree_model, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=feature_cols, class_names=['0', '1', '2', '3', '4'])
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png('hcv.png')
# Image(graph.create_png())

# results = clf.cv_results_
#
# plt.figure(figsize=(13, 13))
# plt.title("GridSearchCV evaluating using multiple scorers simultaneously", fontsize=16)
#
# plt.xlabel("min_samples_split")
# plt.ylabel("Score")
#
# ax = plt.gca()
# ax.set_xlim(0, 402)
# ax.set_ylim(0.73, 1)
#
# # Get the regular numpy array from the MaskedArray
# X_axis = np.array(results["param_min_samples_split"].data, dtype=float)
#
# for scorer, color in zip(sorted(scoring), ["g", "k"]):
#     for sample, style in (("train", "--"), ("test", "-")):
#         sample_score_mean = results["mean_%s_%s" % (sample, scorer)]
#         sample_score_std = results["std_%s_%s" % (sample, scorer)]
#         ax.fill_between(
#             X_axis,
#             sample_score_mean - sample_score_std,
#             sample_score_mean + sample_score_std,
#             alpha=0.1 if sample == "test" else 0,
#             color=color,
#         )
#         ax.plot(
#             X_axis,
#             sample_score_mean,
#             style,
#             color=color,
#             alpha=1 if sample == "test" else 0.7,
#             label="%s (%s)" % (scorer, sample),
#         )
#
#     best_index = np.nonzero(results["rank_test_%s" % scorer] == 1)[0][0]
#     best_score = results["mean_test_%s" % scorer][best_index]
#
#     # Plot a dotted vertical line at the best score for that scorer marked by x
#     ax.plot(
#         [
#             X_axis[best_index],
#         ]
#         * 2,
#         [0, best_score],
#         linestyle="-.",
#         color=color,
#         marker="x",
#         markeredgewidth=3,
#         ms=8,
#     )
#
#     # Annotate the best score for that scorer
#     ax.annotate("%0.2f" % best_score, (X_axis[best_index], best_score + 0.005))
#
# plt.legend(loc="best")
# plt.grid(False)
# plt.show()
