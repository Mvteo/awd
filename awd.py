import graphviz
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

dtc = tree.DecisionTreeClassifier(criterion="entropy",splitter="random")

x = cpuNormalizedData[feature_cols]
y = cpuNormalizedData["NAME"]
parameters = {'max_depth': range(3, 10)}

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


