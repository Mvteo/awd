
import pandas as pd
from six import StringIO
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn import tree
import pydotplus
from IPython.display import Image
from sklearn.tree import export_graphviz

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

# for i in range(2, 15):
#     cv = KFold(n_splits=i, random_state=10, shuffle=True)
#     scores = []
#     for train_index, test_index in cv.split(x):
#         X_train, X_test, y_train, y_test = x[test_index], x[train_index], y[test_index], y[train_index]
#         dtc.fit(X_train, y_train)
#         scores.append(dtc.score(X_test, y_test))
#     print("splits=", i, "scores=", max(scores))

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

# import pandas as pd
# import pydotplus
# from IPython.display import Image
# from six import StringIO
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import KFold
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.tree import export_graphviz
#
# if name == 'main':
#     col_names = ['category', 'age', 'sex', 'alb', 'alp', 'alt', 'ast', 'bil', 'che', 'chol', 'crea', 'ggt', 'prot']
#     pima = pd.read_csv("hcv.csv", header=None, names=col_names)
#     feature_cols = ['age', 'sex', 'alb', 'alp', 'alt', 'ast', 'bil', 'che', 'chol', 'crea', 'ggt', 'prot']
#     X = pima[feature_cols]
#     y = pima.category
#     KFold(n_splits=5, random_state=None, shuffle=False)
#
#     k_fold = KFold(n_splits=5)
#     parameters = {'max_depth': range(3, 60)}
#     clf = GridSearchCV(DecisionTreeClassifier(), parameters, n_jobs=4)
#     clf.fit(X=X, y=y)
#     tree_model = clf.bestestimator
#     print(clf.bestscore, clf.bestparams)
#     dot_data = StringIO()
#     export_graphviz(tree_model, out_file=dot_data,
#                     filled=True, rounded=True,
#                     special_characters=True, feature_names=feature_cols, class_names=['0', '1', '2', '3', '4'])
#     graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#     graph.write_png('hcv.png')
#     Image(graph.create_png())