import pandas as pd
from six import StringIO
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn import tree
import pydotplus
from IPython.display import Image
from sklearn.tree import export_graphviz


col_names = ['category', 'age', 'sex', 'alb', 'alp', 'alt', 'ast', 'bil', 'che', 'chol', 'crea', 'ggt', 'prot']
pima = pd.read_csv("hcv.csv", header=None, names=col_names)
feature_cols = ['age', 'sex', 'alb', 'alp', 'alt', 'ast', 'bil', 'che', 'chol', 'crea', 'ggt', 'prot']
X = pima[feature_cols]
y = pima.category
KFold(n_splits=5, random_state=None, shuffle=False)

k_fold = KFold(n_splits=5)
parameters = {'max_depth': range(3, 60)}
clf = GridSearchCV(tree.DecisionTreeClassifier(), parameters, n_jobs=4, return_train_score=True)

clf.fit(X=X, y=y)
tree_model = clf.best_estimator_
print(clf.best_score_, clf.best_params_)
dot_data = StringIO()
export_graphviz(tree_model, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=feature_cols, class_names=['0', '1', '2', '3', '4'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('hcv.png')
Image(graph.create_png())
