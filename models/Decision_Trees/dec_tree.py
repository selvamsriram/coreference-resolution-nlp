import numpy as np
import pickle
from sklearn import tree
from sklearn import model_selection


def train_decision_tree (fpath):
  data = np.loadtxt(fpath, delimiter=',', dtype=int)
  num_col = len (data[0])
  X = data[:, 1:num_col]
  Y = data[:, 0]

  dec_model = tree.DecisionTreeClassifier()
  dec_model = dec_model.fit(X, Y)

  model_fpath = "../../dec_tree_trained.sav"
  pickle.dump (dec_model, open(model_fpath, 'wb'))

  test_size = 0.25
  seed = 7
  X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
  dec_model = dec_model.fit(X_train, Y_train)
  print (dec_model.score (X_test, Y_test))


if __name__ == "__main__":
  train_file = "../../feature_vector.input"
  train_decision_tree (train_file)

