import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn import model_selection


def train_svm_model (fpath):
  data = np.loadtxt(fpath, delimiter=',', dtype=int)
  num_col = len (data[0])
  X = data[:, 1:num_col]
  Y = data[:, 0]


  #svm_model = SVC (kernel='linear', probability=True)
  svm_model = SVC (probability=True)
  svm_model.fit (X, Y)

  model_fpath = "../../svm_trained.sav"
  pickle.dump (svm_model, open(model_fpath, 'wb'))

  #Temporary testing the performance while training
  test_size = 0.25
  seed = 7
  X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
  svm_model.fit (X_train, Y_train)
  score = svm_model.score(X_test, Y_test)
  print(score)


if __name__ == "__main__":
  train_file = "../../feature_vector.input"
  train_svm_model (train_file)
