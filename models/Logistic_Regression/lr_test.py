import sys
sys.path.insert(1, '../..')
import utils
import numpy as np
import pickle

def test_lr_model (mp_obj, top_obj):
  lst = []
  lst.append (mp_obj)
  data = utils.create_features_handler (None, lst, top_obj, 0)
  num_col = len(X[0])

  X = data[:,1:num_col]
  fpath = "../../lr_trained.sav"
  lr_model = pickle.load (open(fpath, 'rb'))

  Y_prob = lr_model.predict_proba (X)

  return Y_prob
