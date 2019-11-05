import sys
sys.path.insert(1, '../..')
import utils
import numpy as np
import pickle

def test_dt_model (mp_obj, top_obj):
  lst = []
  lst.append (mp_obj)
  data = utils.create_features_handler (None, lst, top_obj, 0)
  num_col = data.shape[0]
  X = data[1:num_col]

  fpath = "dt_trained.sav"
  dt_model = pickle.load (open(fpath, 'rb'))

  Y_prob = dt_model.predict_proba (X.reshape(1, -1))

  return Y_prob[0][1]
