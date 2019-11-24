import sys
sys.path.insert(1, '../..')
import utils
import numpy as np
import pickle

def test_lr_model (mp_obj, top_obj):
  data, valid = utils.single_mp_create_features_handler (mp_obj, top_obj, 0)  
  #Antecedent is pronoun or something invalid about this MP
  if (valid == False):
    return 0
  data = np.asarray(data)
  num_col = data.shape[0]
  X = data[1:num_col]

  fpath = "lr_trained.sav"
  lr_model = pickle.load (open(fpath, 'rb'))

  Y_prob = lr_model.predict_proba (X.reshape(1, -1))

  return Y_prob[0][1]
