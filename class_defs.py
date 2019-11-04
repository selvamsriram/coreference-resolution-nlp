import utils
import utils_temp

class top:
  def __init__ (self):
    self.docs = {}
    self.pos_list = []
    self.neg_list = []
    self.selected_neg_list = []
    self.feature_list = []
    self.matched_ana = 0
    self.gold_ana = 0
    #Temporary measuring of how much markables are wasted
    self.matched_ante_ana = 0
    self.total_markable = 0

        
class document:
  def __init__ (self, top_obj, input_doc_name, key_doc_name):
    self.sentences = {}
    self.clusters_info = {}
    self.result_clusters_info = {}
    self.top_obj = top_obj
    utils.extract_document (self, input_doc_name, key_doc_name)
    if (key_doc_name != None):
      utils_temp.create_data_using_doc (self, True, True)

class sentence:
  def __init__ (self,doc_sentence):
    self.word_list = []
    self.markables = []
    self.gold_markables = []
    utils.extract_sentence_info (self, doc_sentence)

class word:
  def __init__ (self, word, pos_tag, NER_tag, NER_label, chunk_tag):
    self.word = word
    self.pos_tag = pos_tag
    self.NER_tag = NER_tag
    self.NER_label = NER_label
    self.chunk_tag = chunk_tag


MARKABLE_FLAG_ANTECEDENT = 1
MARKABLE_FLAG_ANAPHOR = 2
MARKABLE_FLAG_NEITHER = 3

class markable:
  def __init__ (self, start_idx, end_idx, min_start_idx, min_end_idx, coref_id, flags):
    self.w_s_idx = start_idx
    self.w_e_idx = end_idx
    self.w_min_s_idx = min_start_idx
    self.w_min_e_idx = min_end_idx
    self.coref_id = coref_id
    self.flags = flags

class cluster_info_piece:
  def __init__ (self, sentence_idx, markable_idx):
    self.sent_idx = sentence_idx
    self.mark_idx = markable_idx

class mention_pair:
  def __init__ (self, doc_obj, a_sent_idx, a_mark_idx, b_sent_idx, b_mark_idx):
    self.dobj = doc_obj
    self.a_sent_idx = a_sent_idx
    self.a_mark_idx = a_mark_idx
    self.b_sent_idx = b_sent_idx
    self.b_mark_idx = b_mark_idx
    self.coref_id = None