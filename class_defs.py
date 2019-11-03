import utils

class top:
  def __init__ (self):
    self.docs = {}
    self.pos_list = []
    self.neg_list = []
    self.feature_list = []
        
class document:
  def __init__ (self, input_doc_name, key_doc_name):
    self.sentences = {}
    self.clusters_info = {}
    utils.extract_document (self, input_doc_name, key_doc_name)

class sentence:
  def __init__ (self,doc_sentence):
    self.word_list = []
    self.markables = []
    self.gold_markables = []
    utils.extract_sentence_info (self, doc_sentence)

class word:
  def __init__ (self, word, pos_tag, NER_tag, chunk_tag):
    self.word = word
    self.pos_tag = pos_tag
    self.NER_tag = NER_tag
    self.chunk_tag = chunk_tag


MARKABLE_FLAG_ANTECEDENT = 1
MARKABLE_FLAG_ANAPHOR = 2
MARKABLE_FLAG_NEITHER = 3

class markable:
  def __init__ (self, start_idx, end_idx, coref_id, flags):
    self.w_s_idx = start_idx
    self.w_e_idx = end_idx
    self.coref_id = coref_id
    self.flags = flags

class cluster_info_piece:
  def __init__ (self, sentence_idx, markable_idx):
    self.sent_idx = sentence_idx
    self.mark_idx = markable_idx

class mention_pair:
  def __init__ (self, doc_name, a_sent_idx, a_start_idx, a_end_idx, b_sent_idx, b_start_idx, b_end_idx):
    self.dname =doc_name
    self.a_sent_idx = a_sent_idx
    self.a_start_idx = a_start_idx
    self.a_end_idx = a_end_idx
    self.b_sent_idx = b_sent_idx
    self.b_start_idx = b_start_idx
    self.b_end_idx = b_end_idx
        
