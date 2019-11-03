import class_defs
import re
import nltk
import utils

def extract_markables_from_input_file (doc_obj, line_num, sent_tag_unrem, sent_tag_rem):
  coref_id_string = ""
  antecedent = None
  sent_tag_unrem = nltk.word_tokenize (sent_tag_unrem)
  sent_tag_rem = nltk.word_tokenize (sent_tag_rem)
  index = 0
  extraction = False
  begin_index = -1
  end_index = -1
  max_len = len (sent_tag_unrem)
  number_of_completed_corefs = 0
  for index in range (0, max_len):
    tok = sent_tag_unrem[index]
    if (tok == "ID="):
      #Check if this is due to the <COREF ID=X#>
      if (sent_tag_unrem[index-2] == "<") and (sent_tag_unrem[index-1] == "COREF") and (sent_tag_unrem[index+1] == "''"):
        coref_id_string = sent_tag_unrem[index+2]
        index = index + 5
        begin_index = index
    elif (tok == "<") and (sent_tag_unrem[index+1] == "/COREF"):
      antecedent = sent_tag_unrem[begin_index:index]
      #Note
      #Compute the index in the tag removed sent tok position
      # 7 for <S ID= "X">
      # 7 for <COREF ID="X#">
      # 3 for </COREF>
      begin_index = begin_index -(number_of_completed_corefs * 10) - 7 - 7
      end_index = index - (number_of_completed_corefs * 10) -7 -7

      #Debug Prints
      #print ("Coreference ID ", coref_id_string, "Unremoved Antecedent ", antecedent)
      #print ("Coreference ID ", coref_id_string, "Removed Antecedent ", sent_tag_rem[begin_index:end_index])

      #Create a markable_obj
      markable_obj = class_defs.markable (begin_index, end_index, coref_id_string, class_defs.MARKABLE_FLAG_ANTECEDENT)
      sent_obj = doc_obj.sentences[line_num]
      sent_obj.markables.append (markable_obj)
      begin_index = -1
      number_of_completed_corefs += 1

def create_gold_markable_list (doc_obj, input_file, key_file):
  ifp = open (input_file)
  kfp = open (key_file)

  line_num = 0
  for line in ifp:
    line = line.strip ('\n')
    sent_tag_unrem = line
    sent_tag_rem = utils.preprocess_sentence  (line)
    extract_markables_from_input_file (doc_obj, line_num, sent_tag_unrem, sent_tag_rem)
    line_num += 1
  ifp.close ()
  kfp.close ()
