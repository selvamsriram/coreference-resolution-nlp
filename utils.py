import class_defs
import random
import numpy as np 
import re
import nltk
import utils_temp
from nltk.stem import WordNetLemmatizer
import spacy
from prettytable import PrettyTable

#Globals Start
pronoun_lst = ["a", "an", "the", "this", "these", "that", "those"]
dem_pronoun_lst = ["this", "these", "that", "those"]
male_identifiers = ["mr.", "mr", "he", "him", "himself", "his", "boy", "sir", "boys", "men", "man"]
female_identifiers = ["mrs.", "miss", "ms.", "ms", "she", "her", "herself", "her's", "madam", "lady", "girl", "girls", "women", "woman"]
plural_pronoun_lst = ["these", "those", "both", "few", "fewer", "many", "others", "several", "our",
                      "their", "theirs", "we", "they", "us", "them", "ourselves", "themselves", "some"]
singular_pronoun_lst = ["me", "it", "my", "mine", "its", "myself", "itself", "this", "that", "he", "him", "himself",
                         "his", "she", "her", "herself", "her's", "a", "an"]
#Globals end

def extract_document (doc_obj, input_file, key_file):
  ifp = open (input_file)

  line_num = 0
  for line in ifp:
      doc_obj.sentences[line_num] = class_defs.sentence (line, doc_obj)
      line_num += 1

  ifp.close ()

  utils_temp.create_gold_markable_list (doc_obj, input_file, key_file)
  max_line = line_num
  for i in range (0, max_line):
    sent_obj = doc_obj.sentences[i]
    #Compare and also propogate values from gold to markables list
    compare_gold_and_extracted_markables (doc_obj, sent_obj, i)
    #print ("After compare gold") 
    #utils_temp.debug_printer (doc_obj)
    #Put the missing antecedents in place
    utils_temp.take_care_of_missed_antecedents (doc_obj, sent_obj, i)
    #print ("After take care") 
    #utils_temp.debug_printer (doc_obj)
    find_missed_anaphors (doc_obj, sent_obj, i)
    #print ("After find_missed_anaphors")
    #utils_temp.debug_printer (doc_obj)
    count_total_antecedents_our_markable (doc_obj, sent_obj)

  if (key_file != None):
    top = doc_obj.top_obj
    print ("Results of markable extraction")
    print ("Matched % = ", top.matched_ana/(top.gold_ana))
    print ("Total Markables  = {}".format (top.total_markable))
    print ("Gold Anaphors    = {}".format (top.gold_ana))
    print ("Matched Anaphors = {}".format (top.matched_ana))
    print ("Gold Antecedent  = {}".format (top.gold_ante))
    print ("Loaded Antecdent = {}".format (top.loaded_ante))
    print ("Wasted Markable % = ", (top.total_markable - top.matched_ante_ana)/top.total_markable)
    print ("Missed Anaphor : ", top.missed_anaphors)

def compare_total_antecedents (doc_obj):
  max_sent_len = len (doc_obj.sentences)

  for line_num in range (0, max_sent_len):
    sent_obj = doc_obj.sentences[line_num]
    if (len(sent_obj.markables) == 0) and (len(sent_obj.gold_markables) == 0):
      continue
    mids = []
    coref_ids = []
    indices = []
    flags = []
    print (line_num, ":", sent_obj.full_sentence)
    for i,marker in enumerate (sent_obj.markables):
      mids.append (i)
      coref_ids.append (marker.coref_id)
      indices.append ((marker.w_s_idx, marker.w_e_idx))
      flags.append (marker.flags)
    t = PrettyTable (mids)
    t.add_row (coref_ids)
    t.add_row (indices)
    t.add_row (flags)
    print ("Our Markable Tables")
    print (t)
    mids = []
    coref_ids = []
    indices = []
    flags = []
    for i,marker in enumerate (sent_obj.gold_markables):
      mids.append (i)
      coref_ids.append (marker.coref_id)
      indices.append ((marker.w_s_idx, marker.w_e_idx))
      flags.append (marker.flags)
    t = PrettyTable (mids)
    t.add_row (coref_ids)
    t.add_row (indices)
    t.add_row (flags)
    print ("Gold Markable Tables")
    print (t)

def count_total_antecedents_our_markable (doc_obj, sent_obj):
  markables = sent_obj.markables 
  for marker in markables:
    if (marker.flags == class_defs.MARKABLE_FLAG_ANTECEDENT):
      doc_obj.top_obj.loaded_ante += 1

def find_missed_anaphors (doc_obj, sent_obj, line_num):
  gold_markables = sent_obj.gold_markables
  first = True

  for i, gmarker in enumerate (gold_markables):
    if (gmarker.flags == class_defs.MARKABLE_FLAG_ANAPHOR) and (gmarker.anaphor_detected == False):
      doc_obj.top_obj.missed_anaphors += 1
      if (first == True):
        print ("Line Num", line_num)
        print ("Sentence   : ", sent_obj.full_sentence)
        first = False
      s_idx = gmarker.w_s_idx
      e_idx = gmarker.w_e_idx
      tok_list = ["TOKEN"]
      pos_list = ["POS TAG"]
      NER_tag_list = ["NER TAG"]
      NER_label_list = ["NER LABEL"]
      NP_chunk_tag_list = ["NP CHUNK TAG"]
      tok_header_list = ["Info Name"]
      an_string = ""
      for i in range (s_idx, e_idx+1):
        word = sent_obj.word_list[i]
        tok_header_list.append (i)
        tok_list.append (word.word)
        pos_list.append (word.pos_tag)
        NER_tag_list.append (word.NER_tag)
        NER_label_list.append (word.NER_label) 
        NP_chunk_tag_list.append (word.chunk_tag)
        an_string += word.word
        if (i != e_idx):
          an_string += " "

      print ("Missed Anaphor:")
      print ("---------------")
      print (an_string)

      t = PrettyTable(tok_header_list)
      t.add_row (tok_list)
      t.add_row (pos_list)
      t.add_row (NER_tag_list)
      t.add_row (NER_label_list)
      t.add_row (NP_chunk_tag_list)
      print (t)


def preprocess_sentence (doc_sentence):
  doc_sentence = doc_sentence.strip ('\n')
  pattern = re.compile (r'<.*?>')
  return pattern.sub ('', doc_sentence)


def compare_gold_and_extracted_markables (doc_obj, sent_obj, sent_num):
  top_obj = doc_obj.top_obj
  gold_table = sent_obj.gold_markables
  extracted_table = sent_obj.markables
  max_gold_sent = []
  min_gold_sent = []

  gold_len = len(gold_table)

  for i in range (gold_len):
    if (gold_table[i].flags != class_defs.MARKABLE_FLAG_ANTECEDENT):
      top_obj.gold_ana += 1
    elif (gold_table[i].flags == class_defs.MARKABLE_FLAG_ANTECEDENT):
      top_obj.gold_ante += 1

    #max string
    max_s_idx = gold_table[i].w_s_idx
    max_e_idx = gold_table[i].w_e_idx
    temp_max = ""

    #print ("max_s_idx ", max_s_idx, "max_e_idx", max_e_idx)
    for j in range (max_s_idx, max_e_idx + 1):
      temp_max += sent_obj.word_list[j].word
      if (j != max_e_idx):
        temp_max += " "
    max_gold_sent.append (temp_max)
    
    #min string
    if (gold_table[i].flags != class_defs.MARKABLE_FLAG_ANTECEDENT):
      min_s_idx = gold_table[i].w_min_s_idx
      min_e_idx = gold_table[i].w_min_e_idx
      temp_min = "" 

      for j in range (min_s_idx, min_e_idx + 1):
        temp_min += sent_obj.word_list[j].word
        if (j != min_e_idx):
          temp_min += " "

      min_gold_sent.append (temp_min)
    else:
      min_gold_sent.append (" ")


  extracted_tlen = len (extracted_table)
  top_obj.total_markable += extracted_tlen
  for i in range (extracted_tlen):
    s_idx = extracted_table[i].w_s_idx
    e_idx = extracted_table[i].w_e_idx
    temp_e = ""
    
    for j in range (s_idx, e_idx + 1):
      temp_e += sent_obj.word_list[j].word
      if (j != e_idx):
        temp_e += " "

    for k, phrase in enumerate (max_gold_sent):
      if temp_e in phrase:
        if (gold_table[k].flags != class_defs.MARKABLE_FLAG_ANAPHOR):
          if (temp_e == phrase):
            extracted_table[i].coref_id = gold_table[k].coref_id
            extracted_table[i].flags = gold_table[k].flags
            break

        if (gold_table[k].flags == class_defs.MARKABLE_FLAG_ANAPHOR):
          if (min_gold_sent[k] in temp_e) and (extracted_table[i].flags != class_defs.MARKABLE_FLAG_ANTECEDENT):
            top_obj.matched_ana += 1 
            extracted_table[i].flags = gold_table[k].flags 
            extracted_table[i].coref_id = gold_table[k].coref_id
            gold_table[k].anaphor_detected = True
            break

    #if (extracted_table[i].flags == class_defs.MARKABLE_FLAG_ANAPHOR):
      #print ("Coref ID : {} S ID : {} String : {}".format(extracted_table[i].coref_id,sent_num ,temp_e.lstrip ()))

    if ((extracted_table[i].flags == class_defs.MARKABLE_FLAG_ANAPHOR) or 
        (extracted_table[i].flags == class_defs.MARKABLE_FLAG_ANTECEDENT)):
       top_obj.matched_ante_ana += 1

def spacy_extract_sentence_info (sent_obj, doc_sentence, doc_obj):
  doc_sentence = preprocess_sentence (doc_sentence)
  tok_list = utils_temp.spacy_get_tokenized_word (doc_obj, doc_sentence)
  pos_list = utils_temp.spacy_get_pos_tags (doc_obj, doc_sentence)
  ner_tag_list, ner_label_list = utils_temp.spacy_get_ner_bio_tag (doc_obj, doc_sentence)
  len_sen = len (tok_list)

  for i in range(len_sen):
    curr_word = tok_list[i]
    pos_tag = pos_list[i]
    NER_tag = ner_tag_list[i]
    NER_label = ner_label_list[i]
    sent_obj.word_list.append (class_defs.word (curr_word, pos_tag, NER_tag, NER_label, 'O'))

  utils_temp.spacy_get_np_tags_filled (doc_obj, sent_obj, doc_sentence)
  
  markable_list = spacy_compute_markable_table (sent_obj)
  sent_obj.markables = markable_list
  sent_obj.full_sentence = doc_sentence

def spacy_compute_markable_table (sent_obj):
  len_lst = len (sent_obj.word_list)
  markable_lst = []
  m_start_idx = -1
  m_end_idx = -1

  for i in range (len_lst):
    curr_word = sent_obj.word_list[i]
    pos_tag = curr_word.pos_tag
    NER_tag = curr_word.NER_tag
    np_tag = curr_word.chunk_tag

    if (((np_tag == "B-NP") or (NER_tag == "B")) and (m_start_idx == -1)):
      m_start_idx = i 
      m_end_idx = i 
    elif ((np_tag == "I-NP") or (NER_tag == "I")):
      m_end_idx = i
    else:
      if (m_start_idx != -1):
        markable_obj = class_defs.markable (m_start_idx, m_end_idx, -1, -1, 0, 0)
        markable_lst.append (markable_obj)
        m_start_idx = -1
        m_end_idx = -1
      
      if (pos_tag == "PRP" or pos_tag == "PRP$" or pos_tag == "WP" or pos_tag == "WP$" or 
         pos_tag == "NN" or pos_tag == "NNS" or pos_tag == "NNP" or pos_tag == "NNPS"):
        markable_obj = class_defs.markable (i, i, -1, -1, 0, 0)
        markable_lst.append (markable_obj)

  if (m_start_idx != -1):
    markable_obj = class_defs.markable (m_start_idx, m_end_idx, -1, -1, 0, 0)
    markable_lst.append (markable_obj)

  for marker in markable_lst:
    print ("\n", end="")
    for i in range (marker.w_s_idx, marker.w_e_idx+1):
      print (sent_obj.word_list[i].word, " ", end="")


  return markable_lst

def compute_markable_table (sent_obj):
  len_lst = len (sent_obj.word_list)
  markable_lst = []

  for i in range (len_lst):
    curr_word = sent_obj.word_list[i]
    pos_tag = curr_word.pos_tag
    NER_tag = curr_word.NER_tag
    np_tag = curr_word.chunk_tag
   
    #Pronoun not part of Noun Phrase
    if (pos_tag == "PRP" or pos_tag == "PRP$" or pos_tag == "WP" or pos_tag == "WP$"):
      if (np_tag == "O"):
        markable_obj = class_defs.markable (i, i, -1, -1, 0, 0)
        markable_lst.append (markable_obj)
        continue

    if (pos_tag == "NN" or pos_tag == "NNS" or pos_tag == "NNP" or pos_tag == "NNPS"):
      if (np_tag == "O"):
        markable_obj = class_defs.markable (i, i, -1, -1, 0, 0)
        markable_lst.append (markable_obj)
        continue

    if (np_tag != "O"):
      if (np_tag == "B-NP"):
        markable_obj = class_defs.markable (i, i, -1, -1, 0, 0)

      elif (np_tag == "I-NP"):
        markable_obj.w_e_idx = i

      if (i < len_lst - 1):
        if (sent_obj.word_list[i+1].chunk_tag == "O"):
          markable_lst.append (markable_obj)

      elif (i == len_lst - 1):
        markable_lst.append (markable_obj)


  return markable_lst



def extract_sentence_info (sent_obj, doc_sentence):
  doc_sentence = preprocess_sentence (doc_sentence)
  tokens = nltk.word_tokenize (doc_sentence)

  pos_tags = nltk.pos_tag (tokens)

  ner_tree = nltk.ne_chunk (pos_tags)
  ner_bio = nltk.chunk.tree2conlltags (ner_tree)

  #print (ner_tree)

  #grammar = "NP: {<DT|PRP\$> <VBG> <NN.*>+} {<DT|PRP\$> <NN.*> <POS> <JJ>* <NN.*>+}{<DT|PRP\$>? <JJ>* <NN.*>+ }"
  #grammar = "NP: (?:(?:\w+ DT )?(?:\w+ JJ )*)?\w+ (?:N[NP]|PRN)"
  #grammar = "NP: {(<V\w+>|<NN\w?>)+.*<NN\w?>}"

  grammar = '''NP: {<DT><NN><IN><DT><JJ>}
                   {<DT><NNP><NNPS>}
                   {<DT>?<JJ>*<NN>}
                   {<DT><PRP\$><NN>}
                   {<DT><JJS><NN>}
                   {<DT><NN><IN><NNP><NNP>}
                   {<NN><NN>}
                   '''
  cp = nltk.RegexpParser(grammar)

  np_res = cp.parse (pos_tags)

  #print (np_res)
  iob_chunk = nltk.chunk.tree2conlltags (np_res)
  #print (iob_chunk)
  len_sen = len (iob_chunk)

  for i in range(len_sen):
    curr_word = iob_chunk[i][0]
    pos_tag = iob_chunk[i][1]
    chunk_tag = iob_chunk[i][2]
    NER_tag = ner_bio[i][2]

    sent_obj.word_list.append (class_defs.word (curr_word, pos_tag, NER_tag, None, chunk_tag))

  markable_lst = compute_markable_table (sent_obj)
  sent_obj.markables = markable_lst


def print_mention_pair_all_details (top_obj, mp, row):
  doc_obj = mp.dobj
  a_sent_obj = doc_obj.sentences[mp.a_sent_idx]
  b_sent_obj = doc_obj.sentences[mp.b_sent_idx]
  a_marker = a_sent_obj.markables[mp.a_mark_idx]
  b_marker = b_sent_obj.markables[mp.b_mark_idx]
  a_s_idx = a_marker.w_s_idx
  a_e_idx = a_marker.w_e_idx
  b_s_idx = b_marker.w_s_idx
  b_e_idx = b_marker.w_e_idx
  tok_list = ["TOKEN"]
  pos_list = ["POS TAG"]
  NER_tag_list = ["NER TAG"]
  NER_label_list = ["NER LABEL"]
  NP_chunk_tag_list = ["NP CHUNK TAG"]
  tok_header_list = ["Info Name"]
  ante_string = ""
  for i in range (a_s_idx, a_e_idx+1):
    word = a_sent_obj.word_list[i]
    tok_header_list.append (i)
    tok_list.append (word.word)
    pos_list.append (word.pos_tag)
    NER_tag_list.append (word.NER_tag)
    NER_label_list.append (word.NER_label) 
    NP_chunk_tag_list.append (word.chunk_tag)
    ante_string += word.word
    if (i != a_e_idx):
      ante_string += " "

  ante_t = PrettyTable(tok_header_list)
  ante_t.add_row (tok_list)
  ante_t.add_row (pos_list)
  ante_t.add_row (NER_tag_list)
  ante_t.add_row (NER_label_list)
  ante_t.add_row (NP_chunk_tag_list)

  tok_list = ["TOKEN"]
  pos_list = ["POS TAG"]
  NER_tag_list = ["NER TAG"]
  NER_label_list = ["NER LABEL"]
  NP_chunk_tag_list = ["NP CHUNK TAG"]
  tok_header_list = ["Info Name"]
  ana_string = ""
  for i in range (b_s_idx, b_e_idx+1):
    word = b_sent_obj.word_list[i]
    tok_header_list.append (i)
    tok_list.append (word.word)
    pos_list.append (word.pos_tag)
    NER_tag_list.append (word.NER_tag)
    NER_label_list.append (word.NER_label) 
    NP_chunk_tag_list.append (word.chunk_tag)
    ana_string += word.word
    if (i != b_e_idx):
      ana_string += " "
  
  ana_t = PrettyTable(tok_header_list)
  ana_t.add_row (tok_list)
  ana_t.add_row (pos_list)
  ana_t.add_row (NER_tag_list)
  ana_t.add_row (NER_label_list)
  ana_t.add_row (NP_chunk_tag_list)

  print ("Mention Pair Details")
  print ("Antecedent      : {", mp.a_sent_idx, "}", ante_string)
  print ("Anaphor         : {", mp.b_sent_idx, "}", ana_string)
  print ("Ante Sentence   : ", a_sent_obj.full_sentence)
  print ("Ana  Sentence   : ", b_sent_obj.full_sentence)
  print ("Antecedent Table")
  print ("-----------------")
  print (ante_t)
  print ("\nAnaphor Table")
  print ("----------------")
  print (ana_t)

  if (row != None):
    print_feature_row (top_obj, mp, row)

def print_feature_row (top_obj, mp, row):
  feature_list = ["Label", "Sent Distance", "Ante-Pronoun", "Ana-Pronoun", "Str Match", "Ana Def NP", "Ana Dem NP", "Number Agm", "Sem Class Agm", "Gender Agm", "Both NNP(s)", "Alias", "Appositive"]
  header = ["Feature", "Value"]
  t = PrettyTable(header)
  for idx, feat in enumerate(feature_list):
    pretty_row = [feat, row[idx]]
    t.add_row (pretty_row)
  print (t)

def get_imp_data_for_mention (top_obj, mp, is_ante):
  dobj = mp.dobj

  if (is_ante == True):
    sent_id             = mp.a_sent_idx
    sent_obj            = dobj.sentences[sent_id]
    markables_list      = sent_obj.markables
    w_start_idx         = markables_list[mp.a_mark_idx].w_s_idx
    w_end_idx           = markables_list[mp.a_mark_idx].w_e_idx
  else:
    sent_id             = mp.b_sent_idx
    sent_obj            = dobj.sentences[sent_id]
    markables_list      = sent_obj.markables
    w_start_idx         = markables_list[mp.b_mark_idx].w_s_idx
    w_end_idx           = markables_list[mp.b_mark_idx].w_e_idx

  w_list                          = sent_obj.word_list
  string_with_spaces_wo_pronoun   = ""
  string_with_spaces              = ""
  words_pos_tag_list              = []
  #At the same time fill the POS list, word with spaces and word with spaces and no pronoun
  for s in range (w_start_idx, w_end_idx+1):
    words_pos_tag_list.append (w_list[s].pos_tag)
    string_with_spaces += w_list[s].word.lower()
    if (s != w_end_idx):
      string_with_spaces += " "

    if (w_list[s].word.lower() in pronoun_lst):
      continue

    string_with_spaces_wo_pronoun += w_list[s].word.lower()
    if (s != w_end_idx):
      string_with_spaces_wo_pronoun += " "

  return sent_id, sent_obj, markables_list, w_start_idx, w_end_idx, w_list, string_with_spaces_wo_pronoun, string_with_spaces, words_pos_tag_list

def feature_is_pronoun (brick):
  is_pronoun = False
  if (brick.w_start_idx == brick.w_end_idx):
    pos_tag = brick.w_list[brick.w_start_idx].pos_tag
    if (pos_tag == "PRP" or pos_tag == "PRP$"):
      is_pronoun = True

  if (is_pronoun == True):
    is_pronoun = 1
  else:
    is_pronoun = 0
  return is_pronoun

def feature_is_string_match (antecedent_brick, anaphor_brick):
  is_match = False
  if (antecedent_brick.string_with_spaces_wo_pronoun == anaphor_brick.string_with_spaces_wo_pronoun):
    is_match = True

  if (is_match == True):
    is_match = 1
  else:
    is_match = 0

  return is_match

def feature_is_def_np (brick):
  is_def_np = False
  if (brick.w_list[brick.w_start_idx].word.lower() == "the"):
    is_def_np = True

  if (is_def_np == True):
    is_def_np = 1
  else:
    is_def_np = 0
  return is_def_np

def feature_is_demons_np (brick):
  is_dem_np = False
  if (brick.w_list[brick.w_start_idx].word.lower() in dem_pronoun_lst):
    is_dem_np = True

  if (is_dem_np == True):
    is_dem_np = 1
  else:
    is_dem_np = 0
  return is_dem_np

def feature_is_number_agreement (top_obj, a_brick, b_brick):
  wl = WordNetLemmatizer ()
  #-1 = unknown
  #0 = singular
  #1 = plural
  a_person = -1
  for a_idx in range (a_brick.w_start_idx, a_brick.w_end_idx + 1):
    word_a = a_brick.w_list[a_idx].word
    if (word_a.lower() in plural_pronoun_lst):
      a_person = 1
      break
    elif (word_a.lower() in singular_pronoun_lst):
      a_person = 0

    lemma = wl.lemmatize(word_a, 'n')
    if (word_a not in lemma):
      a_person = 1
      break
    else:
      if (a_brick.w_list[a_idx].pos_tag == "NN" or a_brick.w_list[a_idx].pos_tag == "NNP"):
        if (word_a in lemma):
          a_person = 0

  b_person = -1
  for b_idx in range (b_brick.w_start_idx, b_brick.w_end_idx + 1):
    word_b = b_brick.w_list[b_idx].word
    if (word_b.lower() in plural_pronoun_lst):
      b_person = 1
      break
    elif (word_b.lower() in singular_pronoun_lst):
      b_person = 0

    lemma = wl.lemmatize(word_b, 'n')
    if (word_b not in lemma):
      b_person = 1
      break
    else:
      if (b_brick.w_list[b_idx].pos_tag == "NN" or b_brick.w_list[b_idx].pos_tag == "NNP"):
        if (word_b in lemma):
          b_person = 0
  if (a_person == b_person):
    return 1
  else:
    return 0

def feature_is_sem_class_agreement (top_obj, a_brick, b_brick):
  ner_spacy = top_obj.spacy_obj 

  ner_spacy_ant = ner_spacy (a_brick.string_with_spaces)
  ner_spacy_ana = ner_spacy (b_brick.string_with_spaces)

  #print ("Antecedent: ", antecedent)
  #for ents in ner_spacy_ant.ents:
    #print (ents.text, ents.label_ )

  #print ("Anaphor: ", anaphor)
  flag = False

  for ents in ner_spacy_ana.ents:
    for ant_ents in ner_spacy_ant.ents:
      if (ents.label_ is ant_ents.label_):
        flag = True
        break
    if (flag == True):
      break

  if (flag == True):
    flag = 1
  else:
    flag = 0
  return flag

def feature_gender_agreement (a_brick, b_brick):
  antecedent = a_brick.string_with_spaces
  anaphor    = b_brick.string_with_spaces
  gender_agreement = 3
  if (((antecedent.lower () in male_identifiers) and (anaphor.lower () in male_identifiers)) or
     ((antecedent.lower () in female_identifiers) and (anaphor.lower () in female_identifiers))):
     gender_agreement = 1

  if (((antecedent.lower () in male_identifiers) and (anaphor.lower () in female_identifiers)) or
     ((antecedent.lower () in female_identifiers) and (anaphor.lower () in male_identifiers))):
     gender_agreement = 2

  return gender_agreement

def feature_is_both_proper_name (a_brick, b_brick):
  a_pos_tag_list = a_brick.words_pos_tag_list
  b_pos_tag_list = b_brick.words_pos_tag_list
  both_proper_names = 0

  if (("NNP" in a_pos_tag_list) or ("NNPS" in a_pos_tag_list)):
    if (("NNP" in b_pos_tag_list) or ("NNPS" in b_pos_tag_list)):
      both_proper_names = 1
  
  return both_proper_names

def feature_is_alias(a_brick, b_brick):
  a_tok_list = []
  a_ner_tag_list = []
  a_atleast_one_valid = False
  a_ner_label_type = ""

  b_tok_list = []
  b_ner_tag_list = []
  b_atleast_one_valid = False 
  b_ner_label_type = ""
  
  is_alias = False

  for ner_iter_index  in range (a_brick.w_start_idx, a_brick.w_end_idx+1):
    word = a_brick.w_list[ner_iter_index]
    if (word.NER_tag != 'O'):
      a_atleast_one_valid = True
      a_ner_label_type = word.NER_label
    a_ner_tag_list.append (word.NER_tag)
    a_tok_list.append (word.word)

  for ner_iter_index  in range (b_brick.w_start_idx, b_brick.w_end_idx+1):
    word = b_brick.w_list[ner_iter_index]
    if (word.NER_tag != 'O'):
      b_atleast_one_valid = True
      b_ner_label_type = word.NER_label
    b_ner_tag_list.append (word.NER_tag)
    b_tok_list.append (word.word)

  if ((a_atleast_one_valid != b_atleast_one_valid) or 
      ((a_atleast_one_valid == False) and (b_atleast_one_valid == False)) or
      (a_ner_label_type != b_ner_label_type)):
    is_alias = False
  else:
    a_set = set(a_tok_list)
    b_set = set (b_tok_list)

    if (a_set & b_set):
      #print ("NER Type :", antecedent, a_ner_label_type, anaphor, b_ner_label_type, "Match", )
      is_alias = True
    else:
      is_alias = False

  if (is_alias == True):
    is_alias = 1
  else:
    is_alias = 0
  return is_alias

def feature_is_appositive (a_brick, b_brick):
  sent_diff = b_brick.sent_id - a_brick.sent_id
  is_appositive = False

  if (sent_diff > 0):
    is_appositive = False
  else:
    #They are in the same line, now check how close they are to each other
    index_diff = b_brick.w_start_idx - a_brick.w_end_idx
    if (index_diff > 3):
      is_appositive = False
    else:
      #We have established they are close, now check what is in between if there is any.
      verb_found = False
      if (index_diff > 0):
        for temp_index  in range (a_brick.w_end_idx+1, b_brick.w_start_idx):
          if (a_brick.w_list[temp_index].pos_tag in ["MD", "VBD", "VB", "VBG", "VBN", "VBZ", "NNP", "NN"]):
            verb_found = True
            break
        if (verb_found == True):
          is_appositive = False
      
      if (verb_found == False):
        #Check if one of them are proper noun
        #Dependent on Feature 10
        pos_tag_set = set (a_brick.words_pos_tag_list + b_brick.words_pos_tag_list)
        if ("NNP" in pos_tag_set):
          if (b_brick.w_list[b_brick.w_start_idx].word.lower() == "the"):
            #print ("Appositive Feature : Index Diff : ", index_diff, "Ante :", antecedent, "Anaphor :", anaphor)
            is_appositive = True
          else:
            is_appositive = False
        else:
          is_appositive = False
  
  if (is_appositive == True):
    is_appositive = 1
  else:
    is_appositive = 0
  return is_appositive

def create_feature_per_row (mp, antecedent_brick, anaphor_brick, top_obj, label):
  row = []
  #Label
  row.append (label)

  #Feature 1 (Distance)
  row.append (anaphor_brick.sent_id - antecedent_brick.sent_id)

  #Feature 2 (i-Pronoun)
  antecedent_is_pronoun = feature_is_pronoun (antecedent_brick)
  row.append (antecedent_is_pronoun)
  
  #Feature 3 (j-Pronoun)
  anaphor_is_pronoun = feature_is_pronoun (anaphor_brick)
  row.append (anaphor_is_pronoun)

  #Enforcing rule by Ng and Cardie 2002
  if ((antecedent_is_pronoun == True) and (anaphor_is_pronoun == False)):
      print ("Antecedent is pronoun but anaphor is not so skipping this MP creation")
      return None, False

  #Feature 4 (String Match)
  row.append (feature_is_string_match(antecedent_brick, anaphor_brick))

  #Feature 5 (j is Definitive NP)
  row.append (feature_is_def_np(anaphor_brick))

  #Feature 6 (j is Demonstratie NP)
  row.append (feature_is_demons_np(anaphor_brick))

  #Feature 7 (Number Agreement)
  row.append (feature_is_number_agreement (top_obj, antecedent_brick, anaphor_brick))

  #Feature 8 (Semantic Class Agreement)
  row.append (feature_is_sem_class_agreement(top_obj, antecedent_brick, anaphor_brick))

  #Feature 9 (Gender Agreement)
  row.append (feature_gender_agreement (antecedent_brick, anaphor_brick))

  #Feature 10 (Both proper name)
  row.append (feature_is_both_proper_name(antecedent_brick, anaphor_brick))

  #Feature 11 (Alias Feature)
  row.append (feature_is_alias(antecedent_brick, anaphor_brick))

  #Feature 12  (Appositive Feature)
  row.append (feature_is_appositive (antecedent_brick, anaphor_brick))

  return row, True


def single_mp_create_features_handler (mp, top_obj, label):
  sent_id, sent_obj, markables_list, w_start_idx, w_end_idx, w_list, string_with_spaces_wo_pronoun, string_with_spaces, words_pos_tag_list = get_imp_data_for_mention (top_obj, mp, True)
  antecedent_brick = class_defs.mp_one_comp_feature_brick (sent_id, sent_obj, markables_list, w_start_idx, w_end_idx, w_list, string_with_spaces_wo_pronoun, string_with_spaces, words_pos_tag_list)

  sent_id, sent_obj, markables_list, w_start_idx, w_end_idx, w_list, string_with_spaces_wo_pronoun, string_with_spaces, words_pos_tag_list = get_imp_data_for_mention (top_obj, mp, False)
  anaphor_brick = class_defs.mp_one_comp_feature_brick (sent_id, sent_obj, markables_list, w_start_idx, w_end_idx, w_list, string_with_spaces_wo_pronoun, string_with_spaces, words_pos_tag_list)

  feature_row, valid = create_feature_per_row (mp, antecedent_brick, anaphor_brick, top_obj, label)
  return feature_row, valid

def bulk_mp_modular_create_features_handler (mp_list, top_obj, label, row_list):
  mp_max = len(mp_list)

  for mp_idx in range (0, mp_max):
    mp = mp_list[mp_idx]
    feature_row, valid = single_mp_create_features_handler (mp, top_obj, label)
    if (valid == True):
      #Debug Print
      print_mention_pair_all_details (top_obj, mp, feature_row)
      row_list.append (feature_row)

  return row_list

def dump_features_to_file (file_handle, row_list):
  for row in row_list:
    for idx, col_val in enumerate (row):
      file_handle.write ("%s" %col_val)
      if (idx != len(row) -1):
        file_handle.write (", ")
    file_handle.write ("\n")

def create_features_handler (filename, lst, top_obj, label):
  llen = len (lst)
  pronoun_lst = ["a", "an", "the", "this", "these", "that", "those"]
  dem_pronoun_lst = ["this", "these", "that", "those"]
  male_identifiers = ["mr.", "mr", "he", "him", "himself", "his", "boy", "sir", "boys", "men", "man"]
  female_identifiers = ["mrs.", "miss", "ms.", "ms", "she", "her", "herself", "her's", "madam", "lady", "girl", "girls", "women", "woman"]
  plural_pronoun_lst = ["these", "those", "both", "few", "fewer", "many", "others", "several", "our",
                        "their", "theirs", "we", "they", "us", "them", "ourselves", "themselves"]
  singular_pronoun_lst = ["me", "it", "my", "mine", "its", "myself", "itself", "this", "that", "he", "him", "himself",
                          "his", "she", "her", "herself", "her's"]
  for i in range (llen):
    row = []
    doc_obj = lst[i].dobj
    a_sentid = lst[i].a_sent_idx
    antecedent_sent = lst[i].dobj.sentences[a_sentid]
    a_s_idx = antecedent_sent.markables[lst[i].a_mark_idx].w_s_idx
    a_e_idx = antecedent_sent.markables[lst[i].a_mark_idx].w_e_idx
    a_wordlst = antecedent_sent.word_list
    
    b_sentid = lst[i].b_sent_idx
    anaphor_sent = lst[i].dobj.sentences[b_sentid]
    b_s_idx = anaphor_sent.markables[lst[i].b_mark_idx].w_s_idx
    b_e_idx = anaphor_sent.markables[lst[i].b_mark_idx].w_e_idx
    b_wordlst = anaphor_sent.word_list

    #Label
    row.append (label)

    #Feature 1 (Distance)
    row.append (b_sentid - a_sentid)

    #Feature 2 (i-Pronoun)
    antecedent_is_pronoun = False
    if (a_s_idx == a_e_idx):
      a_pos_tag = antecedent_sent.word_list[a_s_idx].pos_tag
      if (a_pos_tag == "PRP" or a_pos_tag == "PRP$"):
        row.append (1)
        antecedent_is_pronoun = True
      else:
        row.append (0)
    else:
      row.append(0)

    
    #Feature 3 (j-Pronoun)
    anaphor_is_pronoun = False
    if (b_s_idx == b_e_idx):
      b_pos_tag = anaphor_sent.word_list[b_s_idx].pos_tag
      if (b_pos_tag == "PRP" or b_pos_tag == "PRP$"):
        row.append (1)
        anaphor_is_pronoun = True
      else:
        row.append (0)
    else:
      row.append(0)
     
    if ((antecedent_is_pronoun == True) and (anaphor_is_pronoun == False)):
      print ("Antecedent is pronoun but anaphor is not so skipping this MP creation")
      return None, False

    #Feature 4 (String Match)
    temp_s1 = ""
    for s in range (a_s_idx, a_e_idx+1):
      if (a_wordlst[s].word.lower() in pronoun_lst):
        continue

      temp_s1 += a_wordlst[s].word.lower()
      if (s != a_e_idx):
        temp_s1 += " "


    temp_s2 = ""
    for s in range (b_s_idx, b_e_idx+1):
      if (b_wordlst[s].word.lower() in pronoun_lst):
        continue

      temp_s2 += b_wordlst[s].word.lower()
      if (s != b_e_idx):
        temp_s2 += " "

    if (temp_s1 == temp_s2):
      row.append (1)
    else: 
      row.append (0)

    #Feature 5 (Definitive Noun Phrase)
    #Check if j is definitive NP

    if (b_wordlst[b_s_idx].word.lower() == "the"):
      row.append (1)
    else:
      row.append (0)

    #Feature 6 (Demonstrative Noun Phrase)
    if (b_wordlst[b_s_idx].word.lower() in dem_pronoun_lst):
      row.append (1)
    else:
      row.append (0)

    #Feature 7 (Number Agreement) Pending ...
    wl = WordNetLemmatizer ()
    #-1 = unknown
    #0 = singular
    #1 = plural
    a_person = -1
    for a_idx in range (a_s_idx, a_e_idx + 1):
      word_a = a_wordlst[a_idx].word
      if (word_a.lower() in plural_pronoun_lst):
        a_person = 1
        break
      elif (word_a.lower() in singular_pronoun_lst):
        a_person = 0
        break

      lemma = wl.lemmatize(word_a, 'n')
      if (word_a not in lemma):
        a_person = 1
        break
      else:
        if (a_wordlst[a_idx].pos_tag == "NN" or a_wordlst[a_idx].pos_tag == "NNP"):
          if (word_a in lemma):
            a_person = 0
            break

    b_person = -1
    for b_idx in range (b_s_idx, b_e_idx + 1):
      word_b = b_wordlst[b_idx].word
      if (word_b.lower() in plural_pronoun_lst):
        b_person = 1
        break
      elif (word_b.lower() in singular_pronoun_lst):
        b_person = 0
        break

      lemma = wl.lemmatize(word_b, 'n')
      if (word_b not in lemma):
        b_person = 1
        break
      else:
        if (b_wordlst[b_idx].pos_tag == "NN" or b_wordlst[b_idx].pos_tag == "NNP"):
          if (word_b in lemma):
            b_person = 0
            break
    if (a_person == b_person):
      row.append (1)
    else:
      row.append (0)

    #Feature 8 (Semantic Class Agreement)
    ner_spacy = top_obj.spacy_obj 

    antecedent = ""
    for s in range (a_s_idx, a_e_idx+1):

      antecedent += a_wordlst[s].word
      if (s != a_e_idx):
        antecedent += " "

    anaphor = ""
    for s in range (b_s_idx, b_e_idx+1):

      anaphor += b_wordlst[s].word
      if (s != b_e_idx):
        anaphor += " "

    ner_spacy_ant = ner_spacy (antecedent)
    ner_spacy_ana = ner_spacy (anaphor)

    #print ("Antecedent: ", antecedent)
    #for ents in ner_spacy_ant.ents:
      #print (ents.text, ents.label_ )

    #print ("Anaphor: ", anaphor)
    flag = False

    for ents in ner_spacy_ana.ents:
      for ant_ents in ner_spacy_ana.ents:
        if (ents.label_ is ant_ents.label_):
          row.append (1)
          flag = True
          break
      if (flag == True):
        break

    if (flag == False):
      row.append (0)

    #Feature 9 (Gender Agreement)
    #Note dependent on Feature 8's antecedent and anaphor variable
    # 1 = Match
    # 2 = Mismatch
    # 3 = Unknown
    gender_agreement = 3
    if (((antecedent.lower () in male_identifiers) and (anaphor.lower () in male_identifiers)) or
       ((antecedent.lower () in female_identifiers) and (anaphor.lower () in female_identifiers))):
       gender_agreement = 1

    if (((antecedent.lower () in male_identifiers) and (anaphor.lower () in female_identifiers)) or
       ((antecedent.lower () in female_identifiers) and (anaphor.lower () in male_identifiers))):
       gender_agreement = 2

    row.append (gender_agreement)

    #Feature 10 (Both Proper-Name)
    a_pos_tag_list = []
    b_pos_tag_list = []
    both_proper_names = 0

    for pos_iter_index in range (a_s_idx, a_e_idx+1):
      a_pos_tag_list.append (a_wordlst[pos_iter_index].pos_tag)

    for pos_iter_index in range (b_s_idx, b_e_idx+1):
      b_pos_tag_list.append (b_wordlst[pos_iter_index].pos_tag)

    if (("NNP" in a_pos_tag_list) or ("NNPS" in a_pos_tag_list)):
      if (("NNP" in b_pos_tag_list) or ("NNPS" in b_pos_tag_list)):
        both_proper_names = 1

    row.append (both_proper_names)

    #Feature 11 (Alias Feature)
    #Check the NER tag for the antecedents and anaphor
    a_tok_list = []
    a_ner_tag_list = []
    a_atleast_one_valid = False
    a_ner_label_type = ""

    b_tok_list = []
    b_ner_tag_list = []
    b_atleast_one_valid = False 
    b_ner_label_type = ""

    for ner_iter_index  in range (a_s_idx, a_e_idx+1):
      word = a_wordlst[ner_iter_index]
      if (word.NER_tag != 'O'):
        a_atleast_one_valid = True
        a_ner_label_type = word.NER_label
      a_ner_tag_list.append (word.NER_tag)
      a_tok_list.append (word.word)

    for ner_iter_index  in range (b_s_idx, b_e_idx+1):
      word = b_wordlst[ner_iter_index]
      if (word.NER_tag != 'O'):
        b_atleast_one_valid = True
        b_ner_label_type = word.NER_label
      b_ner_tag_list.append (word.NER_tag)
      b_tok_list.append (word.word)

    if ((a_atleast_one_valid != b_atleast_one_valid) or 
        ((a_atleast_one_valid == False) and (b_atleast_one_valid == False)) or
        (a_ner_label_type != b_ner_label_type)):
      row.append (0)
    else:
      a_set = set(a_tok_list)
      b_set = set (b_tok_list)

      if (a_set & b_set):
        #print ("NER Type :", antecedent, a_ner_label_type, anaphor, b_ner_label_type, "Match", )
        row.append (1)
      else:
        row.append (0)

    #Feature 12 (Appositive Feature)
    #Dependent on Feature 10
    #Check if the strings are close to each other
    sent_diff = b_sentid - a_sentid

    if (sent_diff > 0):
      row.append (0)
    else:
      #They are in the same line, now check how close they are to each other
      index_diff = b_s_idx - a_e_idx
      if (index_diff > 1):
        row.append (0)
      else:
        #We have established they are close, now check what is in between if there is any.
        verb_found = False
        if (index_diff > 0):
          for temp_index  in range (a_e_idx+1, b_s_idx):
            if (a_wordlst[temp_index].pos_tag in ["MD", "VBD", "VB", "VBG", "VBN", "VBZ"]):
              verb_found = True
              break
          if (verb_found == True):
            row.append (0)
        
        if (verb_found == False):
          #Check if one of them are proper noun
          #Dependent on Feature 10
          pos_tag_set = set (a_pos_tag_list + b_pos_tag_list)
          if ("NNP" in pos_tag_set):
            if (b_wordlst[b_s_idx].word.lower() == "the"):
              print ("Appositive Feature : Index Diff : ", index_diff, "Ante :", antecedent, "Anaphor :", anaphor)
              row.append (1)
            else:
              row.append (0)
          else:
            row.append (0)

    #Debug Print
    # print_mention_pair_all_details (top_obj, lst[i], row)
    #Feature - ENDS       
    if (filename == None):
      return np.asarray (row), True

    for idx, col_val in enumerate (row):
      filename.write ("%s" %col_val)
      if (idx != len(row) -1):
        filename.write (", ")

    filename.write ("\n")




def create_features (top_obj):
  fv_file = open ("feature_vector.input", 'w+')
  result_fv_row_list = []
  pos_lst = top_obj.pos_list
  neg_lst = top_obj.selected_neg_list
 
  print ("Positive data")
  result_fv_row_list = bulk_mp_modular_create_features_handler (pos_lst, top_obj, 1, result_fv_row_list)

  print ("Negative data")
  result_fv_row_list = bulk_mp_modular_create_features_handler (neg_lst, top_obj, 0, result_fv_row_list)

  # Shuffle the data so that +ve and -ve will get randomized
  random.seed (100)
  random.shuffle (result_fv_row_list)

  # Write the shuffled data
  dump_features_to_file (fv_file, result_fv_row_list)
  fv_file.close ()



