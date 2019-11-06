import class_defs
import re
import nltk
import random
import utils
import spacy
import sys
sys.path.insert(1,'models/Logistic_Regression')
import lr_test

def debug_printer(doc_obj):
  sent_obj = doc_obj.sentences[28]
  for marker in sent_obj.markables:
    print ("DPRINTER: ", marker.coref_id, marker.flags)

def get_all_antecedents_from_input_file (gold_sentence):
  max_sent_len = len (gold_sentence)
  crossed_sent_tag = False
  cur_coref_id = ""
  cur_antecedent_str = ""
  coref_id_list = []
  ante_str_list = []
  start_extraction = False
  for i in range (0, max_sent_len):
    if (crossed_sent_tag == False):
      if (gold_sentence[i] == '>'):
        crossed_sent_tag = True 
      continue
    if (gold_sentence[i] == '<'):
      #This can be one of the following
      # <COREF ID="X0">
      # </COREF>
      # </S>
      if (gold_sentence[i:i+4] == "</S>"):
        #Sentence is complete
        break
      if (gold_sentence[i:i+8] == "</COREF>"):
        ante_str_list.append (cur_antecedent_str)
        cur_antecedent_str = ""
        start_extraction = False
      if (gold_sentence[i:i+11] == "<COREF ID=\""):
        t_loop_idx = i+11
        for t_loop_idx in range (i+11, max_sent_len):
          if (gold_sentence[t_loop_idx] == "\""):
            break
          else:
            cur_coref_id += gold_sentence[t_loop_idx]
        coref_id_list.append (cur_coref_id)

    elif (gold_sentence[i] == ">"):
      if (cur_coref_id != ""):
        start_extraction = True
  
    if (start_extraction == True):
      if (cur_coref_id != ""):
        cur_coref_id = ""
        continue
      cur_antecedent_str += gold_sentence[i]

  print (coref_id_list)
  print (ante_str_list)
  return coref_id_list, ante_str_list

def create_markable_for_coref_id_and_str (doc_obj, sent_obj, coref_id, ante_str):
  tokenized_ante_str = spacy_get_tokenized_word (doc_obj, ante_str)
  max_len = len(sent_obj.word_list)
  len_of_ante_str = len (tokenized_ante_str)
  max_start_idx = -1
  max_end_idx = -1

  match = False
  for i in range (0, max_len):
    #Check if the first token matches
    if sent_obj.word_list[i].word == tokenized_ante_str[0]:
      match = True
      for j in range (1, len_of_ante_str):
        if sent_obj.word_list[i+j].word != tokenized_ante_str[j]:
          match = False
          break
      if (match == True):
        #Max pattern is found in the tokenized obj
        max_start_idx = i
        max_end_idx = i+ len_of_ante_str - 1
        break
  if (max_start_idx != -1) and (max_end_idx != -1):
    markable_obj = class_defs.markable (max_start_idx, max_end_idx, -1, -1, coref_id, class_defs.MARKABLE_FLAG_ANTECEDENT)
    sent_obj.gold_markables.append (markable_obj)


def spacy_extract_markables_from_input_file (doc_obj, line_num, sent_tag_unrem, sent_tag_rem):
  coref_id_list, ante_str_list = get_all_antecedents_from_input_file (sent_tag_unrem)
  sent_obj = doc_obj.sentences[line_num]

  print ("SID : ", line_num, "Coref ID : ", coref_id_list, "ante_str_list : ", ante_str_list)

  for coref_id, ante_str in zip (coref_id_list, ante_str_list):
    #For each of the coref ID and ante string one markable is needed
    create_markable_for_coref_id_and_str (doc_obj, sent_obj, coref_id, ante_str)

def spacy_handle_key_file (doc_obj, kfp):
  for line in kfp:
    line = line.strip ('\n')
    #Patten Check if the string matches for "<COREF ID="
    if (len(line) < 2):
      continue

    if ("<COREF ID=" in line):
      e_start_index = 11
      coref_id_string = ""
      for t_loop_idx in range (e_start_index, len(line)):
        if (line[t_loop_idx] == "\""):
          break
        else:
          coref_id_string += line [t_loop_idx]

      #print (coref_id_string)
    else:
      list_of_str = []
      extract = False
      string_required = ""
      for i in range (0, len(line)):
        if (line[i] == "{"):
          extract = True
        elif (line[i] == "}"):
          extract = False
          string_required = string_required.lstrip (' ')
          list_of_str.append(string_required)
          string_required = ""
        else:
          string_required += line[i]
      #Debug Print
      #print ("Sentence Num :", list_of_str[0], "Max :", list_of_str[1], "Min :", list_of_str[2])
      #Now we got all the anaphorts in a list format, lets do the following tasks
      # 1. Get the sentence from the doc
      # 2. Tokenize the max and min
      # 3. Iterate through the word_list inside sentence and find from which index to index there is a overlap.
      sentence_obj = doc_obj.sentences[int(list_of_str[0])]
      tokenized_max = spacy_get_tokenized_word (doc_obj, list_of_str[1])
      tokenized_min = spacy_get_tokenized_word (doc_obj, list_of_str[2])
      max_len = len(sentence_obj.word_list)
      match = False
      max_start_idx = -1
      max_end_idx = -1
      min_start_idx = -1
      min_end_idx = -1
      len_of_max_str = len (tokenized_max)
      for i in range (0, max_len):
        #Check if the first token matches
        if sentence_obj.word_list[i].word == tokenized_max[0]:
          match = True
          for j  in range (1, len_of_max_str):
            if sentence_obj.word_list[i+j].word != tokenized_max[j]:
              match = False
              break
          if (match == True):
            #Max pattern is found in the tokenized obj
            max_start_idx = i
            max_end_idx = i+len_of_max_str - 1
            break
      len_of_min_str = len (tokenized_min)
      for i in range (0, max_len):
        #Check if the first token matches
        if sentence_obj.word_list[i].word == tokenized_min[0]:
          match = True
          for j  in range (1, len_of_min_str):
            if sentence_obj.word_list[i+j].word != tokenized_min[j]:
              match = False
              break
          if (match == True):
            #Max pattern is found in the tokenized obj
            min_start_idx = i
            min_end_idx = i+len_of_min_str -1
            break

      markable_obj = class_defs.markable (max_start_idx, max_end_idx, min_start_idx, min_end_idx, coref_id_string, class_defs.MARKABLE_FLAG_ANAPHOR)
      sent_obj = doc_obj.sentences[int(list_of_str[0])]
      sent_obj.gold_markables.append (markable_obj)

      '''
      #Debug Prints
      print ("Sentence Num :", list_of_str[0], "Max :", list_of_str[1], "Min :", list_of_str[2])
      print_word = ""
      for print_index in range (max_start_idx, max_end_idx+1):
        print_word  += sentence_obj.word_list[print_index].word + " "
      print ("Tokenized Max :", print_word)

      print_word = ""
      for print_index in range (min_start_idx, min_end_idx+1):
        print_word  += sentence_obj.word_list[print_index].word + " "

      print ("Tokenized Min :", print_word)
      '''

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
      end_index = index - (number_of_completed_corefs * 10) -7 -7 -1
      create_markable_flag = True

      #Debug Prints
      if (antecedent != sent_tag_rem[begin_index:end_index+1]):
        #print ("Mistmatched Antecedent")
        #print ("Coreference ID ", coref_id_string, "Unremoved Antecedent ", antecedent)
        #print ("Coreference ID ", coref_id_string, "Removed Antecedent ", sent_tag_rem[begin_index:end_index])
        create_markable_flag = False

      #Create a markable_obj
      if (create_markable_flag == True):
        markable_obj = class_defs.markable (begin_index, end_index, -1, -1, coref_id_string, class_defs.MARKABLE_FLAG_ANTECEDENT)
        sent_obj = doc_obj.sentences[line_num]
        sent_obj.gold_markables.append (markable_obj)
      begin_index = -1
      number_of_completed_corefs += 1

def handle_key_file (doc_obj, kfp):
  for line in kfp:
    line = line.strip ('\n')
    #Patten Check if the string matches for "<COREF ID="
    if (len(line) < 2):
      continue

    if ("<COREF ID=" in line):
      tokens = nltk.word_tokenize (line)
      coref_id_string = tokens[4]
      #print (coref_id_string)
    else:
      list_of_str = []
      extract = False
      string_required = ""
      for i in range (0, len(line)):
        if (line[i] == "{"):
          extract = True
        elif (line[i] == "}"):
          extract = False
          string_required = string_required.lstrip (' ')
          list_of_str.append(string_required)
          string_required = ""
        else:
          string_required += line[i]
      #Debug Print
      #print ("Sentence Num :", list_of_str[0], "Max :", list_of_str[1], "Min :", list_of_str[2])
      #Now we got all the anaphorts in a list format, lets do the following tasks
      # 1. Get the sentence from the doc
      # 2. Tokenize the max and min
      # 3. Iterate through the word_list inside sentence and find from which index to index there is a overlap.
      sentence_obj = doc_obj.sentences[int(list_of_str[0])]
      tokenized_max = nltk.word_tokenize (list_of_str[1])
      tokenized_min = nltk.word_tokenize (list_of_str[2])
      max_len = len(sentence_obj.word_list)
      match = False
      max_start_idx = -1
      max_end_idx = -1
      min_start_idx = -1
      min_end_idx = -1
      len_of_max_str = len (tokenized_max)
      for i in range (0, max_len):
        #Check if the first token matches
        if sentence_obj.word_list[i].word == tokenized_max[0]:
          match = True
          for j  in range (1, len_of_max_str):
            if sentence_obj.word_list[i+j].word != tokenized_max[j]:
              match = False
              break
          if (match == True):
            #Max pattern is found in the tokenized obj
            max_start_idx = i
            max_end_idx = i+len_of_max_str - 1
            break
      len_of_min_str = len (tokenized_min)
      for i in range (0, max_len):
        #Check if the first token matches
        if sentence_obj.word_list[i].word == tokenized_min[0]:
          match = True
          for j  in range (1, len_of_min_str):
            if sentence_obj.word_list[i+j].word != tokenized_min[j]:
              match = False
              break
          if (match == True):
            #Max pattern is found in the tokenized obj
            min_start_idx = i
            min_end_idx = i+len_of_min_str -1
            break

      markable_obj = class_defs.markable (max_start_idx, max_end_idx, min_start_idx, min_end_idx, coref_id_string, class_defs.MARKABLE_FLAG_ANAPHOR)
      sent_obj = doc_obj.sentences[int(list_of_str[0])]
      sent_obj.gold_markables.append (markable_obj)

      '''
      #Debug Prints
      print ("Sentence Num :", list_of_str[0], "Max :", list_of_str[1], "Min :", list_of_str[2])
      print_word = ""
      for print_index in range (max_start_idx, max_end_idx+1):
        print_word  += sentence_obj.word_list[print_index].word + " "
      print ("Tokenized Max :", print_word)

      print_word = ""
      for print_index in range (min_start_idx, min_end_idx+1):
        print_word  += sentence_obj.word_list[print_index].word + " "

      print ("Tokenized Min :", print_word)
      '''

def create_gold_markable_list (doc_obj, input_file, key_file):
  ifp = open (input_file)

  line_num = 0
  for line in ifp:
    line = line.strip ('\n')
    sent_tag_unrem = line
    sent_tag_rem = utils.preprocess_sentence  (line)
    #extract_markables_from_input_file (doc_obj, line_num, sent_tag_unrem, sent_tag_rem)
    spacy_extract_markables_from_input_file (doc_obj, line_num, sent_tag_unrem, sent_tag_rem)
    line_num += 1
  ifp.close ()
  
  if (key_file == None):
    return

  kfp = open (key_file)
  #handle_key_file (doc_obj, kfp)  
  spacy_handle_key_file (doc_obj, kfp)  
  kfp.close ()

def create_pos_data_using_doc (doc_obj):
  #print ("From create_pos_data_using_doc")
  #debug_printer(doc_obj)
  top = doc_obj.top_obj
  max_line = len (doc_obj.sentences)

  for line_num in range (0, max_line):
    sentence = doc_obj.sentences[line_num]
    markable_list = sentence.markables 
    max_mark_len = len (markable_list)
    for mark_index in range (0, max_mark_len):
      marker = markable_list[mark_index]
      if (marker.flags == class_defs.MARKABLE_FLAG_ANTECEDENT):
        #Add it to the coref cluster info
        clus_info = class_defs.cluster_info_piece (line_num, mark_index)
        doc_obj.clusters_info[marker.coref_id] = clus_info

      elif (marker.flags == class_defs.MARKABLE_FLAG_ANAPHOR):
        top.pos_create_ana_encountered += 1
        #We gotta find the prev mention of this cluster
        if (marker.coref_id  in doc_obj.clusters_info):
          clus_info = doc_obj.clusters_info[marker.coref_id]
          '''
            #Debug Prints
            print ("Added because antecedent is found")
          '''
          #Pair it up
          prev_mention_sent_obj = doc_obj.sentences[clus_info.sent_idx]
          prev_mention_marker   = prev_mention_sent_obj.markables[clus_info.mark_idx] 
          mp = class_defs.mention_pair (doc_obj, clus_info.sent_idx, clus_info.mark_idx, 
                                      line_num, mark_index)

          #Insert the pair in POS list
          top.pos_list.append (mp)

          #Update the latest mention to this mention
          clus_info.sent_idx = line_num
          clus_info.mark_idx = mark_index
          doc_obj.clusters_info[marker.coref_id] = clus_info
        else:
          print ("Skipping because antecedent not in cluster")
          print ("Sentence : ", doc_obj.sentences[line_num].full_sentence)
          print ("line number :", line_num, "Mark Idx : ", mark_index, "Coref ID : ", marker.coref_id)

def create_neg_data_using_doc (doc_obj):
  top = doc_obj.top_obj
  max_line = len (doc_obj.sentences)

  #flush all the cluster keys
  doc_obj.clusters_info.clear ()

  for line_num in range (0, max_line):
    sentence = doc_obj.sentences[line_num]
    markable_list = sentence.markables 
    max_mark_len = len (markable_list)
 
    for mark_index in range (0, max_mark_len):
      marker = markable_list[mark_index]
      if (marker.flags == class_defs.MARKABLE_FLAG_ANTECEDENT):
        #Add it to the coref cluster info
        clus_info = class_defs.cluster_info_piece (line_num, mark_index)
        doc_obj.clusters_info[marker.coref_id] = clus_info
      elif (marker.flags == class_defs.MARKABLE_FLAG_ANAPHOR):
        #Pair a anaphor with anything between the prev mention to this anaphor
        #We gotta find the prev mention of this cluster
        if (marker.coref_id  in doc_obj.clusters_info):
          clus_info = doc_obj.clusters_info[marker.coref_id]

        prev_mention_sent_idx = clus_info.sent_idx
        prev_mention_mark_idx = clus_info.mark_idx
        #Begin the crazy pairing 
        #We will pair this anaphor with everything but its antecedent.
        #Anything in the middle like another antecedent or nothing works good for A component
        sent_iter_idx = 0
        markable_end_idx = 0
        markable_start_idx = 0
        for sent_iter_idx in range (prev_mention_sent_idx, line_num+1):
          if sent_iter_idx == line_num:
            markable_end_idx = mark_index
          else:
            markable_end_idx = len (doc_obj.sentences[sent_iter_idx].markables)

          if sent_iter_idx == prev_mention_sent_idx:
            markable_start_idx = prev_mention_mark_idx + 1
          else:
            markable_start_idx = 0
          
          markable_iter_idx = 0
          for markable_iter_idx in range (markable_start_idx, markable_end_idx):
            a_comp_sent_obj = doc_obj.sentences[sent_iter_idx]
            a_comp_marker   = a_comp_sent_obj.markables[markable_iter_idx] 
            mp = class_defs.mention_pair (doc_obj, sent_iter_idx, markable_iter_idx, 
                                      line_num, mark_index)
            #Insert the pair in POS list
            top.neg_list.append (mp)

        #Update the latest mention to this mention
        clus_info.sent_idx = line_num
        clus_info.mark_idx = mark_index
        doc_obj.clusters_info[marker.coref_id] = clus_info



def create_data_using_doc (doc_obj, pos_reqd, neg_reqd):
  if (pos_reqd == True):
    create_pos_data_using_doc (doc_obj)
  
  if (neg_reqd == True):
    create_neg_data_using_doc (doc_obj)

def select_neg_data (top_obj, neg_ratio):
  pos_size = len (top_obj.pos_list)
  neg_size = len (top_obj.neg_list)

  required_neg = pos_size * neg_ratio
  if (required_neg > neg_size):
    print ("For the given ratio we don't have enough neg samples")
  
  #Set seed and randomly select required samples
  random.seed (100)
  top_obj.selected_neg_list = random.sample (top_obj.neg_list, required_neg)

def take_care_of_missed_antecedents (doc_obj, sent_obj, sent_num):
  gold_markables = sent_obj.gold_markables

  antecedent_markables = []
  #Filter the antecedents
  gm_len = len (gold_markables)
  for i in range (0, gm_len):
    marker =  gold_markables[i]
    if (marker.flags == class_defs.MARKABLE_FLAG_ANTECEDENT):
      antecedent_markables.append (marker)

  gm_len = len (antecedent_markables)
  #Nothing is missed
  if (gm_len == 0):
    return

  for i in range (0, gm_len):
    g_marker = antecedent_markables[i]
    #Check if this g_marker's identical twin is found in our markable list
    our_markable_len = len (sent_obj.markables)
    inserted_or_found = False 
    for j in range (0, our_markable_len):
      o_marker = sent_obj.markables[j]
      if ((o_marker.w_s_idx == g_marker.w_s_idx) and (o_marker.w_e_idx == g_marker.w_e_idx)):
        if (o_marker.flags != g_marker.flags):
          #Update if there is a mismatch in flags
          o_marker.flags = class_defs.MARKABLE_FLAG_ANTECEDENT
          sent_obj.markables[j] = o_marker
        inserted_or_found = True 
        break
      elif ((o_marker.w_s_idx > g_marker.w_s_idx)):
        #We have crossed the place where we should have found this but we still haven't so add it
        #Create a new marker_obj
        new_marker = class_defs.markable (g_marker.w_s_idx, g_marker.w_e_idx, -1, -1, g_marker.coref_id, g_marker.flags)
        sent_obj.markables.insert (j, new_marker)
        inserted_or_found = True
        break
      
    if (inserted_or_found == False):
      new_marker = class_defs.markable (g_marker.w_s_idx, g_marker.w_e_idx, -1, -1, g_marker.coref_id, g_marker.flags)
      sent_obj.markables.append (new_marker)


def predict_wrapper (doc_obj, mp):
  #To be filled after model is ready
  y_prob = lr_test.test_lr_model (mp, doc_obj.top_obj)
  return y_prob

def get_predicted_coref_id_given_mps (doc_obj, test_mp_list):
  max_coref_id = None
  max_score = 0

  #Predict probability or use softmax and get the coref_id responsible for max score
  for mp in test_mp_list:
    prediction_score = predict_wrapper (doc_obj, mp)
    if (prediction_score > max_score):
      max_score = prediction_score
      max_coref_id = mp.coref_id
  
  return max_coref_id

def create_mention_pairs_for_testing (doc_obj, coref_id, a_sent_idx, a_mark_idx, b_sent_idx, b_mark_idx):
  mp = class_defs.mention_pair (doc_obj, a_sent_idx, a_mark_idx, b_sent_idx, b_mark_idx)
  mp.coref_id = coref_id
  return mp

def predict_coref_id_of_cluster (doc_obj, line_num, mark_index):
  #Pair this markable with every other markable found in the cluster and creat mention pairs
  test_mp_list = []
  for coref_id, clus_info_obj in doc_obj.clusters_info.items ():
    #Pair up the clus info obj and our mark index and generate mp
    mp = create_mention_pairs_for_testing (doc_obj, coref_id, 
                                           clus_info_obj.sent_idx, clus_info_obj.mark_idx,
                                           line_num, mark_index)
    test_mp_list.append (mp)

  predicted_coref_id = get_predicted_coref_id_given_mps (doc_obj, test_mp_list)
  return predicted_coref_id

def process_testing_per_sentence (doc_obj, line_num):
  sent_obj = doc_obj.sentences[line_num]
  max_markable = len(sent_obj.markables)
  for mark_index in range (0, max_markable):
    cur_marker = sent_obj.markables[mark_index]

    if (cur_marker.flags == class_defs.MARKABLE_FLAG_ANTECEDENT):
      #Update the last mention of this coref cluster
      clus_info_obj = class_defs.cluster_info_piece (line_num, mark_index)
      doc_obj.clusters_info[cur_marker.coref_id] = clus_info_obj
      new_cluster_list = [clus_info_obj]
      doc_obj.result_clusters_info[cur_marker.coref_id] = new_cluster_list
    else:
      #This is a markable without coref id yet
      #So, we begin creating mention pairs and work on a coref id for this.
      predicted_coref_id = predict_coref_id_of_cluster (doc_obj, line_num, mark_index)
      if (predicted_coref_id != None):
        #Update the appearance on the cluster info
        clus_info_obj = class_defs.cluster_info_piece (line_num, mark_index)
        doc_obj.clusters_info[predicted_coref_id] = clus_info_obj
        doc_obj.result_clusters_info[predicted_coref_id].append (clus_info_obj)
        
        #Set flag as Anaphor and also update the markable 
        cur_marker.flags = class_defs.MARKABLE_FLAG_ANAPHOR
        cur_marker.coref_id = predicted_coref_id
        sent_obj.markables[mark_index] = cur_marker


def begin_testing (doc_obj):
  #Clear the cluster info before beginning to process this doc
  doc_obj.clusters_info.clear ()
  max_sentence = len (doc_obj.sentences)
  for line_num in range (0, max_sentence):
    process_testing_per_sentence (doc_obj, line_num)


def spacy_get_tokenized_word (doc_obj, doc_sentence):
  spacy_obj = doc_obj.top_obj.spacy_obj
  doc = spacy_obj (doc_sentence)
  token_list = []
  for tok in doc:
    token_list.append (tok.text)

  return token_list

def spacy_get_pos_tags (doc_obj, doc_sentence):
  spacy_obj = doc_obj.top_obj.spacy_obj
  doc = spacy_obj (doc_sentence)
  pos_tag_list = []

  #We opt for Penn Tree tags over spacy Tags
  for tok in doc:
    pos_tag_list.append (tok.tag_)
  
  return pos_tag_list

def spacy_get_ner_bio_tag (doc_obj, doc_sentence):
  spacy_obj = doc_obj.top_obj.spacy_obj
  doc = spacy_obj (doc_sentence)
  ner_iob_tag_list = []
  ner_label_list = []
  label = None
  for ents in doc:
    ner_iob_tag_list.append (ents.ent_iob_)
    ner_label_list.append (ents.ent_type_)

  return ner_iob_tag_list, ner_label_list

def spacy_get_np_tags_filled (doc_obj, sent_obj, doc_sentence):
  spacy_obj = doc_obj.top_obj.spacy_obj
  doc = spacy_obj (doc_sentence)

  for chunk in doc.noun_chunks:
    sent_obj.word_list[chunk.start].chunk_tag = "B-NP"
    for i in range (chunk.start+1, chunk.end):
      sent_obj.word_list[i].chunk_tag = "I-NP"
