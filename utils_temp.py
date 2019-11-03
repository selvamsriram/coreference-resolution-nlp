import class_defs
import re
import nltk
import random
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
  kfp = open (key_file)

  line_num = 0
  for line in ifp:
    line = line.strip ('\n')
    sent_tag_unrem = line
    sent_tag_rem = utils.preprocess_sentence  (line)
    extract_markables_from_input_file (doc_obj, line_num, sent_tag_unrem, sent_tag_rem)
    line_num += 1
  ifp.close ()
  handle_key_file (doc_obj, kfp)  
  kfp.close ()

def create_pos_data_using_doc (doc_obj):
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
        #We gotta find the prev mention of this cluster
        if (marker.coref_id  in doc_obj.clusters_info):
          clus_info = doc_obj.clusters_info[marker.coref_id]
        '''
          #Debug Prints
          print ("Added because antecedent is found")
        else:
          print ("Skipping because antecedent not in cluster")
        '''
        #Pair it up
        prev_mention_sent_obj = doc_obj.sentences[clus_info.sent_idx]
        prev_mention_marker   = prev_mention_sent_obj.markables[clus_info.mark_idx] 
        mp = class_defs.mention_pair (doc_obj, clus_info.sent_idx, prev_mention_marker.w_s_idx, prev_mention_marker.w_e_idx, 
                                      line_num, marker.w_s_idx, marker.w_e_idx)

        #Insert the pair in POS list
        top.pos_list.append (mp)

        #Update the latest mention to this mention
        clus_info.sent_idx = line_num
        clus_info.mark_idx = mark_index
        doc_obj.clusters_info[marker.coref_id] = clus_info

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
            mp = class_defs.mention_pair (doc_obj, sent_iter_idx, a_comp_marker.w_s_idx, a_comp_marker.w_e_idx, 
                                      line_num, marker.w_s_idx, marker.w_e_idx)
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
