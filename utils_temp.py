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
      end_index = index - (number_of_completed_corefs * 10) -7 -7 -1

      #Debug Prints
      #print ("Coreference ID ", coref_id_string, "Unremoved Antecedent ", antecedent)
      #print ("Coreference ID ", coref_id_string, "Removed Antecedent ", sent_tag_rem[begin_index:end_index])

      #Create a markable_obj
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
      print (coref_id_string)
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
