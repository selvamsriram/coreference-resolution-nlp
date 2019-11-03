import class_defs 
import re
import nltk
import utils_temp

def extract_document (doc_obj, input_file, key_file):
  ifp = open (input_file)
  kfp = open (key_file)

  line_num = 0
  for line in ifp:
      doc_obj.sentences[line_num] = class_defs.sentence (line)
      line_num += 1

  ifp.close ()
  kfp.close ()
  utils_temp.create_gold_markable_list (doc_obj, input_file, key_file)
  max_line = line_num
  for i in range (0, max_line):
    sent_obj = doc_obj.sentences[i]
    compare_gold_and_extracted_markables (doc_obj, sent_obj, i)

  top = doc_obj.top_obj
  print ("Results of markable extraction")
  print ("Matched % = ", top.matched_ana/(top.gold_ana))
  print ("Wasted Markable % = ", (top.total_markable - top.matched_ante_ana)/top.total_markable)


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
        extracted_table[i].coref_id = gold_table[k].coref_id
        if (gold_table[k].flags != class_defs.MARKABLE_FLAG_ANAPHOR):
          extracted_table[i].flags = gold_table[k].flags

        if (gold_table[k].flags == class_defs.MARKABLE_FLAG_ANAPHOR):
          if min_gold_sent[k] in temp_e:
            top_obj.matched_ana += 1 
            extracted_table[i].flags = gold_table[k].flags 

    if (extracted_table[i].flags == class_defs.MARKABLE_FLAG_ANAPHOR):
      print ("Coref ID : {} S ID : {} String : {}".format(extracted_table[i].coref_id,sent_num ,temp_e.lstrip ()))

    if ((extracted_table[i].flags == class_defs.MARKABLE_FLAG_ANAPHOR) or 
        (extracted_table[i].flags == class_defs.MARKABLE_FLAG_ANTECEDENT)):
       top_obj.matched_ante_ana += 1

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

    sent_obj.word_list.append (class_defs.word (curr_word, pos_tag, NER_tag, chunk_tag))

  markable_lst = compute_markable_table (sent_obj)
  sent_obj.markables = markable_lst



