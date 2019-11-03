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

def preprocess_sentence (doc_sentence):
  doc_sentence = doc_sentence.strip ('\n')
  pattern = re.compile (r'<.*?>')
  return pattern.sub ('', doc_sentence)



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
      if (np_tag == "0"):
        markable_obj = class_defs.markable (i, i, 0, 0)
        markable_lst.append (markable_obj)
        continue

    if (pos_tag == "NN" or pos_tag == "NNS" or pos_tag == "NNP" or pos_tag == "NNPS"):
      if (np_tag == "0"):
        markable_obj = class_defs.markable (i, i, 0, 0)
        markable_lst.append (markable_obj)
        continue

    if (np_tag != "0"):
      if (np_tag == "B-NP"):
        markable_obj = class_defs.markable (i, i, 0, 0)

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

  grammar = "NP: {<DT>?<JJ>*<NN>}"
  cp = nltk.RegexpParser(grammar)

  np_res = cp.parse (pos_tags)

  print (np_res)
  iob_chunk = nltk.chunk.tree2conlltags (np_res)
  print (iob_chunk)
  len_sen = len (iob_chunk)

  for i in range(len_sen):
    curr_word = iob_chunk[i][0]
    pos_tag = iob_chunk[i][1]
    chunk_tag = iob_chunk[i][2]
    NER_tag = ner_bio[i][2]

    sent_obj.word_list.append (class_defs.word (curr_word, pos_tag, NER_tag, chunk_tag))

  markable_lst = compute_markable_table (sent_obj)
  sent_obj.markables = markable_lst



