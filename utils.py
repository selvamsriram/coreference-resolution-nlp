import class_defs 
import re
import nltk

def extract_document (doc_obj, input_file, key_file):
  ifp = open (input_file)
  kfp = open (key_file)

  line_num = 0
  for line in ifp:
      doc_obj.sentences[line_num] = class_defs.sentence (line)
      line_num += 1

  ifp.close ()
  kfp.close ()


def preprocess_sentence (doc_sentence):
  doc_sentence = doc_sentence.strip ('\n')
  pattern = re.compile (r'<.*?>')
  return pattern.sub ('', doc_sentence)


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


