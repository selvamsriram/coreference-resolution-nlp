import class_defs 
import re
import nltk

def extract_document (doc_obj, input_file, key_file):
    ifp = open (input_file)
    kfp = open (key_file)

    line_num = 0
    for line in ifp:
        doc_obj.sentences[line_num] = class_defs.sentence (line)
        print (line_num, line)
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

    