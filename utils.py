import class_defs 

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