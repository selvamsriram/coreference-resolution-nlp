import class_defs
import utils
import utils_temp
import sys
import os.path

def generate_doc_specific_op (doc_obj, doc_name):

def begin_input_doc_processing (top_obj, input_file_list):
  ifp = open (input_file_list)

  for fname  in ifp:
    fname = fname.strip ("\n")
    print ("Processing : ", fname)
    top_obj.docs[fname] = class_defs.document (top_obj, fname, None)

  ifp.close ()

def main ():
  if (len(sys.argv) != 3):
    print ("Invalid Number of Arguments")
    return

  arg_list = sys.argv
  input_file_list = arg_list[1]
  response_dir = arg_list[2]

  top_obj = class_defs.top ()

  begin_input_doc_processing (top_obj, input_file_list)

main ()