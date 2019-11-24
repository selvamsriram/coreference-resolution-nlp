import class_defs
import utils
import utils_temp
import sys
import os

def get_word_from_markables (doc_obj, sent_obj, markable_obj):
  start_index = markable_obj.w_s_idx
  end_index = markable_obj.w_e_idx+1
  ret_string = ""
  for index in range (start_index, end_index):
    word_obj = sent_obj.word_list[index]
    if index != end_index -1:
      ret_string += word_obj.word + " "
    else:
      ret_string += word_obj.word

  return ret_string 

def truncate_anaphor (antecedent, anaphor):
  lo_antecedent = antecedent.lower ()
  lo_anaphor = anaphor.lower()
  if lo_antecedent in lo_anaphor:
    ante_words = lo_antecedent.split (" ")
    ana_words = lo_anaphor.split (" ")
    max_len = len(ana_words)
    len_of_ante_str = len(ante_words)
    start_idx = 0
    end_idx = 0

    match = False
    for i in range (0, max_len):
      #Check if the first token matches
      if ana_words[i] == ante_words[0]:
        match = True
        for j in range (1, len_of_ante_str):
          if ana_words[i+j] != ante_words[j]:
            match = False
            break
        if (match == True):
          #Max pattern is found in the tokenized obj
          max_start_idx = i
          max_end_idx = i+ len_of_ante_str
          break
    ana_words = anaphor.split (" ")
    if (match == True):
      selected_ana_words = ana_words[max_start_idx: max_end_idx]
      anaphor = " ".join(selected_ana_words)
    return anaphor
  else:
    return anaphor

def generate_cluster_specific_op (rfp, doc_obj, coref_id, cluster_info_list):
  print_str = "<COREF ID=\"" + coref_id+ "\">"
  antecedent_clus_obj = cluster_info_list[0]
  antecedent_sent_obj = doc_obj.sentences[antecedent_clus_obj.sent_idx]
  antecedent_marker = antecedent_sent_obj.markables[antecedent_clus_obj.mark_idx]
  antecedent_string = get_word_from_markables (doc_obj, antecedent_sent_obj, antecedent_marker)
  print_str += antecedent_string + "</COREF>\n"
  print (print_str)
  rfp.write (print_str)

  max_clus_list_len = len (cluster_info_list)
  for i in range (1, max_clus_list_len):
    clus_obj   = cluster_info_list[i] 
    sent_obj   = doc_obj.sentences[clus_obj.sent_idx]
    marker_obj = sent_obj.markables[clus_obj.mark_idx]
    
    anaphor_string = get_word_from_markables (doc_obj, sent_obj, marker_obj)

    anaphor_string = truncate_anaphor (antecedent_string, anaphor_string)
    print_str  = "{" + str(clus_obj.sent_idx) + "} {" + anaphor_string + "}\n"
    rfp.write (print_str)

def generate_doc_specific_op (doc_obj, doc_name, response_dir):
  dname_words = doc_name.split ("/")
  ip_fname = dname_words[len(dname_words)-1]
  ip_fname_words = ip_fname.split (".")
  file_id = ip_fname_words[0]
  response_file_name = file_id + ".response"
  response_file_name = os.path.join (response_dir, response_file_name)
  
  rfp = open (response_file_name, "w+")

  for coref_id, cluster_info_list in doc_obj.result_clusters_info.items():
    generate_cluster_specific_op (rfp, doc_obj, coref_id, cluster_info_list)
    rfp.write ("\n")

  rfp.close ()

def begin_output_processing (top_obj, response_dir):
  if not os.path.exists (response_dir):
    os.mkdir (response_dir)

  for doc_name, doc_obj in top_obj.docs.items ():
    generate_doc_specific_op (doc_obj, doc_name, response_dir)  


def begin_input_doc_processing (top_obj, input_file_list):
  ifp = open (input_file_list)

  for fname  in ifp:
    fname = fname.strip ("\n")
    print ("Processing : ", fname)
    top_obj.docs[fname] = class_defs.document (top_obj, fname, None)
    #Begin Testing
    #  This method gets the coref_id assigned to each markable that we detected
    utils_temp.begin_testing (top_obj.docs[fname])

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
  begin_output_processing (top_obj, response_dir)

main ()