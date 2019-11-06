import class_defs
import utils
import utils_temp


def main():
  top_obj = class_defs.top ()
  input_list_fp = open ("input_file_list.txt")
  key_list_fp = open ("key_file_list.txt")

  for ifile,kfile in zip(input_list_fp, key_list_fp):
    ifile = ifile.strip ('\n')
    kfile = kfile.strip ('\n')
    print ("Now Processing {}, {}".format(ifile, kfile))

    top_obj.docs[ifile] = class_defs.document (top_obj, ifile, kfile)


  input_list_fp.close ()
  key_list_fp.close ()
  #Select a subset of the negative data generated randomly
  utils_temp.select_neg_data (top_obj, 2)
  print ("Pos Create Ana Encountered : ", top_obj.pos_create_ana_encountered)
  print ("Number of Positive and Negative Samples Generated")
  print ("Positive : {} Negative {} Selected Negative {}".format (len(top_obj.pos_list), len(top_obj.neg_list), len(top_obj.selected_neg_list)))

  #Debug Prints
  #for key, dobj in top_obj.docs.items():
  #  utils.compare_total_antecedents(dobj)

  utils.create_features (top_obj)

if __name__ =="__main__":
  main ()
