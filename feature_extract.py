import class_defs
import utils


def main():
    top_obj = class_defs.top ()
    input_list_fp = open ("input_file_list.txt")
    key_list_fp = open ("key_file_list.txt")

    for ifile,kfile in zip(input_list_fp, key_list_fp):
        ifile = ifile.strip ('\n')
        kfile = kfile.strip ('\n')
        print ("Now Processing {}, {}".format(ifile, kfile))

        top_obj.docs[ifile] = class_defs.document (ifile, kfile)

    input_list_fp.close ()
    key_list_fp.close ()

main ()