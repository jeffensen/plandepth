import os, sys, codecs, getopt

def convUTF8_1252(filename_in, filename_out):
    with codecs.open(filename_in, 'r', 'utf-8') as fin:
        with codecs.open(filename_out, 'w', 'windows-1252') as fout:
            for line in fin:
                fout.write(line)


def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ho:v", ["help", "output="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
    output = None
    verbose = False


    for fl in args:
        if ".m" in fl and "~" not in fl:
            print("Converting %s to Win-1252"%fl)
            convUTF8_1252(fl, "%s_win"%fl)

if __name__ == "__main__":
    main()