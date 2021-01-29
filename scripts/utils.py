import sys


### Fonction pour le suivi du transfert Hadoop => local
def progress(filename, size, sent):
    sys.stdout.write("%s's progress: %.2f%%   \r" % (filename, float(sent) / float(size) * 100))
