import os
import shutil
#it creates one folder for each time, if the folder you want to delete doesn't exist, it doesn't work.
def folder_creator(folder_name,deleteIfExists):
    try:
        if deleteIfExists == 1:
            try:
                shutil.rmtree(folder_name)
            except:
                pass
        os.mkdir(folder_name)
    except OSError:
        #print("Creation of the directory %s failed" % folder_name)
        pass
    else:
        print("Successfully created the directory %s " % folder_name)

