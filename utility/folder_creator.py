import os
import shutil
#it creates one folder for each time, if the folder you want to delete doesn't exist, it doesn't work.
def folder_creator(path,deleteIfExists):
    try:
        if deleteIfExists == 1:
            try:
                shutil.rmtree(path)
            except:
                pass
        os.makedirs(path, exist_ok=True)
    except OSError:
        #print("Creation of the directory %s failed" % folder_name)
        pass
    else:
        #print("Successfully created the directory %s " % path)
        pass

