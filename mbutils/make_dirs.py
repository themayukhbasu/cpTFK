import os


def make_dirs(path):
    #   Creates directories along a path
    #   Builds the path if needed
    #   Arguments:    path

    if not os.path.exists(path):
        os.makedirs(path)
        print("Successfully Created: ", path)
        return True
    else:
        print('Path Already exists: ', path)
        return False


path = "tensorflow/tensorflow/python/keras/layers"
make_dirs(path)
