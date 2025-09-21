import os

def is_running_on_cluster():
    return os.path.exists("/vol/")