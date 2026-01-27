import os

"""
    Delete all directories, subdirectories and files under a certain directory. The directory itself gets destroyed
    as well
    
    IMPORTANT : This is a destructive acts, these files get destroyed, not recycled. Proceed with caution
    
    Author :        Martijn Folmer
    Date created :  04-01-2026    
"""


def delete_everything(root_dir, n_verbose=1):
    for current_root, dirs, files in os.walk(root_dir, topdown=False):
        for i_f, f in enumerate(files):
            try:
                os.remove(os.path.join(current_root, f))
            except Exception as e:
                print(e)

            if i_f % n_verbose == 0:
                print(f"We are at {i_f} / {len(files)} in {current_root}")

        for d in dirs:
            try:
                os.rmdir(os.path.join(current_root, d))
            except Exception as e:
                print(e)

    os.rmdir(root_dir)


if __name__ == "__main__":
    TARGET_DIR = "images2"      # which directory to delete
    N_VERBOSE = 2               # how often to print where we are

    delete_everything(TARGET_DIR, N_VERBOSE)

