# Similar to https://github.com/naver/mast3r/blob/main/mast3r/utils/path_to_dust3r.py

import sys
import os.path as path

HERE_PATH = path.normpath(path.dirname(__file__))
MASt3R_REPO_PATH = path.normpath(path.join(HERE_PATH, "../../dynamic_mast3r"))
MASt3R_LIB_PATH = path.join(MASt3R_REPO_PATH, "mast3r")
# check the presence of models directory in repo to be sure its cloned
if path.isdir(MASt3R_LIB_PATH):
    # workaround for sibling import
    sys.path.insert(0, MASt3R_REPO_PATH)
else:
    raise ImportError(
        f"dust3r is not initialized, could not find: {MASt3R_LIB_PATH}.\n "
        "Did you forget to run 'git submodule update --init --recursive' ?"
    )
