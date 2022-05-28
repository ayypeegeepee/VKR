import os
from pyntcloud import PyntCloud


FILENAME = "forest_bauman"
FILENAME = "broken_single_tree_no_leaves"
# FILENAME = "single_tree_with_leaves"

OUT_FORMAT = "ply"


def main():
    pc = PyntCloud.from_file(os.path.join("..", "files", f"{FILENAME}.las"))
    pc.to_file(os.path.join("..", "files", f"{FILENAME}.{OUT_FORMAT}"))


if __name__ == "__main__":
    main()
