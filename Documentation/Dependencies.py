"""
Created on July 31, 2018

@author: Alejandro Molina
"""
import glob
import os
import re
import subprocess
from stdlib_list import stdlib_list

stdlibs = stdlib_list()


def is_std_lib(module_name):
    return module_name in stdlibs


def is_module(name):
    status, _ = subprocess.getstatusoutput("pip3 show " + name)
    return status == 0


if __name__ == "__main__":
    current_path = os.path.split(__file__)[0]
    project_path = os.path.abspath(current_path + "/../")
    print(project_path)

    all_imports = set()

    for filename in glob.iglob(project_path + "/**/*.py", recursive=True):

        # module finder couldn't parse our files :(

        with open(filename) as f:
            imports = filter(lambda s: "import" in s, f.read().splitlines())  # get imports
            imports = map(lambda s: s.strip(), imports)  # strip
            imports = filter(lambda s: re.match(r"^(import|from)\s+[A-Za-z\\.]+($|\s+[A-Za-z]+)", s), imports)
            imports = map(lambda s: re.search("^((import|from)\s+[A-Za-z]+)($|\\.|\s+)", s).group(1), imports)
            imports = map(lambda s: s.split()[1], imports)  # get just the names
            imports = filter(lambda s: not is_std_lib(s), imports)
            all_imports |= set(imports)

    print("source code checked")

    all_imports = sorted(all_imports, key=lambda s: s.lower())

    module_names = []
    other_names = []
    for import_name in all_imports:
        if is_module(import_name):
            module_names.append(import_name)
        else:
            other_names.append(import_name)

    print("----------------------modules----------------------")
    print("\n".join(module_names))
    print("----------------------other----------------------")
    print("\n".join(other_names))
