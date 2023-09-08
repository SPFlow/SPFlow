import os
import subprocess
import tensorly as tl
import re

def run_test(test_file, backends):
    for backend in backends:
        tl.set_backend(backend)
        new_value = backend

        with open("F:\\Pycharm\\PycharmProjects\\SPFlow_Repo\\src\\tests\\tensorly_torch\\conftest.py", "r") as file:
            content = file.read()

        # Modify the CONFIG_VALUE using regular expression
        content = re.sub(r'(CONFIG_VALUE = )".*?"', rf'\1"{new_value}"', content)

        with open("F:\\Pycharm\\PycharmProjects\\SPFlow_Repo\\src\\tests\\tensorly_torch\\conftest.py", "w") as file:
            file.write(content)
        cmd = ["pytest", test_file]
        subprocess.run(cmd, shell=True)

def main():
    test_dir = "F:/Pycharm/PycharmProjects/SPFlow_Repo/src/tests/tensorly_torch/structure/general/layers/leaves/parametric"
    test_files = [file for file in os.listdir(test_dir) if file.endswith(".py")]

    backends = ["numpy","pytorch"]  # Replace with your actual settings

    for test_file in test_files:
        full_path = os.path.join(test_dir, test_file)
        run_test(full_path, backends)
        return

if __name__ == "__main__":
    main()