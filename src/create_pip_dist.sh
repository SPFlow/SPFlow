set -e

echo "Testing"
PYTHONPATH=.  python3 -m unittest discover

echo "Creating package"
python3 setup.py sdist bdist_wheel