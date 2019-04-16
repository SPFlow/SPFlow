set -e

echo "Testing"
export PYTHONPATH=.
find spn/tests/test*.py -print0 | xargs -n 1 -0 python3 -m pytest

echo "Creating package"
rm -rf dist
rm -rf cache
rm -rf build
rm -rf spflow.egg-info
python3 setup.py sdist bdist_wheel