set -e

echo "Testing"
PYTHONPATH=. python3 -m pytest --cache-clear spn/tests/

echo "Creating package"
rm -rf dist
rm -rf cache
rm -rf build
rm -rf spflow.egg-info
python3 setup.py sdist bdist_wheel