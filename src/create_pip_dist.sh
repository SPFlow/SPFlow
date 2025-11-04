set -e

export PYTHONPATH=.

echo "Creating package"
rm -rf dist
rm -rf cache
rm -rf build
rm -rf spflow.egg-info
python -m build