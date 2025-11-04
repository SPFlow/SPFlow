set -e

echo "Testing"
export PYTHONPATH=.
pytest spn/tests/
