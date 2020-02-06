set -e

echo "Testing"
export PYTHONPATH=.
find spn/tests/test_*.py -print0 | xargs -n 1 -0 python3 -m pytest

