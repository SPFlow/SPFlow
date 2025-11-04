set -e

if [ ! -d "dist" ]; then
    echo "Error: dist/ directory not found. Run create_pip_dist.sh first."
    exit 1
fi

if [ -z "$(ls -A dist/)" ]; then
    echo "Error: dist/ directory is empty. Run create_pip_dist.sh first."
    exit 1
fi

echo "Uploading distribution packages..."
twine upload dist/*