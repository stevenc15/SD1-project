docker build -t ubuntu-gpu:latest .
docker run --gpus all -d -v .:/workspace ubuntu-gpu:latest