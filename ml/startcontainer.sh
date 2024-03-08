docker build -t ubuntu-gpu:latest ./ml
docker run --gpus all -d -v .:/workspace ubuntu-gpu:latest