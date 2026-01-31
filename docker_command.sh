docker run --rm -it \
  --gpus '"device=3"' \
  -v "$PWD":/workspace -w /workspace \
  hayoung_flowdps