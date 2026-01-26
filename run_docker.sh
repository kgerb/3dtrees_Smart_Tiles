#!/bin/bash

INPUT_FOLDER=/home/kg281/data/ecosense_mini/
OUTPUT_FOLDER=/home/kg281/data/ecosense_mini/out

mkdir -p $OUTPUT_FOLDER

# Run the processing
docker run --rm -it \
  --cpuset-cpus="0-49" \
  --memory=200g \
  --runtime=nvidia \
  --gpus device=1 \
  --user $(id -u):$(id -g) \
  -v "$INPUT_FOLDER":/in \
  -v "$OUTPUT_FOLDER":/out \
  3dtrees_smart_tile \
  bash -c "python run.py --task tile --input_dir /in --output_dir /out --tile_length 200 --tile_buffer 20 --threads 10 --workers 2 --skip_dimension_reduction"