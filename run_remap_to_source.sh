#!/bin/bash
# Remap merged predictions to original source point clouds
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default paths for bechstedt
DEFAULT_MERGED_LAZ="/home/kg281/data/kaltenborn25_lidar/bechstedt/merged_detailview.laz"
DEFAULT_SOURCE_FOLDER="/home/kg281/data/kaltenborn25_lidar/bechstedt/input"
DEFAULT_OUTPUT_FOLDER="/home/kg281/data/kaltenborn25_lidar/bechstedt/source_with_predictions"
DEFAULT_NUM_THREADS=8
DEFAULT_MAX_DISTANCE=0.5

MERGED_LAZ="${1:-$DEFAULT_MERGED_LAZ}"
SOURCE_FOLDER="${2:-$DEFAULT_SOURCE_FOLDER}"
OUTPUT_FOLDER="${3:-$DEFAULT_OUTPUT_FOLDER}"
NUM_THREADS="${4:-$DEFAULT_NUM_THREADS}"
MAX_DISTANCE="${5:-$DEFAULT_MAX_DISTANCE}"

echo "Merged LAZ:    $MERGED_LAZ"
echo "Source folder: $SOURCE_FOLDER"
echo "Output folder: $OUTPUT_FOLDER"
echo "Threads:       $NUM_THREADS"
echo "Max distance:  $MAX_DISTANCE"

# Find all source LAZ files (excluding already processed ones)
SOURCE_FILES=$(find "$SOURCE_FOLDER" -maxdepth 1 -name "*.laz" ! -name "*_with_predictions.laz" ! -name "merged_*.laz")

if [ -z "$SOURCE_FILES" ]; then
    echo "Error: No source LAZ files found in $SOURCE_FOLDER"
    exit 1
fi

echo "Source files found:"
for f in $SOURCE_FILES; do
    echo "  - $(basename $f)"
done

python "$SCRIPT_DIR/remap_to_source.py" \
    --merged_laz "$MERGED_LAZ" \
    --source_files $SOURCE_FILES \
    --output_folder "$OUTPUT_FOLDER" \
    --num_threads "$NUM_THREADS" \
    --max_distance "$MAX_DISTANCE"

