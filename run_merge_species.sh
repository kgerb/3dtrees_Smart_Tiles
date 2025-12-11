#!/bin/bash
# Merge LAZ tiles with species predictions
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_INPUT="/home/kg281/data/kaltenborn25_lidar/bechstedt/tiles_200m/detailview_predictions"
DEFAULT_OUTPUT_LAZ="/home/kg281/data/kaltenborn25_lidar/bechstedt/merged_detailview.laz"
DEFAULT_OUTPUT_TILES="/home/kg281/data/kaltenborn25_lidar/bechstedt/tiles_200m/detailview_merged"
DEFAULT_SOURCE_FOLDER="/home/kg281/data/kaltenborn25_lidar/bechstedt/input"
DEFAULT_OUTPUT_WITH_PREDICTIONS="/home/kg281/data/kaltenborn25_lidar/bechstedt/input_with_predictions"

INPUT_FOLDER="${1:-$DEFAULT_INPUT}"
OUTPUT_LAZ="${2:-$DEFAULT_OUTPUT_LAZ}"
OUTPUT_TILES="${3:-$DEFAULT_OUTPUT_TILES}"
NUM_THREADS="${4:-8}"
INPUT_SOURCE_FOLDER="${5:-$DEFAULT_SOURCE_FOLDER}"
OUTPUT_WITH_PREDICTIONS="${6:-$DEFAULT_OUTPUT_WITH_PREDICTIONS}"

echo "Input:   $INPUT_FOLDER"
echo "Output:  $OUTPUT_LAZ"
echo "Tiles:   $OUTPUT_TILES"
echo "Threads: $NUM_THREADS"
echo "Source:  $INPUT_SOURCE_FOLDER"
echo "Remapped: $OUTPUT_WITH_PREDICTIONS"

python "$SCRIPT_DIR/merge_species_tiles.py" \
    --input_folder "$INPUT_FOLDER" \
    --output_merged_laz "$OUTPUT_LAZ" \
    --output_tiles_folder "$OUTPUT_TILES" \
    --buffer 0.2 \
    --min_cluster_size 300 \
    --num_threads "$NUM_THREADS"

echo ""
echo "Remapping predictions to original input files..."

"$SCRIPT_DIR/run_remap_to_source.sh" \
    "$OUTPUT_LAZ" \
    "$INPUT_SOURCE_FOLDER" \
    "$OUTPUT_WITH_PREDICTIONS" \
    "$NUM_THREADS"

