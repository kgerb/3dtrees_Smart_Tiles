#!/bin/bash
# Check if input folder parameter is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <subsampled_10cm_folder>"
    echo "Example: $0 /home/kg281/data/kaltenborn25_lidar/bechstedt/tiles_200m/subsampled_10cm"
    exit 1
fi

# Input folder from parameter
SUBSAMPLED_10CM_FOLDER="$1"

# Derive other folder paths from input folder
SUBSAMPLED_2CM_FOLDER="${SUBSAMPLED_10CM_FOLDER/subsampled_10cm/subsampled_2cm}"
OUTPUT_FOLDER="${SUBSAMPLED_10CM_FOLDER/subsampled_10cm/segmented_remapped}"

# Create output folder if it doesn't exist
mkdir -p $OUTPUT_FOLDER

# Get script directory to find remapping_original_res.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find all *_results directories in subsampled_10cm folder
for RESULTS_DIR in $SUBSAMPLED_10CM_FOLDER/*_results; do
    # Check if directory exists (glob might return literal if no matches)
    if [ ! -d "$RESULTS_DIR" ]; then
        continue
    fi
    
    # Get the results folder name (e.g., c00_r00.copc_subsampled0.02m_subsampled0.1m_results)
    RESULTS_DIRNAME=$(basename "$RESULTS_DIR")
    
    # Extract tile ID (first 7 characters: c##_r##)
    TILE_ID=${RESULTS_DIRNAME:0:7}
    
    # Path to segmented_pc.laz in results folder
    SEGMENTED_FILE="$RESULTS_DIR/segmented_pc.laz"
    
    # Path to corresponding 2cm original file
    ORIGINAL_FILE="$SUBSAMPLED_2CM_FOLDER/${TILE_ID}.copc_subsampled0.02m.laz"
    
    # Output file path
    OUTPUT_FILE="$OUTPUT_FOLDER/${TILE_ID}_segmented_remapped.laz"
    
    # Validate that required files exist
    if [ ! -f "$SEGMENTED_FILE" ]; then
        echo "Warning: Segmented file not found: $SEGMENTED_FILE"
        continue
    fi
    
    if [ ! -f "$ORIGINAL_FILE" ]; then
        echo "Warning: Original 2cm file not found: $ORIGINAL_FILE"
        continue
    fi
    
    echo "Processing tile: $TILE_ID"
    echo "  Segmented file: $SEGMENTED_FILE"
    echo "  Original file: $ORIGINAL_FILE"
    echo "  Output file: $OUTPUT_FILE"
    
    # Remap the file
    python "$SCRIPT_DIR/remapping_original_res.py" \
        --subsampled_file "$SEGMENTED_FILE" \
        --original_file "$ORIGINAL_FILE" \
        --output_file "$OUTPUT_FILE"
    
    echo ""

done
