#!/bin/bash
# Main orchestrator script for retiling and multi-stage subsampling
#
# Usage: main.sh <input_dir> <output_dir> <first_resolution> [tile_length] [tile_buffer] [threads] [num_threads] [second_resolution]
#   input_dir: Input directory with LAZ files
#   output_dir: Output directory for all stages
#   first_resolution: First resolution in meters (e.g., 0.02 for 2cm) - REQUIRED
#   tile_length: Tile size in meters (default: 100)
#   tile_buffer: Buffer size in meters (default: 5)
#   threads: Threads per COPC writer (default: 5)
#   num_threads: Number of parallel threads for subsampling (default: 4)
#   second_resolution: Optional second resolution in meters (e.g., 0.2 for 20cm) (default: empty, skips step)

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
if [ $# -lt 3 ]; then
    echo "Usage: $0 <input_dir> <output_dir> <first_resolution> [tile_length] [tile_buffer] [threads] [num_threads] [second_resolution]"
    echo "  input_dir: Input directory with LAZ files"
    echo "  output_dir: Output directory for all stages"
    echo "  first_resolution: First resolution in meters (e.g., 0.02 for 2cm) - REQUIRED"
    echo "  tile_length: Tile size in meters (default: 100)"
    echo "  tile_buffer: Buffer size in meters (default: 5)"
    echo "  threads: Threads per COPC writer (default: 5)"
    echo "  num_threads: Number of parallel threads for subsampling (default: 4)"
    echo "  second_resolution: Optional second resolution in meters (e.g., 0.2 for 20cm) (default: empty, skips step)"
    exit 1
fi

input_dir="$1"
output_dir="$2"
first_resolution="$3"
tile_length="${4:-100}"
tile_buffer="${5:-5}"
threads="${6:-5}"
num_threads="${7:-4}"
second_resolution="${8:-}"

# Convert resolutions to directory names (e.g., 0.02 -> 2cm, 0.1 -> 10cm)
first_res_cm=$(awk "BEGIN {printf \"%.0f\", $first_resolution * 100}")
second_res_cm=""
if [ -n "$second_resolution" ]; then
    second_res_cm=$(awk "BEGIN {printf \"%.0f\", $second_resolution * 100}")
fi

# Stage directories
tiles_dir="${output_dir}/tiles_${tile_length}m"
subsampled_first_dir="${tiles_dir}/subsampled_${first_res_cm}cm"
subsampled_10cm_dir="${tiles_dir}/subsampled_10cm"
# Second resolution directory (only created if second_resolution is provided)
if [ -n "$second_resolution" ]; then
    subsampled_second_dir="${tiles_dir}/subsampled_${second_res_cm}cm"
fi

# Output prefix for subsampled files (e.g., "ULS_test_100m")
output_prefix="$(basename "$output_dir")_${tile_length}m"

echo "=========================================="
echo "Main Orchestrator: Retiling + Subsample Pipeline"
echo "=========================================="
echo "Input directory: $input_dir"
echo "Output directory: $output_dir"
echo "First resolution: ${first_resolution}m (${first_res_cm}cm)"
echo "Tile length: ${tile_length}m"
echo "Tile buffer: ${tile_buffer}m"
echo "Threads: $threads (retiling), $num_threads (subsampling)"
if [ -n "$second_resolution" ]; then
    echo "Second resolution: ${second_resolution}m (${second_res_cm}cm) - optional step enabled"
else
    echo "Second resolution: disabled (optional step skipped)"
fi
echo ""

# Step 1: Retiling
echo "=========================================="
echo "Step 1: Retiling"
echo "=========================================="
echo "Running retiling.sh..."
# Use 1m offset: start grid at min_x + 1m and min_y + 1m of input files
grid_offset="1.0"
if ! bash "${SCRIPT_DIR}/retiling.sh" "$input_dir" "$output_dir" "$tile_length" "$tile_buffer" "$threads" "$grid_offset"; then
    echo "ERROR: Retiling failed!"
    exit 1
fi

echo ""
echo "✓ Retiling completed. Tiles saved to: ${tiles_dir}"
echo ""


# Step 2: Subsample to first resolution
echo "=========================================="
echo "Step 2: Subsample to ${first_res_cm}cm (${first_resolution}m)"
echo "=========================================="
echo "Subsampling retiled COPC tiles to ${first_resolution}m resolution..."

# Check if tiles directory exists and has files
if [ ! -d "$tiles_dir" ]; then
    echo "ERROR: Tiles directory not found: $tiles_dir"
    exit 1
fi

tile_count=$(find "$tiles_dir" -name "*.copc.laz" -o -name "*.laz" 2>/dev/null | wc -l)
if [ "$tile_count" -eq 0 ]; then
    echo "ERROR: No tile files found in: $tiles_dir"
    exit 1
fi

echo "Found $tile_count tile files to subsample"

# Create output directory for first resolution subsampling
mkdir -p "$subsampled_first_dir"

# Run subsampling on each tile file
# Note: subsampling.sh processes all files in a directory
# We'll copy/link tiles to a temp directory or process directly
if ! bash "${SCRIPT_DIR}/subsampling.sh" "$tiles_dir" "$first_resolution" "$num_threads" "$output_prefix"; then
    echo "ERROR: ${first_resolution}m subsampling failed!"
    exit 1
fi

# Move subsampled files to our output directory
# subsampling.sh saves to <input_dir>/subsampled/
if [ -d "${tiles_dir}/subsampled" ]; then
    echo "Moving ${first_res_cm}cm subsampled files..."
    mv "${tiles_dir}/subsampled"/* "$subsampled_first_dir/" 2>/dev/null || true
    rmdir "${tiles_dir}/subsampled" 2>/dev/null || true
fi

echo ""
echo "✓ ${first_res_cm}cm subsampling completed. Files saved to: ${subsampled_first_dir}"
echo ""

# Step 3: Subsample to 10cm
echo "=========================================="
echo "Step 3: Subsample to 10cm (0.1m)"
echo "=========================================="
echo "Subsampling ${first_res_cm}cm files to 10cm resolution..."

# Check if first resolution subsampled directory has files
subsampled_first_count=$(find "$subsampled_first_dir" -name "*.laz" 2>/dev/null | wc -l)
if [ "$subsampled_first_count" -eq 0 ]; then
    echo "ERROR: No ${first_res_cm}cm subsampled files found in: $subsampled_first_dir"
    exit 1
fi

echo "Found $subsampled_first_count files to subsample to 10cm"

# Create output directory for 10cm subsampling
mkdir -p "$subsampled_10cm_dir"

# Run subsampling on first resolution files
if ! bash "${SCRIPT_DIR}/subsampling.sh" "$subsampled_first_dir" "0.1" "$num_threads" "$output_prefix"; then
    echo "ERROR: 10cm subsampling failed!"
    exit 1
fi

# Move subsampled files to our output directory
if [ -d "${subsampled_first_dir}/subsampled" ]; then
    echo "Moving 10cm subsampled files..."
    mv "${subsampled_first_dir}/subsampled"/* "$subsampled_10cm_dir/" 2>/dev/null || true
    rmdir "${subsampled_first_dir}/subsampled" 2>/dev/null || true
fi

echo ""
echo "✓ 10cm subsampling completed. Files saved to: ${subsampled_10cm_dir}"
echo ""

# Step 4: Optional second resolution subsampling
if [ -n "$second_resolution" ]; then
    echo "=========================================="
    echo "Step 4: Subsample to ${second_res_cm}cm (${second_resolution}m)"
    echo "=========================================="
    echo "Subsampling 10cm files to ${second_resolution}m resolution..."
    
    # Check if 10cm subsampled directory has files
    subsampled_10cm_count=$(find "$subsampled_10cm_dir" -name "*.laz" 2>/dev/null | wc -l)
    if [ "$subsampled_10cm_count" -eq 0 ]; then
        echo "ERROR: No 10cm subsampled files found in: $subsampled_10cm_dir"
        exit 1
    fi
    
    echo "Found $subsampled_10cm_count files to subsample to ${second_resolution}m"
    
    # Create output directory for second resolution subsampling
    mkdir -p "$subsampled_second_dir"
    
    # Run subsampling on 10cm files
    if ! bash "${SCRIPT_DIR}/subsampling.sh" "$subsampled_10cm_dir" "$second_resolution" "$num_threads" "$output_prefix"; then
        echo "ERROR: ${second_resolution}m subsampling failed!"
        exit 1
    fi
    
    # Move subsampled files to our output directory
    if [ -d "${subsampled_10cm_dir}/subsampled" ]; then
        echo "Moving ${second_resolution}m subsampled files..."
        mv "${subsampled_10cm_dir}/subsampled"/* "$subsampled_second_dir/" 2>/dev/null || true
        rmdir "${subsampled_10cm_dir}/subsampled" 2>/dev/null || true
    fi
    
    echo ""
    echo "✓ ${second_resolution}m subsampling completed. Files saved to: ${subsampled_second_dir}"
    echo ""
fi

# Summary
echo "=========================================="
echo "Pipeline Summary"
echo "=========================================="
echo "✓ Step 1: Retiling -> ${tiles_dir}"
echo "✓ Step 2: ${first_res_cm}cm subsampling -> ${subsampled_first_dir}"
echo "✓ Step 3: 10cm subsampling -> ${subsampled_10cm_dir}"
if [ -n "$second_resolution" ]; then
    echo "✓ Step 4: ${second_res_cm}cm subsampling -> ${subsampled_second_dir}"
fi
echo ""
echo "All stages completed successfully!"
echo "=========================================="