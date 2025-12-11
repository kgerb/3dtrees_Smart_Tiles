#!/bin/bash
# Main orchestrator script for retiling and multi-stage subsampling
#
# Usage: main.sh <input_dir> <output_dir> [tile_length] [tile_buffer] [threads] [num_threads]
#   input_dir: Input directory with LAZ files
#   output_dir: Output directory for all stages
#   tile_length: Tile size in meters (default: 100)
#   tile_buffer: Buffer size in meters (default: 5)
#   threads: Threads per COPC writer (default: 5)
#   num_threads: Number of parallel threads for subsampling (default: 4)

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_dir> <output_dir> [tile_length] [tile_buffer] [threads] [num_threads]"
    echo "  input_dir: Input directory with LAZ files"
    echo "  output_dir: Output directory for all stages"
    echo "  tile_length: Tile size in meters (default: 100)"
    echo "  tile_buffer: Buffer size in meters (default: 5)"
    echo "  threads: Threads per COPC writer (default: 5)"
    echo "  num_threads: Number of parallel threads for subsampling (default: 4)"
    exit 1
fi

input_dir="$1"
output_dir="$2"
tile_length="${3:-100}"
tile_buffer="${4:-5}"
threads="${5:-5}"
num_threads="${6:-4}"

# Stage directories
tiles_dir="${output_dir}/tiles_${tile_length}m"
subsampled_2cm_dir="${tiles_dir}/subsampled_2cm"
subsampled_10cm_dir="${tiles_dir}/subsampled_10cm"

echo "=========================================="
echo "Main Orchestrator: Retiling + Subsample Pipeline"
echo "=========================================="
echo "Input directory: $input_dir"
echo "Output directory: $output_dir"
echo "Tile length: ${tile_length}m"
echo "Tile buffer: ${tile_buffer}m"
echo "Threads: $threads (retiling), $num_threads (subsampling)"
echo ""

# Step 1: Retiling
echo "=========================================="
echo "Step 1: Retiling"
echo "=========================================="
echo "Running retiling.sh..."
if ! bash "${SCRIPT_DIR}/retiling.sh" "$input_dir" "$output_dir" "$tile_length" "$tile_buffer" "$threads"; then
    echo "ERROR: Retiling failed!"
    exit 1
fi

echo ""
echo "✓ Retiling completed. Tiles saved to: ${tiles_dir}"
echo ""


# Step 2: Subsample to 2cm
echo "=========================================="
echo "Step 2: Subsample to 2cm (0.02m)"
echo "=========================================="
echo "Subsampling retiled COPC tiles to 2cm resolution..."

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

# Create output directory for 2cm subsampling
mkdir -p "$subsampled_2cm_dir"

# Run subsampling on each tile file
# Note: subsampling.sh processes all files in a directory
# We'll copy/link tiles to a temp directory or process directly
if ! bash "${SCRIPT_DIR}/subsampling.sh" "$tiles_dir" "0.02" "$num_threads"; then
    echo "ERROR: 2cm subsampling failed!"
    exit 1
fi

# Move subsampled files to our output directory
# subsampling.sh saves to <input_dir>/subsampled/
if [ -d "${tiles_dir}/subsampled" ]; then
    echo "Moving 2cm subsampled files..."
    mv "${tiles_dir}/subsampled"/* "$subsampled_2cm_dir/" 2>/dev/null || true
    rmdir "${tiles_dir}/subsampled" 2>/dev/null || true
fi

echo ""
echo "✓ 2cm subsampling completed. Files saved to: ${subsampled_2cm_dir}"
echo ""

# Step 3: Subsample to 10cm
echo "=========================================="
echo "Step 3: Subsample to 10cm (0.1m)"
echo "=========================================="
echo "Subsampling 2cm files to 10cm resolution..."

# Check if 2cm subsampled directory has files
subsampled_2cm_count=$(find "$subsampled_2cm_dir" -name "*.laz" 2>/dev/null | wc -l)
if [ "$subsampled_2cm_count" -eq 0 ]; then
    echo "ERROR: No 2cm subsampled files found in: $subsampled_2cm_dir"
    exit 1
fi

echo "Found $subsampled_2cm_count files to subsample to 10cm"

# Create output directory for 10cm subsampling
mkdir -p "$subsampled_10cm_dir"

# Run subsampling on 2cm files
if ! bash "${SCRIPT_DIR}/subsampling.sh" "$subsampled_2cm_dir" "0.1" "$num_threads"; then
    echo "ERROR: 10cm subsampling failed!"
    exit 1
fi

# Move subsampled files to our output directory
if [ -d "${subsampled_2cm_dir}/subsampled" ]; then
    echo "Moving 10cm subsampled files..."
    mv "${subsampled_2cm_dir}/subsampled"/* "$subsampled_10cm_dir/" 2>/dev/null || true
    rmdir "${subsampled_2cm_dir}/subsampled" 2>/dev/null || true
fi

echo ""
echo "✓ 10cm subsampling completed. Files saved to: ${subsampled_10cm_dir}"
echo ""



echo ""
echo "✓ Segmentation started"
echo ""

bash "${SCRIPT_DIR}/../3dTrees_SAT/run_docker_locally.sh" "$subsampled_10cm_dir" "$(dirname $subsampled_10cm_dir)/segmented" 1



## step 4 Remapping
echo "=========================================="
echo "Step 4: Remapping"
echo "=========================================="
echo "Remapping segmented files back to original resolution..."

# Define segmented directory (same level as subsampled_10cm_dir)
segmented_dir="$(dirname "$subsampled_10cm_dir")/segmented"

# Check if segmented directory exists
if [ ! -d "$segmented_dir" ]; then
    echo "ERROR: Segmented directory not found: $segmented_dir"
    exit 1
fi

# Find all segmented files (either *_segmented.laz or *_results/segmented_pc.laz)
echo "Finding segmented files..."
segmented_files=()

# Find direct segmented files (*_segmented.laz)
while IFS= read -r -d '' file; do
    segmented_files+=("$file")
done < <(find "$segmented_dir" -maxdepth 1 -name "*_segmented.laz" -type f -print0 2>/dev/null)

# Find segmented files in results directories (*_results/segmented_pc.laz)
while IFS= read -r -d '' file; do
    segmented_files+=("$file")
done < <(find "$segmented_dir" -type f -path "*/results/segmented_pc.laz" -print0 2>/dev/null)

if [ ${#segmented_files[@]} -eq 0 ]; then
    echo "ERROR: No segmented files found in: $segmented_dir"
    exit 1
fi

echo "Found ${#segmented_files[@]} segmented file(s) to remap"
echo ""

# Process each segmented file
remapped_count=0
failed_count=0

for segmented_file in "${segmented_files[@]}"; do
    # Extract base filename from segmented file
    segmented_basename=$(basename "$segmented_file")
    
    # Handle different naming patterns
    if [[ "$segmented_basename" == "segmented_pc.laz" ]]; then
        # File is in results directory, extract from parent directory name
        results_dir=$(dirname "$segmented_file")
        results_basename=$(basename "$results_dir")
        # Remove _results suffix to get base filename
        base_filename="${results_basename%_results}"
    else
        # File is *_segmented.laz, remove _segmented.laz suffix
        base_filename="${segmented_basename%_segmented.laz}"
    fi
    
    # Find corresponding original file in subsampled_2cm_dir
    # Segmented file name is based on 10cm file: c00_r00.copc_subsampled0.02m_subsampled0.1m
    # Original file is 2cm file: c00_r00.copc_subsampled0.02m.laz
    # Remove _subsampled0.1m suffix if present
    original_pattern="${base_filename%_subsampled0.1m}"
    # Also handle the case where it might be just the base name
    original_pattern="${original_pattern%_subsampled0.02m_subsampled0.1m}"
    
    # Look for matching file in subsampled_2cm_dir (try multiple patterns)
    original_file=""
    
    # Try exact match with .laz
    if [ -f "${subsampled_2cm_dir}/${original_pattern}.laz" ]; then
        original_file="${subsampled_2cm_dir}/${original_pattern}.laz"
    # Try exact match with .copc.laz
    elif [ -f "${subsampled_2cm_dir}/${original_pattern}.copc.laz" ]; then
        original_file="${subsampled_2cm_dir}/${original_pattern}.copc.laz"
    # Try pattern matching (remove everything after last _subsampled pattern)
    else
        # Extract the tile ID part (everything before _subsampled0.02m_subsampled0.1m)
        tile_id="${base_filename%%_subsampled*}"
        # Try to find files starting with tile_id
        original_file=$(find "$subsampled_2cm_dir" -maxdepth 1 -name "${tile_id}*.laz" -type f | head -1)
    fi
    
    if [ -z "$original_file" ] || [ ! -f "$original_file" ]; then
        echo "⚠️  WARNING: Could not find original file for: $base_filename"
        echo "   Segmented file: $segmented_file"
        echo "   Looking for pattern: ${original_pattern}*.laz in ${subsampled_2cm_dir}"
        failed_count=$((failed_count + 1))
        continue
    fi
    
    # Create output filename (remapped original resolution with attributes)
    output_file="${subsampled_2cm_dir}/${original_pattern}_remapped.laz"
    
    echo "Processing: $(basename "$base_filename")"
    echo "  Original: $(basename "$original_file")"
    echo "  Segmented: $(basename "$segmented_file")"
    echo "  Output: $(basename "$output_file")"
    
    # Run remapping script
    if python3 "${SCRIPT_DIR}/remapping_original_res.py" \
        --original_file "$original_file" \
        --subsampled_file "$segmented_file" \
        --output_file "$output_file"; then
        echo "  ✓ Remapping completed"
        remapped_count=$((remapped_count + 1))
    else
        echo "  ✗ Remapping failed"
        failed_count=$((failed_count + 1))
    fi
    echo ""
done

echo "=========================================="
echo "Remapping Summary:"
echo "  Successfully remapped: $remapped_count"
echo "  Failed: $failed_count"
echo "=========================================="
echo ""


# Summary
echo "=========================================="
echo "Pipeline Summary"
echo "=========================================="
echo "✓ Step 1: Retiling -> ${tiles_dir}"
echo "✓ Step 2: 2cm subsampling -> ${subsampled_2cm_dir}"
echo "✓ Step 3: 10cm subsampling -> ${subsampled_10cm_dir}"
echo ""
echo "All stages completed successfully!"
echo "=========================================="