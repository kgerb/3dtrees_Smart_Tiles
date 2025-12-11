#!/bin/bash
# Subsample COPC/LAZ/LAS files using voxel centroid nearest neighbor
#
# Usage: subsampling.sh <input_directory> <resolution> [num_threads] [output_prefix]
#   input_directory: Path to directory containing COPC/LAZ/LAS files
#   resolution: Voxel resolution in meters (e.g., 0.1 for 10cm)
#   num_threads: Number of parallel threads (default: 4)
#   output_prefix: Optional prefix for output files (e.g., "ULS_test_100m")
#                  If provided, output files will be named: <prefix>_c##_r##_<resolution>cm.laz
#                  If not provided, uses legacy naming: <input_stem>_subsampled<resolution>m.laz
#
# Output: Saved to <input_directory>/subsampled/
# Note: For COPC files, uses readers.copc with bounds for efficient reading

set -euo pipefail

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_directory> <resolution> [num_threads] [output_prefix]"
    echo "  input_directory: Path to directory containing COPC/LAZ/LAS files"
    echo "  resolution: Voxel resolution in meters (e.g., 0.1 for 10cm)"
    echo "  num_threads: Number of parallel threads (default: 4)"
    echo "  output_prefix: Optional prefix for output files (e.g., 'ULS_test_100m')"
    exit 1
fi

INPUT_DIRECTORY="$1"
SUBSAMPLING_RESOLUTION="$2"
NUMBER_OF_THREADS="${3:-4}"
OUTPUT_PREFIX="${4:-}"

# Validate input directory
if [ ! -d "$INPUT_DIRECTORY" ]; then
    echo "ERROR: Input directory not found: $INPUT_DIRECTORY"
    exit 1
fi

# Function to get COPC node count for given bounds
get_copc_node_count() {
    local COPC_FILE="$1"
    local BOUNDS="$2"
    
    # Check if file exists and is readable
    if [ ! -r "$COPC_FILE" ]; then
        echo "N/A"
        return
    fi
    
    # Try to detect if file is COPC by attempting to read with copc reader
    # COPC files can have .laz extension, so we'll try for all .laz files
    if [[ ! "$COPC_FILE" =~ \.(copc\.laz|laz)$ ]]; then
        echo "N/A"
        return
    fi
    
    # Method 1: Try using Python copclib library (most reliable)
    if command -v python3 &> /dev/null; then
        # Test if copclib is available
        if python3 -c "import copclib" 2>/dev/null; then
            local PYTHON_SCRIPT=$(cat <<'PYEOF'
import sys
import re
import copclib

copc_file = sys.argv[1]
bounds_str = sys.argv[2]

# Parse bounds: ([minx,maxx],[miny,maxy])
match = re.match(r'\(\[([\d.-]+),([\d.-]+)\],\[([\d.-]+),([\d.-]+)\]\)', bounds_str)
if not match:
    sys.exit(1)

minx, maxx, miny, maxy = map(float, match.groups())

try:
    reader = copclib.CopcReader(copc_file)
    hierarchy = reader.GetHierarchy()
    
    node_count = 0
    for node in hierarchy.GetAllNodes():
        try:
            bounds = node.GetBounds()
            # Check if node bounds intersect with query bounds
            if not (bounds.max_x < minx or bounds.min_x > maxx or 
                    bounds.max_y < miny or bounds.min_y > maxy):
                node_count += 1
        except:
            continue
    
    if node_count > 0:
        print(node_count)
        sys.exit(0)
except Exception as e:
    sys.exit(1)
PYEOF
)
            local NODE_COUNT=$(python3 -c "$PYTHON_SCRIPT" "$COPC_FILE" "$BOUNDS" 2>/dev/null)
            if [ -n "$NODE_COUNT" ] && [ "$NODE_COUNT" != "N/A" ] && [ "$NODE_COUNT" -gt 0 ] 2>/dev/null; then
                echo "$NODE_COUNT"
                return
            fi
        fi
    fi
    
    # Method 2: Try using copc command-line utility if available
    if command -v copc &> /dev/null; then
        local INFO_OUTPUT=$(copc info "$COPC_FILE" --bounds "$BOUNDS" 2>/dev/null || \
                          copc query "$COPC_FILE" --bounds "$BOUNDS" 2>/dev/null)
        local NODE_COUNT=$(echo "$INFO_OUTPUT" | grep -iE '(node|tile).*[0-9]+|[0-9]+.*(node|tile)' | \
                          grep -oP '\d+' | head -1)
        if [ -n "$NODE_COUNT" ] && [ "$NODE_COUNT" -gt 0 ] 2>/dev/null; then
            echo "$NODE_COUNT"
            return
        fi
    fi
    
    # Method 3: Try to use PDAL info with COPC reader and parse verbose output
    # PDAL might log node information when reading COPC files
    local TEMP_JSON=$(mktemp)
    local TEMP_LOG=$(mktemp)
    cat > "$TEMP_JSON" <<EOF
{
  "pipeline": [
    {
      "type": "readers.copc",
      "filename": "$COPC_FILE",
      "bounds": "$BOUNDS"
    }
  ]
}
EOF
    
    # Run PDAL info with verbose output to capture any node information
    # Redirect both stdout and stderr to capture all output
    pdal info "$TEMP_JSON" > "$TEMP_LOG" 2>&1
    
    # Try to extract node count from various possible output formats
    # Look for patterns like "nodes: 42", "42 nodes", "node_count: 42", etc.
    local NODE_COUNT=$(grep -iE '(node|tile).*[0-9]+|[0-9]+.*(node|tile)' "$TEMP_LOG" | \
                      grep -oP '\b[0-9]+\b' | head -1)
    
    # Also check metadata
    local METADATA=$(pdal info --metadata "$TEMP_JSON" 2>/dev/null)
    if [ -z "$NODE_COUNT" ]; then
        # Try to find node count in metadata JSON
        NODE_COUNT=$(echo "$METADATA" | grep -oP '"(node|tile).*":\s*\K\d+' | head -1)
    fi
    
    # Also try parsing the full JSON output for any node-related fields
    if [ -z "$NODE_COUNT" ]; then
        local FULL_OUTPUT=$(pdal info "$TEMP_JSON" 2>/dev/null)
        NODE_COUNT=$(echo "$FULL_OUTPUT" | grep -oP '"(node_count|nodes_accessed|nodes_read|tile_count)":\s*\K\d+' | head -1)
    fi
    
    rm -f "$TEMP_JSON" "$TEMP_LOG" 2>/dev/null
    
    # Validate result - node count should be a reasonable positive number
    if [ -n "$NODE_COUNT" ] && [ "$NODE_COUNT" -gt 0 ] 2>/dev/null && [ "$NODE_COUNT" -lt 1000000 ] 2>/dev/null; then
        echo "$NODE_COUNT"
    else
        echo "N/A"
    fi
}

# Function to process a single file
process_file() {
    local ORIGINAL_FILE="$1"
    local SUBSAMPLING_RESOLUTION="$2"
    local NUMBER_OF_THREADS="$3"
    local OUTPUT_PREFIX="$4"

    # Get input file directory and base name
    local INPUT_DIR=$(dirname "$(realpath "$ORIGINAL_FILE")")
    local INPUT_BASENAME=$(basename "$ORIGINAL_FILE")
    local INPUT_STEM="${INPUT_BASENAME%.las}"
    INPUT_STEM="${INPUT_STEM%.laz}"
    INPUT_STEM="${INPUT_STEM%.copc.laz}"

    # Set output paths
    local OUTPUT_DIR="${INPUT_DIR}/subsampled"
    local CHUNKS_DIR="${OUTPUT_DIR}/chunks"
    
    # Generate output filename
    if [ -n "$OUTPUT_PREFIX" ]; then
        # Extract c##_r## pattern from input filename
        local TILE_ID=""
        if [[ "$INPUT_STEM" =~ (c[0-9]+_r[0-9]+) ]]; then
            TILE_ID="${BASH_REMATCH[1]}"
        else
            # Fallback to input stem if pattern not found
            TILE_ID="$INPUT_STEM"
        fi
        # Convert resolution to cm (e.g., 0.02 -> 2, 0.1 -> 10)
        local RES_CM=$(awk "BEGIN {printf \"%.0f\", $SUBSAMPLING_RESOLUTION * 100}")
        local SUBSAMPLED_FILE="${OUTPUT_DIR}/${OUTPUT_PREFIX}_${TILE_ID}_${RES_CM}cm.laz"
        local MERGE_LOG_NAME="${OUTPUT_PREFIX}_${TILE_ID}_${RES_CM}cm"
    else
        # Legacy naming
        local SUBSAMPLED_FILE="${OUTPUT_DIR}/${INPUT_STEM}_subsampled${SUBSAMPLING_RESOLUTION}m.laz"
        local MERGE_LOG_NAME="${INPUT_STEM}"
    fi

    # Create output directories
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$CHUNKS_DIR"

    echo "=== Subsample Point Cloud ==="
    echo "Input file: $ORIGINAL_FILE"
    echo "Resolution: ${SUBSAMPLING_RESOLUTION}m"
    echo "Threads: $NUMBER_OF_THREADS"
    echo "Output: $SUBSAMPLED_FILE"
    
    # Check if COPC node counting is available
    if [[ "$ORIGINAL_FILE" =~ \.(copc\.laz|laz)$ ]]; then
        if command -v python3 &> /dev/null && python3 -c "import copclib" 2>/dev/null; then
            echo "COPC node counting: Available"
        else
            echo "COPC node counting: Not available (install: pip install copclib)"
        fi
    fi
    echo ""

    # Get the bounds of the input file
    echo "Getting spatial bounds of input file..."
    local METADATA=$(pdal info --metadata "$ORIGINAL_FILE" 2>/dev/null)
    local MINX=$(echo "$METADATA" | grep -oP '"minx":\s*\K[0-9.-]+' | head -1)
    local MAXX=$(echo "$METADATA" | grep -oP '"maxx":\s*\K[0-9.-]+' | head -1)
    local MINY=$(echo "$METADATA" | grep -oP '"miny":\s*\K[0-9.-]+' | head -1)
    local MAXY=$(echo "$METADATA" | grep -oP '"maxy":\s*\K[0-9.-]+' | head -1)

    echo "Bounds: minx=$MINX, maxx=$MAXX, miny=$MINY, maxy=$MAXY"

    # Validate bounds extraction
    if [ -z "$MINX" ] || [ -z "$MAXX" ] || [ -z "$MINY" ] || [ -z "$MAXY" ]; then
        echo "ERROR: Failed to extract bounds from file. Check if file is valid."
        return 1
    fi

    # Calculate chunk size based on number of threads
    # We'll split along the X axis for simplicity
    local X_RANGE=$(echo "$MAXX - $MINX" | bc)
    local CHUNK_SIZE=$(echo "$X_RANGE / $NUMBER_OF_THREADS" | bc -l)

    # Align chunk size to voxel grid to avoid boundary issues
    # Round chunk size up to nearest multiple of voxel size
    local CHUNK_SIZE_ALIGNED=$(echo "scale=10; tmp = $CHUNK_SIZE / $SUBSAMPLING_RESOLUTION; scale=0; (tmp + 0.999999)/1 * $SUBSAMPLING_RESOLUTION" | bc)

    echo "Raw chunk size: $CHUNK_SIZE, Aligned to voxel grid: $CHUNK_SIZE_ALIGNED (voxel size: $SUBSAMPLING_RESOLUTION)"
    echo "Splitting into $NUMBER_OF_THREADS chunks along X axis"

    # Align MINX to voxel grid as well for consistent grid origin (floor operation)
    local MINX_ALIGNED=$(echo "scale=10; tmp = $MINX / $SUBSAMPLING_RESOLUTION; scale=0; if (tmp < 0) tmp = (tmp - 0.999999)/1 else tmp = tmp/1; tmp * $SUBSAMPLING_RESOLUTION" | bc)

    # Create and process chunks in parallel
    local pids=()
    local chunk_bounds=()
    for i in $(seq 0 $((NUMBER_OF_THREADS - 1))); do
        local CHUNK_MINX=$(echo "$MINX_ALIGNED + ($i * $CHUNK_SIZE_ALIGNED)" | bc -l)
        local CHUNK_MAXX=$(echo "$MINX_ALIGNED + (($i + 1) * $CHUNK_SIZE_ALIGNED)" | bc -l)
        
        # For the last chunk, ensure we capture everything up to MAXX (and slightly beyond to be safe)
        if [ $i -eq $((NUMBER_OF_THREADS - 1)) ]; then
            CHUNK_MAXX=$(echo "$MAXX + $SUBSAMPLING_RESOLUTION" | bc -l)
        fi
        
        local CHUNK_BOUNDS="([$CHUNK_MINX,$CHUNK_MAXX],[$MINY,$MAXY])"
        chunk_bounds+=("$CHUNK_BOUNDS")
        
        local CHUNK_FILE="${CHUNKS_DIR}/chunk_${i}.laz"
        
        # Get node count for this chunk (for COPC files)
        # Note: This may take a moment for COPC files
        local NODE_COUNT=$(get_copc_node_count "$ORIGINAL_FILE" "$CHUNK_BOUNDS" 2>/dev/null)
        if [ -n "$NODE_COUNT" ] && [ "$NODE_COUNT" != "N/A" ] && [ "$NODE_COUNT" != "0" ]; then
            echo "Processing chunk $i: x range [$CHUNK_MINX, $CHUNK_MAXX], COPC nodes: $NODE_COUNT"
        else
            echo "Processing chunk $i: x range [$CHUNK_MINX, $CHUNK_MAXX]"
        fi
        
        # Process chunk in background
        # Use readers.copc with bounds for efficient COPC reading
        (
            local CHUNK_PIPELINE="${CHUNKS_DIR}/chunk_${i}_pipeline.json"
            
            # Determine if file is COPC
            local is_copc=false
            if [[ "$ORIGINAL_FILE" =~ \.(copc\.laz)$ ]]; then
                is_copc=true
            fi
            
            if [ "$is_copc" = true ]; then
                cat > "$CHUNK_PIPELINE" <<EOF
[
    {
        "type": "readers.copc",
        "filename": "$ORIGINAL_FILE",
        "bounds": "${chunk_bounds[$i]}"
    },
    {
        "type": "filters.voxelcentroidnearestneighbor",
        "cell": $SUBSAMPLING_RESOLUTION
    },
    {
        "type": "writers.las",
        "filename": "$CHUNK_FILE",
        "compression": true
    }
]
EOF
            else
                cat > "$CHUNK_PIPELINE" <<EOF
[
    {
        "type": "readers.las",
        "filename": "$ORIGINAL_FILE"
    },
    {
        "type": "filters.crop",
        "bounds": "${chunk_bounds[$i]}"
    },
    {
        "type": "filters.voxelcentroidnearestneighbor",
        "cell": $SUBSAMPLING_RESOLUTION
    },
    {
        "type": "writers.las",
        "filename": "$CHUNK_FILE",
        "compression": true
    }
]
EOF
            fi
            
            if pdal pipeline "$CHUNK_PIPELINE" > "${CHUNKS_DIR}/chunk_${i}.log" 2>&1; then
                rm -f "$CHUNK_PIPELINE"
            else
                rm -f "$CHUNK_PIPELINE" "$CHUNK_FILE"
            fi
        ) &
        
        pids+=($!)
    done

    # Wait for all parallel processes to complete with progress tracking
    echo "Waiting for all $NUMBER_OF_THREADS chunks to complete processing..."
    local completed=0
    local total=$NUMBER_OF_THREADS

    for i in "${!pids[@]}"; do
        local pid=${pids[$i]}
        wait $pid
        local exit_code=$?
        
        if [ $exit_code -ne 0 ]; then
            echo "Error: Chunk $i processing failed (PID $pid, exit code: $exit_code)"
            return 1
        fi
        
        completed=$((completed + 1))
        local percentage=$((completed * 100 / total))
        echo "✓ Chunk $i completed ($completed/$total, ${percentage}%)"
    done

    echo "All chunks completed successfully!"

    echo ""
    echo "Chunk processing summary:"
    echo "========================="
    local total_points=0
    for i in $(seq 0 $((NUMBER_OF_THREADS - 1))); do
        local CHUNK_FILE="${CHUNKS_DIR}/chunk_${i}.laz"
        if [ -f "$CHUNK_FILE" ]; then
            local size=$(stat -c%s "$CHUNK_FILE" | awk '{printf "%.2f MB", $1/1024/1024}')
            local points=$(pdal info --metadata "$CHUNK_FILE" 2>/dev/null | grep -m1 '"count"' | grep -oP '\d+')
            
            # Get node count for this chunk
            local NODE_COUNT=$(get_copc_node_count "$ORIGINAL_FILE" "${chunk_bounds[$i]}")
            
            if [ -n "$points" ]; then
                total_points=$((total_points + points))
                if [ "$NODE_COUNT" != "N/A" ]; then
                    echo "  Chunk $i: $size, $points points, $NODE_COUNT COPC nodes"
                else
                    echo "  Chunk $i: $size, $points points"
                fi
            else
                if [ "$NODE_COUNT" != "N/A" ]; then
                    echo "  Chunk $i: $size, $NODE_COUNT COPC nodes"
                else
                    echo "  Chunk $i: $size"
                fi
            fi
        fi
    done
    echo "  Total points: $total_points"
    echo "========================="
    echo ""

    echo "Merging chunks back together..."

    # Merge all chunks back together
    local CHUNK_FILES=("${CHUNKS_DIR}"/chunk_*.laz)
    
    # Check if any chunk files exist
    if [ ${#CHUNK_FILES[@]} -eq 0 ] || [ ! -f "${CHUNK_FILES[0]}" ]; then
        echo "ERROR: No chunk files found to merge!"
        return 1
    fi

    # Merge chunks
    if pdal_wrench merge --output="$SUBSAMPLED_FILE" "${CHUNK_FILES[@]}" > "${OUTPUT_DIR}/${MERGE_LOG_NAME}_merge.log" 2>&1; then
        # Verify output file
        if [ ! -f "$SUBSAMPLED_FILE" ] || [ ! -s "$SUBSAMPLED_FILE" ]; then
            echo "ERROR: Output file was not created or is empty"
            rm -rf "$CHUNKS_DIR"
            return 1
        fi
        
        # Get output file info
        local OUTPUT_SIZE=$(stat -c%s "$SUBSAMPLED_FILE" | awk '{printf "%.2f MB", $1/1024/1024}')
        local OUTPUT_POINTS=$(pdal info --metadata "$SUBSAMPLED_FILE" 2>/dev/null | grep -m1 '"count"' | grep -oP '\d+' || echo "unknown")

        # Clean up chunk files
        echo "Cleaning up temporary chunk files..."
        rm -rf "$CHUNKS_DIR"
    else
        echo "ERROR: Failed to merge chunks"
        cat "${OUTPUT_DIR}/${MERGE_LOG_NAME}_merge.log" 2>/dev/null
        rm -rf "$CHUNKS_DIR"
        return 1
    fi

    echo ""
    echo "Subsampling completed!"
    echo "Output file: $SUBSAMPLED_FILE"
    echo "Output size: $OUTPUT_SIZE"
    echo "Output points: $OUTPUT_POINTS"
    echo ""
}

# Validate resolution
if ! echo "$SUBSAMPLING_RESOLUTION > 0" | bc -l > /dev/null 2>&1; then
    echo "ERROR: Resolution must be a positive number"
    exit 1
fi

# Get absolute path to input directory
INPUT_DIRECTORY=$(realpath "$INPUT_DIRECTORY")

# Find all COPC/LAZ/LAS files in the directory
FILES=()
while IFS= read -r -d '' file; do
    FILES+=("$file")
done < <(find "$INPUT_DIRECTORY" -maxdepth 1 -type f \( -name "*.copc.laz" -o -name "*.laz" -o -name "*.las" \) -print0 | sort -z)

# Check if any files were found
if [ ${#FILES[@]} -eq 0 ]; then
    echo "ERROR: No COPC/LAZ/LAS files found in directory: $INPUT_DIRECTORY"
    exit 1
fi

echo "Found ${#FILES[@]} file(s) to process"
echo "======================================"
echo ""

# Process each file
file_count=0
for ORIGINAL_FILE in "${FILES[@]}"; do
    file_count=$((file_count + 1))
    echo "Processing file $file_count of ${#FILES[@]}: $(basename "$ORIGINAL_FILE")"
    echo "======================================"
    
    if ! process_file "$ORIGINAL_FILE" "$SUBSAMPLING_RESOLUTION" "$NUMBER_OF_THREADS" "$OUTPUT_PREFIX"; then
        echo "ERROR: Failed to process file: $ORIGINAL_FILE"
        echo "Continuing with next file..."
        echo ""
        continue
    fi
    
    echo ""
done

echo "======================================"
echo "All files processed!"
echo "Processed $file_count file(s) from: $INPUT_DIRECTORY"

