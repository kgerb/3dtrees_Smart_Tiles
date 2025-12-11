#!/bin/bash
# Retiling script using tindex and direct COPC readers with bounds filtering
# This approach ensures efficient spatial filtering at the COPC level
#
# Usage: retiling.sh [input_dir] [output_dir] [tile_length] [tile_buffer] [threads] [grid_offset] [parallel_conversion]
#   OR: retiling.sh --file-list <list_file> [output_dir] [tile_length] [tile_buffer] [threads] [grid_offset] [parallel_conversion]
#   input_dir: Input directory with LAZ files (default: /home/kg281/data/output/pdal_experiments/uls_copc_input)
#   --file-list: File containing list of LAS/LAZ file paths (one per line)
#   output_dir: Output directory (default: /home/kg281/data/output/pdal_experiments)
#   tile_length: Tile size in meters (default: 100)
#   tile_buffer: Buffer size in meters (default: 5)
#   threads: Threads per COPC writer (default: 5) - Note: not used with pdal wrench
#   grid_offset: Offset in meters to add to minx and miny before starting grid (default: 0.0)
#   parallel_conversion: Parallel processes for COPC conversion (default: 4, always used)

set -euo pipefail

# Parse arguments
USE_FILE_LIST=false
FILE_LIST=""
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --file-list)
            USE_FILE_LIST=true
            FILE_LIST="$2"
            shift 2
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Restore positional arguments
set -- "${POSITIONAL_ARGS[@]}"

tile_length=${3:-"100"}
tile_buffer=${4:-"5"}
# MAX_TILE_PROCS: How many tiles to process in parallel (process-level parallelism)
MAX_TILE_PROCS=${MAX_TILE_PROCS:-5}
# threads: Number of threads for PDAL COPC writing/compression (internal thread-level parallelism)
# Note: Not used with pdal wrench, but kept for compatibility
threads=${5:-5}
# grid_offset: Offset in meters to add to minx and miny before starting grid (default: 0.0)
grid_offset=${6:-"0.0"}
# parallel_conversion: Number of parallel processes for COPC conversion (always 4)
parallel_conversion=4

if [ "$USE_FILE_LIST" = true ]; then
    if [ ! -f "${FILE_LIST}" ]; then
        echo "ERROR: File list does not exist: ${FILE_LIST}"
        exit 1
    fi
    input_dir=""  # Not used when using file list
    echo "Using file list: ${FILE_LIST}"
else
    input_dir=${1:-"/home/kg281/data/output/pdal_experiments/uls_copc_input"}
fi

output_dir=${2:-"/home/kg281/data/output/pdal_experiments"}
tiles_dir="${output_dir}/tiles_${tile_length}m"
copc_dir="${output_dir}/uls_copc"
log_folder="${output_dir}/logs"
tindex_file="${output_dir}/ULS_tiles_${tile_length}m_tindex.gpkg"

build_tindex_script="$(dirname "$0")/build_tindex.sh"

mkdir -p "${output_dir}"
mkdir -p "${tiles_dir}"
mkdir -p "${log_folder}"
mkdir -p "${copc_dir}"

# Step 1: Convert to COPC if needed
copc_list_file="${output_dir}/copc_files.txt"

if [ ! -s "${copc_list_file}" ]; then
  if [ "$USE_FILE_LIST" = true ]; then
    echo "=== Step 1: Preparing COPC files from file list using pdal wrench ==="
    
    # Count input files (handle both .las and .laz)
    input_file_count=$(grep -E '\.(las|laz)$' "${FILE_LIST}" 2>/dev/null | wc -l)
    
    echo "  Found ${input_file_count} files. Using parallel conversion with ${parallel_conversion} processes"
    echo "  Using pdal wrench for conversion"
    
    # Create filtered file list (only .las and .laz files, exclude .copc.laz)
    file_list=$(mktemp)
    grep -E '\.(las|laz)$' "${FILE_LIST}" | grep -v '\.copc\.laz$' | sort > "${file_list}"
    total_files=$(wc -l < "${file_list}")
    files_per_process=$(( (total_files + parallel_conversion - 1) / parallel_conversion ))
    
    # Split files and run parallel conversions
    conv_pids=()
    
    for i in $(seq 0 $((parallel_conversion - 1))); do
      process_file_list="${file_list}_${i}"
      
      # Split files using split command or sed
      start_line=$((i * files_per_process + 1))
      end_line=$((start_line + files_per_process - 1))
      
      if [ $i -eq $((parallel_conversion - 1)) ]; then
        # Last process takes remaining files
        tail -n +${start_line} "${file_list}" > "${process_file_list}"
      else
        sed -n "${start_line},${end_line}p" "${file_list}" > "${process_file_list}"
      fi
      
      # Count files for this process
      process_file_count=$(wc -l < "${process_file_list}")
      if [ "$process_file_count" -eq 0 ]; then
        rm -f "${process_file_list}"
        continue
      fi
      
      # Run conversion for this subset in background
      (
        file_num=0
        echo "    [Process $((i+1))/${parallel_conversion}] Converting ${process_file_count} files..."
        while IFS= read -r input_file; do
          file_num=$((file_num + 1))
          # Remove any trailing whitespace
          input_file=$(echo "${input_file}" | xargs)
          
          if [ ! -f "${input_file}" ]; then
            echo "    [Process $((i+1))/${parallel_conversion}] ✗ File not found: ${input_file}"
            continue
          fi
          
          # Get base filename without extension
          base_name=$(basename "${input_file}")
          base_name="${base_name%.las}"
          base_name="${base_name%.laz}"
          
          # Output to flat structure in copc_dir
          out="${copc_dir}/${base_name}.copc.laz"
          
          # Use pdal wrench for conversion (prefer retile_env)
          PDAL_WRENCH="${PDAL_WRENCH:-pdal_wrench}"
          if [ -f "/home/kg281/anaconda3/envs/retile_env/bin/pdal_wrench" ]; then
            PDAL_WRENCH="/home/kg281/anaconda3/envs/retile_env/bin/pdal_wrench"
          fi
          if ${PDAL_WRENCH} translate --input="${input_file}" --output="${out}" \
            > "${log_folder}/copc_conv_${i}_${base_name}.log" 2>&1
          then
            if [ -f "$out" ] && [ -s "$out" ]; then
              echo "    [Process $((i+1))/${parallel_conversion}] ✓ Finished ${file_num}/${process_file_count}: ${base_name}"
            else
              echo "    [Process $((i+1))/${parallel_conversion}] ✗ Failed ${file_num}/${process_file_count}: ${base_name} - Output empty"
              rm -f "$out" 2>/dev/null
            fi
          else
            echo "    [Process $((i+1))/${parallel_conversion}] ✗ Failed ${file_num}/${process_file_count}: ${base_name} - Conversion error"
            # Show last few lines of error log
            if [ -f "${log_folder}/copc_conv_${i}_${base_name}.log" ]; then
              tail -5 "${log_folder}/copc_conv_${i}_${base_name}.log" 2>/dev/null | head -3
            fi
            rm -f "$out" 2>/dev/null
          fi
        done < "${process_file_list}"
        echo "    [Process $((i+1))/${parallel_conversion}] Completed all ${process_file_count} files"
      ) &
      conv_pids+=($!)
    done
    
    # Wait for all conversion processes
    echo "  Waiting for ${#conv_pids[@]} parallel conversion processes..."
    for pid in "${conv_pids[@]}"; do
      wait $pid
    done
    
    # Cleanup
    rm -f "${file_list}" "${file_list}"_*
    
    echo "  Parallel conversion completed"
  elif [ ! -d "${input_dir}" ]; then
    echo "WARNING: Input directory does not exist: ${input_dir}"
    echo "Skipping COPC conversion. Assuming COPC files already exist in ${copc_dir}"
  else
    echo "=== Step 1: Preparing COPC files using pdal wrench ==="
    
    # Count input files
    input_file_count=$(find "${input_dir}" -name "*.laz" 2>/dev/null | wc -l)
    
    echo "  Found ${input_file_count} files. Using parallel conversion with ${parallel_conversion} processes"
    echo "  Using pdal wrench for conversion"
    
    # Create file list
    file_list=$(mktemp)
    find "${input_dir}" -name "*.laz" 2>/dev/null | sort > "${file_list}"
    total_files=$(wc -l < "${file_list}")
    files_per_process=$(( (total_files + parallel_conversion - 1) / parallel_conversion ))
    
    # Split files and run parallel conversions
    conv_pids=()
    
    for i in $(seq 0 $((parallel_conversion - 1))); do
      process_file_list="${file_list}_${i}"
      
      # Split files using split command or sed
      start_line=$((i * files_per_process + 1))
      end_line=$((start_line + files_per_process - 1))
      
      if [ $i -eq $((parallel_conversion - 1)) ]; then
        # Last process takes remaining files
        tail -n +${start_line} "${file_list}" > "${process_file_list}"
      else
        sed -n "${start_line},${end_line}p" "${file_list}" > "${process_file_list}"
      fi
      
      # Count files for this process
      process_file_count=$(wc -l < "${process_file_list}")
      if [ "$process_file_count" -eq 0 ]; then
        rm -f "${process_file_list}"
        continue
      fi
      
      # Run conversion for this subset in background
      (
        file_num=0
        echo "    [Process $((i+1))/${parallel_conversion}] Converting ${process_file_count} files..."
        while IFS= read -r laz_file; do
          file_num=$((file_num + 1))
          rel="${laz_file#"${input_dir}/"}"
          base="${rel%.laz}"
          out="${copc_dir}/${base}.copc.laz"
          mkdir -p "$(dirname "${out}")"
          
          # Use pdal wrench for conversion
          # Use pdal wrench for conversion (prefer retile_env)
          PDAL_WRENCH="${PDAL_WRENCH:-pdal_wrench}"
          if [ -f "/home/kg281/anaconda3/envs/retile_env/bin/pdal_wrench" ]; then
            PDAL_WRENCH="/home/kg281/anaconda3/envs/retile_env/bin/pdal_wrench"
          fi
          if ${PDAL_WRENCH} translate --input="${laz_file}" --output="${out}" \
            > "${log_folder}/copc_conv_${i}_$(basename "${base}").log" 2>&1
          then
            if [ -f "$out" ] && [ -s "$out" ]; then
              echo "    [Process $((i+1))/${parallel_conversion}] ✓ Finished ${file_num}/${process_file_count}: $(basename "${rel}")"
            else
              echo "    [Process $((i+1))/${parallel_conversion}] ✗ Failed ${file_num}/${process_file_count}: $(basename "${rel}") - Output empty"
              rm -f "$out" 2>/dev/null
            fi
          else
            echo "    [Process $((i+1))/${parallel_conversion}] ✗ Failed ${file_num}/${process_file_count}: $(basename "${rel}") - Conversion error"
            rm -f "$out" 2>/dev/null
          fi
        done < "${process_file_list}"
        echo "    [Process $((i+1))/${parallel_conversion}] Completed all ${process_file_count} files"
      ) &
      conv_pids+=($!)
    done
    
    # Wait for all conversion processes
    echo "  Waiting for ${#conv_pids[@]} parallel conversion processes..."
    for pid in "${conv_pids[@]}"; do
      wait $pid
    done
    
    # Cleanup
    rm -f "${file_list}" "${file_list}"_*
    
    echo "  Parallel conversion completed"
  fi
  # Create list of existing COPC files
  find "${copc_dir}" -name "*.copc.laz" 2>/dev/null | sort > "${copc_list_file}" || touch "${copc_list_file}"
fi

if [ ! -s "${copc_list_file}" ]; then
  echo "ERROR: No COPC files found in ${copc_dir}."
  echo "Please either:"
  if [ "$USE_FILE_LIST" = true ]; then
    echo "  1. Check that the file list contains valid LAS/LAZ file paths, or"
  else
    echo "  1. Set input_dir to point to your input LAZ files, or"
  fi
  echo "  2. Place COPC files directly in ${copc_dir}"
  exit 1
fi

# Step 2: Build tindex if it doesn't exist
if [ ! -f "${tindex_file}" ]; then
  echo "=== Step 2: Building tindex from COPC files ==="
  bash "${build_tindex_script}" "${copc_dir}" "${tindex_file}"
else
  echo "=== Step 2: Using existing tindex: ${tindex_file} ==="
fi

# Step 3: Calculate tile bounds and prepare jobs
echo "=== Step 3: Calculating tile bounds ==="
tile_jobs_file="${output_dir}/tile_jobs_${tile_length}m.txt"
tile_bounds_json="${output_dir}/tile_bounds_tindex.json"

eval "$(
  python "$(dirname "$0")/prepare_tile_jobs.py" "${tindex_file}" \
    --tile-length="${tile_length}" \
    --tile-buffer="${tile_buffer}" \
    --jobs-out="${tile_jobs_file}" \
    --bounds-out="${tile_bounds_json}" \
    --grid-offset="${grid_offset}"
)"

# Generate the plot
echo "=== Step 4: Inspect the extents and files ==="
python "$(dirname "$0")/plot_tiles_and_copc.py" "${tindex_file}" "${tile_bounds_json}" --output="${tiles_dir}/tile_${tile_length}m_and_copc_overview.png"

# Step 5: Clean up old .laz files (from previous runs that used pdal merge)
echo "=== Step 5: Cleaning up old .laz files ==="
if [ -d "${tiles_dir}" ]; then
    cleaned=0
    for old_laz in "${tiles_dir}"/*.laz; do
        # Check if file exists and is not a .copc.laz file
        if [ -f "$old_laz" ] && [[ ! "$old_laz" =~ \.copc\.laz$ ]]; then
            local basename=$(basename "$old_laz" .laz)
            local copc_file="${tiles_dir}/${basename}.copc.laz"
            # Remove .laz file if corresponding .copc.laz exists
            if [ -f "$copc_file" ]; then
                rm -f "$old_laz"
                cleaned=$((cleaned + 1))
                echo "  Removed old .laz file: $(basename "$old_laz") (corresponding .copc.laz exists)" >&2
            fi
        fi
    done
    if [ $cleaned -gt 0 ]; then
        echo "  Cleaned up $cleaned old .laz file(s)" >&2
    fi
fi

# Step 6: Process each tile
echo "=== Step 6: Processing tiles ==="

process_tile() {
    local label="$1"
    local proj_bounds="$2"
    local geo_bounds="$3"
    
    # Check if final tile already exists and is valid
    local final_tile="${tiles_dir}/${label}.copc.laz"
    if [ -f "$final_tile" ] && [ -s "$final_tile" ]; then
        # Remove old .laz file if it exists (from previous runs)
        local old_laz_file="${tiles_dir}/${label}.laz"
        if [ -f "$old_laz_file" ]; then
            rm -f "$old_laz_file"
            echo "  Removed old .laz file: ${label}.laz" >&2
        fi
        echo "  Skipping tile ${label} (already exists)" >&2
        return 0
    fi
    
    echo "Processing tile ${label}..." >&2
    
    # Get all COPC files from tindex (bounds filtering happens in COPC reader)
    local copc_files=$(python -c "
import sys
import sqlite3

tindex_path = '${tindex_file}'

try:
    conn = sqlite3.connect(tindex_path)
    cursor = conn.cursor()
    
    # Get the table name from gpkg_contents (pdal tindex creates one table with all files)
    cursor.execute('SELECT table_name FROM gpkg_contents WHERE data_type = \"features\" LIMIT 1')
    result = cursor.fetchone()
    if result:
        table_name = result[0]
        cursor.execute(f'SELECT DISTINCT Location FROM \"{table_name}\"')
        files = [row[0] for row in cursor.fetchall()]
        print(' '.join(f\"'{f}'\" for f in files))
    conn.close()
except Exception as e:
    print('')
" 2>/dev/null || echo "")
    
    if [ -z "$copc_files" ]; then
        echo "  No COPC files for tile ${label}" >&2
        return 0
    fi
    
    # Create temporary tile directory for parts
    local tile_dir="${tiles_dir}/${label}"
    mkdir -p "${tile_dir}"
    
    # Extract from each COPC file with bounds filtering using direct COPC reader
    local part_num=0
    local parts_created=0
    
    for copc_file in $copc_files; do
        # Remove quotes if present
        copc_file=$(echo "$copc_file" | sed "s/'//g")
        
        if [ ! -f "$copc_file" ]; then
            echo "  Warning: COPC file not found: $copc_file" >&2
            continue
        fi
        
        local part_file="${tile_dir}/part_${part_num}.copc.laz"
        local pipeline_tmp="${log_folder}/${label}_part${part_num}_pipeline.json"
        
        # Create temporary pipeline JSON file
        # Note: writers.las doesn't support threads, but we can use it for readers.copc if needed
        # For now, threads parameter is mainly for COPC writer operations
        cat > "${pipeline_tmp}" <<EOF
[
    {
        "type": "readers.copc",
        "filename": "${copc_file}",
        "bounds": "${proj_bounds}"
    },
    {
        "type": "writers.copc",
        "filename": "${part_file}",
        "threads": ${threads},
        "forward": "all"
    }
]
EOF
        
        # Use direct COPC reader with bounds for efficient spatial filtering
        if pdal pipeline "${pipeline_tmp}" > "${log_folder}/${label}_part${part_num}.log" 2>&1
        then
            rm -f "${pipeline_tmp}"
            if [ -f "$part_file" ] && [ -s "$part_file" ]; then
                parts_created=$((parts_created + 1))
            else
                rm -f "${part_file}" 2>/dev/null
            fi
        fi
        part_num=$((part_num + 1))
    done
    
    if [ $parts_created -eq 0 ]; then
        echo "  No parts created for tile ${label}" >&2
        rmdir "${tile_dir}" 2>/dev/null
        return 0
    fi
    
    # Merge all parts into final COPC tile
    local parts=("${tile_dir}"/part_*.copc.laz)
    
    if [ ${#parts[@]} -eq 0 ]; then
        echo "  No parts to merge for tile ${label}" >&2
        rmdir "${tile_dir}" 2>/dev/null
        return 1
    fi
    
    # Create PDAL pipeline to merge COPC parts into COPC output
    local merge_pipeline="${log_folder}/${label}_merge_pipeline.json"
    
    # Build pipeline JSON with multiple COPC readers and one COPC writer
    echo "[" > "${merge_pipeline}"
    local first=true
    for part in "${parts[@]}"; do
        if [ "$first" = true ]; then
            first=false
        else
            echo "," >> "${merge_pipeline}"
        fi
        cat >> "${merge_pipeline}" <<EOF
    {
        "type": "readers.copc",
        "filename": "${part}"
    }
EOF
    done
    cat >> "${merge_pipeline}" <<EOF
,
    {
        "type": "filters.merge"
    },
    {
        "type": "writers.copc",
        "filename": "${final_tile}",
        "threads": ${threads},
        "forward": "all",
        "scale_x": 0.01,
        "scale_y": 0.01,
        "scale_z": 0.01
    }
]
EOF
    
    # Merge using PDAL pipeline to preserve COPC format
    if pdal pipeline "${merge_pipeline}" \
        > "${log_folder}/${label}_merge.log" 2>&1
    then
        rm -f "${merge_pipeline}"
        # Verify output file exists and has content
        if [ -f "${final_tile}" ] && [ -s "${final_tile}" ]; then
            # Clean up old .laz file if it exists (from previous runs)
            local old_laz_file="${tiles_dir}/${label}.laz"
            if [ -f "$old_laz_file" ]; then
                rm -f "$old_laz_file"
                echo "  Removed old .laz file: ${label}.laz" >&2
            fi
            # Clean up parts and temporary directory
            rm -f "${parts[@]}"
            rmdir "${tile_dir}" 2>/dev/null
            echo "  Completed tile ${label} (${parts_created} parts) -> COPC" >&2
            return 0
        else
            echo "  Error: Merged COPC file is empty for tile ${label}" >&2
            return 1
        fi
    else
        echo "  Error merging tile ${label} to COPC" >&2
        cat "${log_folder}/${label}_merge.log" >&2
        rm -f "${merge_pipeline}"
        return 1
    fi
}

# Process tiles in parallel
# Process tiles in parallel
max_procs=${MAX_TILE_PROCS}
pids=()
failed_tiles=()

while IFS='|' read -r label proj_bounds geo_bounds; do
    [[ -z "$label" ]] && continue
    
    process_tile "$label" "$proj_bounds" "$geo_bounds" &
    pids+=($!)
    
    # Wait for any process to finish if we've reached the limit
    if [ ${#pids[@]} -ge ${max_procs} ]; then
        wait -n  # Wait for any background process to finish
        # Remove finished PIDs from array
        new_pids=()
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                # Process still running
                new_pids+=("$pid")
            else
                # Process finished, check exit code
                wait "$pid" 2>/dev/null
                exit_code=$?
                if [ $exit_code -ne 0 ]; then
                    failed_tiles+=("tile_${pid}")
                fi
            fi
        done
        pids=("${new_pids[@]}")
    fi
done < "${tile_jobs_file}"

# Wait for remaining jobs
for pid in "${pids[@]}"; do
    wait $pid
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        failed_tiles+=("tile_${pid}")
    fi
done

if [ ${#failed_tiles[@]} -gt 0 ]; then
    echo "ERROR: Some tiles failed to process:"
    for tile in "${failed_tiles[@]}"; do
        echo "  - ${tile}"
    done
    exit 1
else
    echo "All tiles processed successfully!"
fi

