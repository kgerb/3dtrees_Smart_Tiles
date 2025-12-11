#!/bin/bash
# Check point density in overlapping areas between adjacent tiles

TILES_DIR="$1"
TILE_JOBS_FILE="$2"

if [ -z "$TILES_DIR" ] || [ -z "$TILE_JOBS_FILE" ]; then
    echo "Usage: $0 <tiles_dir> <tile_jobs_file>"
    exit 1
fi

echo "=================================================================================="
echo "Point Density Analysis in Overlapping Areas"
echo "=================================================================================="
echo ""

# Parse tile bounds and analyze overlaps
prev_label=""
prev_xmin=""
prev_xmax=""
prev_ymin=""
prev_ymax=""

while IFS='|' read -r label proj_bounds geo_bounds; do
    [[ -z "$label" ]] && continue
    
    # Parse bounds: ([xmin, xmax], [ymin, ymax])
    bounds_clean=$(echo "$proj_bounds" | sed 's/[()]//g')
    x_part=$(echo "$bounds_clean" | cut -d',' -f1 | sed 's/\[//')
    y_part=$(echo "$bounds_clean" | cut -d',' -f2 | sed 's/\]//')
    
    xmin=$(echo "$x_part" | cut -d',' -f1)
    xmax=$(echo "$x_part" | cut -d',' -f2)
    ymin=$(echo "$y_part" | cut -d',' -f1)
    ymax=$(echo "$y_part" | cut -d',' -f2)
    
    if [ -n "$prev_label" ]; then
        # Find overlap region
        overlap_xmin=$(echo "$xmin $prev_xmin" | awk '{if ($1 > $2) print $1; else print $2}')
        overlap_xmax=$(echo "$xmax $prev_xmax" | awk '{if ($1 < $2) print $1; else print $2}')
        overlap_ymin=$(echo "$ymin $prev_ymin" | awk '{if ($1 > $2) print $1; else print $2}')
        overlap_ymax=$(echo "$ymax $prev_ymax" | awk '{if ($1 < $2) print $1; else print $2}')
        
        # Check if there's actual overlap
        if (( $(echo "$overlap_xmin < $overlap_xmax" | bc -l) )) && \
           (( $(echo "$overlap_ymin < $overlap_ymax" | bc -l) )); then
            
            overlap_area=$(echo "($overlap_xmax - $overlap_xmin) * ($overlap_ymax - $overlap_ymin)" | bc -l)
            
            tile1_file="${TILES_DIR}/${prev_label}.copc.laz"
            tile2_file="${TILES_DIR}/${label}.copc.laz"
            
            if [ ! -f "$tile1_file" ] || [ ! -f "$tile2_file" ]; then
                echo "Warning: One or both tile files not found"
                continue
            fi
            
            echo "Analyzing overlap between ${prev_label} and ${label}:"
            echo "  Overlap bounds: X=[$overlap_xmin, $overlap_xmax], Y=[$overlap_ymin, $overlap_ymax]"
            printf "  Overlap area: %.2f m²\n" "$overlap_area"
            
            # Count points using pdal info
            echo "  Counting points in ${prev_label}..."
            count1=$(pdal info --summary "${tile1_file}" \
                --filters.hexbin.edge=1000 \
                --filters.hexbin.threshold=1 \
                2>/dev/null | \
                python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    bounds = data.get('metadata', {}).get('filters.hexbin', {}).get('boundary', {})
    if bounds:
        print(int(bounds.get('points', 0)))
    else:
        # Fallback: use stats
        stats = data.get('stats', {})
        print(int(stats.get('bbox', {}).get('native', {}).get('points', 0)))
except:
    print('0')
" 2>/dev/null || echo "0")
            
            # Better approach: use pdal pipeline with bounds filter
            temp_pipeline1=$(mktemp)
            cat > "$temp_pipeline1" <<EOF
[
    {
        "type": "readers.copc",
        "filename": "${tile1_file}",
        "bounds": "([${overlap_xmin},${overlap_xmax}],[${overlap_ymin},${overlap_ymax}])"
    }
]
EOF
            
            count1=$(pdal info "$temp_pipeline1" 2>/dev/null | \
                python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    stats = data.get('stats', {})
    count = stats.get('bbox', {}).get('native', {}).get('points', 0)
    print(int(count))
except:
    print('0')
" 2>/dev/null || echo "0")
            
            temp_pipeline2=$(mktemp)
            cat > "$temp_pipeline2" <<EOF
[
    {
        "type": "readers.copc",
        "filename": "${tile2_file}",
        "bounds": "([${overlap_xmin},${overlap_xmax}],[${overlap_ymin},${overlap_ymax}])"
    }
]
EOF
            
            count2=$(pdal info "$temp_pipeline2" 2>/dev/null | \
                python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    stats = data.get('stats', {})
    count = stats.get('bbox', {}).get('native', {}).get('points', 0)
    print(int(count))
except:
    print('0')
" 2>/dev/null || echo "0")
            
            rm -f "$temp_pipeline1" "$temp_pipeline2"
            
            if [ "$count1" -gt 0 ] && [ "$count2" -gt 0 ]; then
                density1=$(echo "scale=2; $count1 / $overlap_area" | bc -l)
                density2=$(echo "scale=2; $count2 / $overlap_area" | bc -l)
                
                printf "  Points in %s: %'d (density: %.2f pts/m²)\n" "$prev_label" "$count1" "$density1"
                printf "  Points in %s: %'d (density: %.2f pts/m²)\n" "$label" "$count2" "$density2"
                
                # Calculate ratio
                if [ "$count1" -gt "$count2" ]; then
                    ratio=$(echo "scale=2; $count1 / $count2" | bc -l)
                else
                    ratio=$(echo "scale=2; $count2 / $count1" | bc -l)
                fi
                
                if (( $(echo "$ratio > 2.0" | bc -l) )); then
                    printf "  ⚠️  WARNING: Large difference (ratio: %.2fx) - merge may not be working correctly\n" "$ratio"
                elif (( $(echo "$ratio > 1.5" | bc -l) )); then
                    printf "  ⚠️  CAUTION: Moderate difference (ratio: %.2fx) - check merge filter\n" "$ratio"
                else
                    printf "  ✓ Similar point counts (ratio: %.2fx) - merge appears to be working\n" "$ratio"
                fi
            else
                echo "  Could not count points (count1=$count1, count2=$count2)"
            fi
            
            echo ""
        fi
    fi
    
    prev_label="$label"
    prev_xmin="$xmin"
    prev_xmax="$xmax"
    prev_ymin="$ymin"
    prev_ymax="$ymax"
done < "$TILE_JOBS_FILE"






