#!/usr/bin/env python3
"""
Visualize the overlap density test to show how it works and the results.
"""

import sys
import json
import pdal
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

def parse_bounds(bounds_str):
    """Parse bounds string: ([xmin, xmax], [ymin, ymax])"""
    bounds_str = bounds_str.strip('()')
    parts = bounds_str.split('],[')
    x_part = parts[0].strip('[')
    y_part = parts[1].strip(']')
    
    xmin, xmax = map(float, x_part.split(','))
    ymin, ymax = map(float, y_part.split(','))
    
    return xmin, xmax, ymin, ymax

def count_points_in_bounds(copc_file, xmin, xmax, ymin, ymax):
    """Count points in a bounded region using Python PDAL."""
    try:
        pipeline_json = [
            {
                "type": "readers.copc",
                "filename": str(copc_file),
                "bounds": f"([{xmin},{xmax}],[{ymin},{ymax}])"
            }
        ]
        
        pipeline = pdal.Pipeline(json.dumps(pipeline_json))
        pipeline.execute()
        
        arrays = pipeline.arrays
        if len(arrays) > 0:
            return len(arrays[0])
        return 0
    except Exception as e:
        return None

def main():
    if len(sys.argv) != 3:
        print("Usage: visualize_overlap_test.py <tiles_dir> <tile_jobs_file>")
        sys.exit(1)
    
    tiles_dir = Path(sys.argv[1])
    tile_jobs_file = sys.argv[2]
    
    # Read tile jobs
    tiles = []
    with open(tile_jobs_file, 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 2:
                label = parts[0].strip()
                proj_bounds = parts[1].strip()
                xmin, xmax, ymin, ymax = parse_bounds(proj_bounds)
                tiles.append({
                    'label': label,
                    'xmin': xmin,
                    'xmax': xmax,
                    'ymin': ymin,
                    'ymax': ymax,
                    'file': tiles_dir / f"{label}.copc.laz"
                })
    
    # Find all overlaps
    overlaps = []
    for i in range(len(tiles) - 1):
        tile1 = tiles[i]
        tile2 = tiles[i + 1]
        
        # Check if they overlap (adjacent tiles)
        overlap_xmin = max(tile1['xmin'], tile2['xmin'])
        overlap_xmax = min(tile1['xmax'], tile2['xmax'])
        overlap_ymin = max(tile1['ymin'], tile2['ymin'])
        overlap_ymax = min(tile1['ymax'], tile2['ymax'])
        
        if overlap_xmin < overlap_xmax and overlap_ymin < overlap_ymax:
            overlap_area = (overlap_xmax - overlap_xmin) * (overlap_ymax - overlap_ymin)
            
            # Count points
            count1 = count_points_in_bounds(tile1['file'], overlap_xmin, overlap_xmax, overlap_ymin, overlap_ymax)
            count2 = count_points_in_bounds(tile2['file'], overlap_xmin, overlap_xmax, overlap_ymin, overlap_ymax)
            
            overlaps.append({
                'tile1': tile1['label'],
                'tile2': tile2['label'],
                'xmin': overlap_xmin,
                'xmax': overlap_xmax,
                'ymin': overlap_ymin,
                'ymax': overlap_ymax,
                'count1': count1,
                'count2': count2,
                'area': overlap_area
            })
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Adjust layout to prevent overflow
    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.12, wspace=0.3)
    
    # Left plot: Tile layout with overlaps highlighted
    ax1 = axes[0]
    ax1.set_title('Tile Layout with Overlapping Regions', fontsize=14, fontweight='bold')
    
    # Draw tiles
    for tile in tiles:
        if tile['file'].exists():
            rect = patches.Rectangle(
                (tile['xmin'], tile['ymin']),
                tile['xmax'] - tile['xmin'],
                tile['ymax'] - tile['ymin'],
                linewidth=2, edgecolor='black', facecolor='lightblue', alpha=0.5
            )
            ax1.add_patch(rect)
            
            # Label
            center_x = (tile['xmin'] + tile['xmax']) / 2
            center_y = (tile['ymin'] + tile['ymax']) / 2
            ax1.text(center_x, center_y, tile['label'], 
                    ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Highlight overlaps
    for overlap in overlaps:
        if overlap['count1'] is not None and overlap['count2'] is not None:
            # Color based on whether merge is working
            if overlap['count1'] > 0 and overlap['count2'] > 0:
                ratio = max(overlap['count1'], overlap['count2']) / min(overlap['count1'], overlap['count2'])
                if ratio > 2.0:
                    color = 'red'
                    alpha = 0.7
                elif ratio > 1.5:
                    color = 'orange'
                    alpha = 0.6
                else:
                    color = 'green'
                    alpha = 0.5
            else:
                color = 'red'
                alpha = 0.8
            
            rect = patches.Rectangle(
                (overlap['xmin'], overlap['ymin']),
                overlap['xmax'] - overlap['xmin'],
                overlap['ymax'] - overlap['ymin'],
                linewidth=3, edgecolor=color, facecolor=color, alpha=alpha
            )
            ax1.add_patch(rect)
            
            # Add annotation
            center_x = (overlap['xmin'] + overlap['xmax']) / 2
            center_y = (overlap['ymin'] + overlap['ymax']) / 2
            if overlap['count1'] is not None and overlap['count2'] is not None:
                if overlap['count1'] > 0 and overlap['count2'] > 0:
                    ratio = max(overlap['count1'], overlap['count2']) / min(overlap['count1'], overlap['count2'])
                    label = f"Overlap\n{overlap['tile1']}/{overlap['tile2']}\nRatio: {ratio:.2f}x"
                else:
                    label = f"Overlap\n{overlap['tile1']}/{overlap['tile2']}\n⚠️ Zero points!"
            else:
                label = f"Overlap\n{overlap['tile1']}/{overlap['tile2']}"
            
            ax1.text(center_x, center_y, label, 
                    ha='center', va='center', fontsize=8, 
                    fontweight='bold', color='white' if color in ['red', 'green'] else 'black',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    ax1.set_xlabel('X (meters)', fontsize=12)
    ax1.set_ylabel('Y (meters)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # Right plot: Point count comparison
    ax2 = axes[1]
    ax2.set_title('Point Count Comparison in Overlapping Regions', fontsize=14, fontweight='bold')
    
    overlap_labels = []
    counts1 = []
    counts2 = []
    ratios = []
    
    for overlap in overlaps:
        if overlap['count1'] is not None and overlap['count2'] is not None:
            overlap_labels.append(f"{overlap['tile1']}\n/\n{overlap['tile2']}")
            counts1.append(overlap['count1'] / 1e6)  # Convert to millions
            counts2.append(overlap['count2'] / 1e6)
            if overlap['count1'] > 0 and overlap['count2'] > 0:
                ratios.append(max(overlap['count1'], overlap['count2']) / min(overlap['count1'], overlap['count2']))
            else:
                ratios.append(float('inf'))
    
    if overlap_labels:
        x = np.arange(len(overlap_labels))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, counts1, width, label='Tile 1', color='skyblue', alpha=0.8)
        bars2 = ax2.bar(x + width/2, counts2, width, label='Tile 2', color='lightcoral', alpha=0.8)
        
        # Add ratio annotations
        for i, (bar1, bar2, ratio) in enumerate(zip(bars1, bars2, ratios)):
            if ratio != float('inf'):
                if ratio > 2.0:
                    color = 'red'
                    symbol = '⚠️'
                elif ratio > 1.5:
                    color = 'orange'
                    symbol = '⚠️'
                else:
                    color = 'green'
                    symbol = '✓'
                
                height = max(bar1.get_height(), bar2.get_height())
                ax2.text(bar1.get_x() + bar1.get_width() / 2, height * 1.05,
                        f"{symbol}\n{ratio:.2f}x", ha='center', va='bottom',
                        fontsize=9, fontweight='bold', color=color)
            else:
                height = max(bar1.get_height(), bar2.get_height())
                ax2.text(bar1.get_x() + bar1.get_width() / 2, height * 1.05,
                        '⚠️\nZero!', ha='center', va='bottom',
                        fontsize=9, fontweight='bold', color='red')
        
        ax2.set_xlabel('Overlapping Tile Pairs', fontsize=12)
        ax2.set_ylabel('Point Count (millions)', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(overlap_labels, fontsize=9)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
    
    # Save figure
    output_file = tiles_dir.parent / 'overlap_density_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)  # Close to free memory
    print(f"\nVisualization saved to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("OVERLAP DENSITY TEST EXPLANATION")
    print("=" * 80)
    print("""
HOW THE TEST WORKS:
1. For each pair of adjacent tiles, we identify the overlapping region
2. We count points in that overlap region from BOTH tiles
3. We compare the point counts:
   - If merge filter is working: Both tiles should have similar counts (ratio ~1.0x)
   - If merge filter is NOT working: One tile will have many fewer points (ratio >2.0x)
   
WHY THIS MATTERS:
- Without merge filter: Only the last file's points are written, so overlapping areas
  from earlier files are lost
- With merge filter: All points from all overlapping source files are combined,
  so both tiles should have the same merged points in their overlap regions

INTERPRETATION:
- ✓ Green (ratio ~1.0x): Merge is working correctly - both tiles have merged points
- ⚠️ Orange (ratio 1.5-2.0x): Moderate difference - check merge filter
- ⚠️ Red (ratio >2.0x or zero): Merge may not be working - points are missing
""")
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    for overlap in overlaps:
        if overlap['count1'] is not None and overlap['count2'] is not None:
            if overlap['count1'] > 0 and overlap['count2'] > 0:
                ratio = max(overlap['count1'], overlap['count2']) / min(overlap['count1'], overlap['count2'])
                density1 = overlap['count1'] / overlap['area']
                density2 = overlap['count2'] / overlap['area']
                status = "✓ OK" if ratio <= 1.5 else "⚠️ WARNING" if ratio <= 2.0 else "❌ FAIL"
                print(f"{status} {overlap['tile1']} ↔ {overlap['tile2']}: "
                      f"Counts: {overlap['count1']:,} vs {overlap['count2']:,} "
                      f"(ratio: {ratio:.2f}x, densities: {density1:.1f} vs {density2:.1f} pts/m²)")
            else:
                print(f"❌ FAIL {overlap['tile1']} ↔ {overlap['tile2']}: "
                      f"One tile has zero points! (Counts: {overlap['count1']} vs {overlap['count2']})")
    
    # Don't show plot, just save it
    # plt.show()  # Commented out to avoid GUI requirement

if __name__ == "__main__":
    main()

