# Overlap Density Test Explanation

## What is the Test?

The overlap density test verifies that the **merge filter** in the retiling script is working correctly. It checks whether overlapping regions between adjacent tiles contain points from **all contributing source files** or just the **last file**.

## How It Works

### Step 1: Identify Overlapping Regions
For each pair of adjacent tiles, we calculate their overlapping area:
- **Tile 1 bounds**: `[xmin1, xmax1] × [ymin1, ymax1]`
- **Tile 2 bounds**: `[xmin2, xmax2] × [ymin2, ymax2]`
- **Overlap region**: `[max(xmin1, xmin2), min(xmax1, xmax2)] × [max(ymin1, ymin2), min(ymax1, ymax2)]`

### Step 2: Count Points in Overlap
For each tile, we count how many points exist in the overlapping region:
- Use PDAL to read only the overlap bounds from each COPC tile
- Count the points returned

### Step 3: Compare Point Counts
Compare the point counts from both tiles:
- **If merge filter works**: Both tiles should have **similar counts** (ratio ~1.0x)
- **If merge filter fails**: One tile will have **significantly fewer points** (ratio >2.0x) or zero

## Why This Matters

### Without Merge Filter (❌ Broken)
```
Source Files:  [File A]  [File B]  [File C]
                    ↓         ↓         ↓
Tile 1:        [Points from A, B, C merged] ✓
Tile 2:        [Only points from C] ✗  (A and B overwritten!)
```

**Problem**: When multiple COPC readers feed into one COPC writer without a merge filter, PDAL processes them as separate "views". The writer only writes the **last view**, losing all points from earlier files.

### With Merge Filter (✓ Fixed)
```
Source Files:  [File A]  [File B]  [File C]
                    ↓         ↓         ↓
Merge Filter:  [All points combined] ✓
                    ↓
Tile 1:        [Points from A, B, C merged] ✓
Tile 2:        [Points from A, B, C merged] ✓
```

**Solution**: The `filters.merge` stage combines all views into one before writing, so both tiles get all the merged points.

## Visual Representation

```
┌─────────────────────────────────────────────────────────┐
│                    TILE LAYOUT                           │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐  ┌──────────────┐                    │
│  │   c00_r00    │  │   c01_r00    │                    │
│  │              │  │              │                    │
│  │  [Tile 1]    │  │  [Tile 3]    │                    │
│  └──────┬───────┘  └──────┬───────┘                    │
│         │                  │                             │
│    ┌────▼──────────────────▼────┐  ← OVERLAP REGION   │
│    │   OVERLAP AREA             │     (should have     │
│    │   Both tiles contribute    │      same points)    │
│    │   points here              │                      │
│    └────▲──────────────────▲────┘                     │
│         │                  │                             │
│  ┌──────┴───────┐  ┌──────┴───────┐                    │
│  │   c00_r01    │  │   c01_r01    │                    │
│  │              │  │              │                    │
│  │  [Tile 2]    │  │  [Tile 4]    │                    │
│  └──────────────┘  └──────────────┘                    │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Interpreting Results

### ✓ Good (Ratio ~1.0x)
```
Points in c00_r00: 32,735,059 (density: 1515.51 pts/m²)
Points in c00_r01: 32,733,510 (density: 1515.44 pts/m²)
✓ Similar point counts (ratio: 1.00x) - merge appears to be working
```
**Meaning**: Both tiles have the same merged points in the overlap. Merge filter is working!

### ⚠️ Warning (Ratio 1.5-2.0x)
```
Points in tile1: 1,000,000
Points in tile2: 600,000
⚠️ CAUTION: Moderate difference (ratio: 1.67x) - check merge filter
```
**Meaning**: Some points might be missing. Check merge filter configuration.

### ❌ Fail (Ratio >2.0x or Zero)
```
Points in tile1: 1,000,000
Points in tile2: 0
❌ FAIL: One tile has zero points!
```
**Meaning**: Merge filter is NOT working. Only one tile has points, meaning earlier files were overwritten.

## Example Results from kaltenborn Dataset

```
Analyzing overlap between c00_r00 and c00_r01:
  Overlap bounds: X=[585777.00, 586137.00], Y=[5626014.46, 5626074.46]
  Overlap area: 21600.00 m²
  Points in c00_r00: 32,735,059 (density: 1515.51 pts/m²)
  Points in c00_r01: 32,733,510 (density: 1515.44 pts/m²)
  ✓ Similar point counts (ratio: 1.00x) - merge appears to be working
```

**Interpretation**: Perfect! Both tiles have nearly identical point counts (difference of only ~1,500 points out of 32+ million), confirming the merge filter is working correctly.

## Running the Test

```bash
# Activate environment
conda activate retile_env

# Run analysis
python3 check_overlap_density_pdalpy.py <tiles_dir> <tile_jobs_file>

# Generate visualization
python3 visualize_overlap_test.py <tiles_dir> <tile_jobs_file>
```

## Key Takeaway

The test verifies that **overlapping regions contain merged points from all source files**, not just the last one. When the merge filter works correctly, adjacent tiles will have **identical point counts** in their overlapping regions, confirming that all source data is preserved.






