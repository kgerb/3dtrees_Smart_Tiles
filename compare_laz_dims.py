#!/usr/bin/env python3
"""Print and compare dimension names of two LAZ/LAS files.

Usage:
  python compare_laz_dims.py <file1.laz> <file2.laz>

Where the merged file with added dimensions is stored:
  - Merge task (multi-tile): The main merged LAZ is enriched in place, so it is at
    output_merged (e.g. /out/merged.laz or --output-merged-laz). A copy is also
    written to output_tiles_dir/merged.laz when that differs.
  - remap_to_originals task: Written to output_dir/merged_with_originals.laz by
    default, or to the path given by --output-merged-with-originals.

To verify merged has same dimensions as originals:
  python compare_laz_dims.py <path/to/original.laz> <path/to/merged.laz>
"""

import sys
from pathlib import Path

try:
    import laspy
except ImportError:
    print("Error: laspy required. Install with: pip install laspy[lazrs]", file=sys.stderr)
    sys.exit(1)


def get_dimension_names(path: Path) -> list[str]:
    with laspy.open(str(path), laz_backend=laspy.LazBackend.LazrsParallel) as f:
        return list(f.header.point_format.dimension_names)


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_laz_dims.py <original.laz> <merged.laz>", file=sys.stderr)
        print("  Compares dimension names; use to verify merged has original dimensions.", file=sys.stderr)
        sys.exit(1)
    p1 = Path(sys.argv[1])
    p2 = Path(sys.argv[2])
    if not p1.exists():
        print(f"Error: not found: {p1}", file=sys.stderr)
        sys.exit(1)
    if not p2.exists():
        print(f"Error: not found: {p2}", file=sys.stderr)
        sys.exit(1)

    dims1 = get_dimension_names(p1)
    dims2 = get_dimension_names(p2)
    set1 = set(dims1)
    set2 = set(dims2)

    print(f"File 1: {p1.name}")
    print(f"  Dimensions ({len(dims1)}): {', '.join(dims1)}")
    print()
    print(f"File 2: {p2.name}")
    print(f"  Dimensions ({len(dims2)}): {', '.join(dims2)}")
    print()
    only1 = set1 - set2
    only2 = set2 - set1
    common = set1 & set2
    if only1:
        print(f"  Only in file 1: {', '.join(sorted(only1))}")
    if only2:
        print(f"  Only in file 2: {', '.join(sorted(only2))}")
    print(f"  Common: {len(common)} dimensions")
    if set1 != set2:
        print("  => Different dimensions.")
    else:
        print("  => Same dimensions.")


if __name__ == "__main__":
    main()
