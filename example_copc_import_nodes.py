#!/usr/bin/env python3
"""
Example script: Import COPC file with specific bounds and show how many nodes are imported.

This demonstrates:
1. Reading a COPC file with spatial bounds filtering
2. Counting total nodes in the COPC file
3. Counting nodes that overlap with the specified bounds
4. Extracting points using PDAL and showing node count from verbose output

Usage:
    python example_copc_import_nodes.py <copc_file> <bounds>
    
    bounds format: "([minx,maxx],[miny,maxy])" in the COPC file's CRS
    Example: "([500000,500100],[6000000,6000100])"
"""

import sys
import subprocess
import re
import argparse
from pathlib import Path
import tempfile

try:
    import copclib as copc
    COPCLIB_AVAILABLE = True
except ImportError:
    COPCLIB_AVAILABLE = False
    print("WARNING: copclib not available. Install with: pip install copc-lib")
    print("Will use PDAL method only.\n")


def get_total_nodes_copclib(copc_file: Path) -> int:
    """Get total number of nodes in COPC file using copclib."""
    if not COPCLIB_AVAILABLE:
        return -1
    
    try:
        reader = copc.FileReader(str(copc_file))
        hierarchy = reader.GetHierarchy()
        return len(hierarchy)
    except Exception as e:
        print(f"ERROR reading with copclib: {e}", file=sys.stderr)
        return -1


def count_nodes_in_bounds_copclib(copc_file: Path, bounds: str) -> int:
    """Count nodes that overlap with bounds using copclib."""
    if not COPCLIB_AVAILABLE:
        return -1
    
    try:
        # Parse bounds: "([minx,maxx],[miny,maxy])"
        match = re.match(r'\(\[([\d.]+),([\d.]+)\],\[([\d.]+),([\d.]+)\]\)', bounds)
        if not match:
            print(f"ERROR: Invalid bounds format: {bounds}", file=sys.stderr)
            return -1
        
        minx, maxx, miny, maxy = map(float, match.groups())
        
        reader = copc.FileReader(str(copc_file))
        hierarchy = reader.GetHierarchy()
        
        node_count = 0
        for node in hierarchy.GetAllNodes():
            node_bounds = node.GetBounds()
            # Check if node bounds intersect with query bounds
            if (node_bounds.min.x < maxx and node_bounds.max.x > minx and
                node_bounds.min.y < maxy and node_bounds.max.y > miny):
                node_count += 1
        
        return node_count
    except Exception as e:
        print(f"ERROR counting nodes with copclib: {e}", file=sys.stderr)
        return -1


def extract_with_pdal_and_count_nodes(copc_file: Path, bounds: str) -> dict:
    """
    Extract from COPC file using PDAL and count nodes from verbose output.
    Returns dict with 'nodes', 'points', and 'success' keys.
    """
    # Create temporary pipeline JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        pipeline_json = Path(f.name)
        pipeline_content = f'''[
    {{
        "type": "readers.copc",
        "filename": "{copc_file}",
        "bounds": "{bounds}"
    }},
    {{
        "type": "writers.null"
    }}
]'''
        pipeline_json.write_text(pipeline_content)
    
    try:
        # Run PDAL with verbose output to capture node information
        result = subprocess.run(
            ['pdal', 'pipeline', str(pipeline_json), '--verbose=8'],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        output = result.stdout + result.stderr
        
        # Extract node count from PDAL verbose output
        # Look for pattern like "(pdal pipeline readers.copc Debug) 358 overlapping nodes"
        node_match = re.search(r'(\d+)\s+overlapping nodes', output, re.IGNORECASE)
        nodes = int(node_match.group(1)) if node_match else None
        
        # Extract point count if available
        point_match = re.search(r'(\d+)\s+points', output, re.IGNORECASE)
        points = int(point_match.group(1)) if point_match else None
        
        return {
            'nodes': nodes,
            'points': points,
            'success': result.returncode == 0,
            'output': output
        }
    except subprocess.TimeoutExpired:
        return {'nodes': None, 'points': None, 'success': False, 'output': 'Timeout'}
    except Exception as e:
        return {'nodes': None, 'points': None, 'success': False, 'output': str(e)}
    finally:
        # Cleanup
        if pipeline_json.exists():
            pipeline_json.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Import COPC file with bounds and show node count",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With bounds in projected coordinates
  python example_copc_import_nodes.py file.copc.laz "([500000,500100],[6000000,6000100])"
  
  # With bounds from a tile
  python example_copc_import_nodes.py file.copc.laz "([500000,500100],[6000000,6000100])" --show-details
        """
    )
    parser.add_argument(
        "copc_file",
        type=Path,
        help="Path to COPC file (.copc.laz)"
    )
    parser.add_argument(
        "bounds",
        type=str,
        help='Bounds in format: "([minx,maxx],[miny,maxy])"'
    )
    parser.add_argument(
        "--show-details",
        action="store_true",
        help="Show detailed PDAL output"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional: Save extracted points to this file (LAZ format)"
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not args.copc_file.exists():
        print(f"ERROR: COPC file not found: {args.copc_file}")
        sys.exit(1)
    
    print("=" * 70)
    print("COPC File Import with Bounds - Node Count Example")
    print("=" * 70)
    print(f"\nCOPC File: {args.copc_file}")
    print(f"Bounds:    {args.bounds}")
    print()
    
    # Method 1: Using copclib (if available)
    if COPCLIB_AVAILABLE:
        print("--- Method 1: Using copclib ---")
        total_nodes = get_total_nodes_copclib(args.copc_file)
        if total_nodes >= 0:
            print(f"  Total nodes in COPC file: {total_nodes:,}")
        
        overlapping_nodes = count_nodes_in_bounds_copclib(args.copc_file, args.bounds)
        if overlapping_nodes >= 0:
            print(f"  Nodes overlapping bounds: {overlapping_nodes:,}")
            if total_nodes > 0:
                percentage = (overlapping_nodes / total_nodes) * 100
                print(f"  Percentage of nodes used: {percentage:.1f}%")
        print()
    
    # Method 2: Using PDAL (always available)
    print("--- Method 2: Using PDAL ---")
    print("  Running PDAL pipeline with bounds filtering...")
    
    pdal_result = extract_with_pdal_and_count_nodes(args.copc_file, args.bounds)
    
    if pdal_result['success']:
        if pdal_result['nodes'] is not None:
            print(f"  ✓ Nodes read from COPC: {pdal_result['nodes']:,}")
        else:
            print(f"  ⚠ Node count not found in PDAL output")
            print(f"     (PDAL may not report node count in this version)")
        
        if pdal_result['points'] is not None:
            print(f"  ✓ Points extracted: {pdal_result['points']:,}")
        
        if args.show_details:
            print("\n  PDAL Output:")
            print("  " + "-" * 66)
            # Show relevant lines from output
            for line in pdal_result['output'].split('\n'):
                if any(keyword in line.lower() for keyword in ['node', 'point', 'bound', 'copc']):
                    print(f"  {line}")
    else:
        print(f"  ✗ PDAL extraction failed")
        print(f"  Error: {pdal_result['output']}")
    
    print()
    
    # Optional: Save output if requested
    if args.output:
        print(f"--- Saving extracted points ---")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            pipeline_json = Path(f.name)
            pipeline_content = f'''[
    {{
        "type": "readers.copc",
        "filename": "{args.copc_file}",
        "bounds": "{args.bounds}"
    }},
    {{
        "type": "writers.las",
        "filename": "{args.output}",
        "compression": true
    }}
]'''
            pipeline_json.write_text(pipeline_content)
        
        try:
            result = subprocess.run(
                ['pdal', 'pipeline', str(pipeline_json)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and args.output.exists():
                size_mb = args.output.stat().st_size / (1024 * 1024)
                print(f"  ✓ Saved to: {args.output} ({size_mb:.2f} MB)")
            else:
                print(f"  ✗ Failed to save output")
                print(f"  {result.stderr}")
        finally:
            if pipeline_json.exists():
                pipeline_json.unlink()
        print()
    
    # Summary
    print("=" * 70)
    print("Summary:")
    if COPCLIB_AVAILABLE and overlapping_nodes >= 0:
        print(f"  Nodes imported (copclib): {overlapping_nodes:,}")
    if pdal_result['nodes'] is not None:
        print(f"  Nodes imported (PDAL):    {pdal_result['nodes']:,}")
    if pdal_result['points'] is not None:
        print(f"  Points extracted:         {pdal_result['points']:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()


