#!/usr/bin/env python3
"""
Example: Transform ANTs points CSV through arbitrary number of transforms

This example demonstrates how to transform a CSV file containing points in
standard ANTs format (x, y, z, t) through an arbitrary number of transforms.

Input CSV format:
    x,y,z,t
    -41.73,-39.58,21.66,0
    -46.59,-40.00,56.32,0
    ...

The script applies transforms in the order specified and allows you to
invert individual transforms using a list of boolean flags.

Coordinate Systems:
- LPS (default): Left-Posterior-Superior - used by ITK/ANTs
- RAS (with --ras flag): Right-Anterior-Superior - used by NIfTI/neuroimaging tools

Usage:
    python transform_ants_points.py \\
        input.csv \\
        output.csv \\
        transform1.mat transform2.mat ... \\
        --useInverse 1 1 ... \\
        [--ras]

Example:
    python transform_ants_points.py \\
        test_points.csv \\
        transformed_points.csv \\
        postop_to_preop.mat preop_to_t1w.mat \\
        --useInverse 1 1

    With RAS coordinates:
    python transform_ants_points.py \\
        test_points_ras.csv \\
        transformed_points_ras.csv \\
        postop_to_preop.mat \\
        --useInverse 1 \\
        --ras
"""

import sys
from pathlib import Path
import csv
import argparse
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dbstoolbox.utils.transform_coordinates import apply_ants_transforms_to_points


def load_ants_points_csv(csv_path: Path) -> tuple[np.ndarray, list]:
    """
    Load ANTs points CSV file.

    Args:
        csv_path: Path to CSV file with x, y, z, t columns

    Returns:
        Tuple of (points array Nx3, t_values list)
    """
    points = []
    t_values = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = float(row['x'])
                y = float(row['y'])
                z = float(row['z'])
                t = float(row.get('t', 0))

                points.append([x, y, z])
                t_values.append(t)
            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping invalid row: {row}")
                continue

    return np.array(points), t_values


def save_ants_points_csv(csv_path: Path, points: np.ndarray, t_values: list,
                         original_points: np.ndarray = None):
    """
    Save points to ANTs CSV format with optional original coordinates.

    Args:
        csv_path: Output CSV path
        points: Nx3 array of transformed points
        t_values: List of t values (time/label)
        original_points: Optional Nx3 array of original points
    """
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['x', 'y', 'z', 't']
        if original_points is not None:
            fieldnames.extend(['x_original', 'y_original', 'z_original'])

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, (point, t) in enumerate(zip(points, t_values)):
            row = {
                'x': point[0],
                'y': point[1],
                'z': point[2],
                't': t
            }

            if original_points is not None:
                row['x_original'] = original_points[i, 0]
                row['y_original'] = original_points[i, 1]
                row['z_original'] = original_points[i, 2]

            writer.writerow(row)


def main():
    """
    Transform ANTs points CSV through arbitrary number of transforms.
    """
    parser = argparse.ArgumentParser(
        description='Transform ANTs points CSV through multiple transforms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transform with two transforms, both inverted
  python transform_ants_points.py input.csv output.csv \\
      transform1.mat transform2.mat --useInverse 1 1

  # Transform with RAS coordinate system
  python transform_ants_points.py input.csv output.csv \\
      transform.mat --useInverse 1 --ras

  # Mix of inverted and forward transforms
  python transform_ants_points.py input.csv output.csv \\
      transform1.mat transform2.mat transform3.mat --useInverse 1 0 1
        """
    )

    parser.add_argument('input_csv', type=Path,
                       help='Input CSV file with x,y,z,t columns')
    parser.add_argument('output_csv', type=Path,
                       help='Output CSV file for transformed points')
    parser.add_argument('transforms', nargs='+', type=Path,
                       help='Transform files (.mat or .nii.gz)')
    parser.add_argument('--useInverse', nargs='+', type=int, required=True,
                       help='Use inverse flags for each transform (0 or 1)')
    parser.add_argument('--ras', action='store_true',
                       help='Input/output coordinates are in RAS (default: LPS)')

    args = parser.parse_args()

    # Validate inputs
    if not args.input_csv.exists():
        print(f"Error: Input file not found: {args.input_csv}")
        sys.exit(1)

    if len(args.useInverse) != len(args.transforms):
        print(f"Error: Number of useInverse flags ({len(args.useInverse)}) must match "
              f"number of transforms ({len(args.transforms)})")
        sys.exit(1)

    # Validate useInverse flags are 0 or 1
    for i, flag in enumerate(args.useInverse):
        if flag not in [0, 1]:
            print(f"Error: useInverse flag {i+1} must be 0 or 1, got {flag}")
            sys.exit(1)

    # Verify transform files exist
    for tf in args.transforms:
        if not tf.exists():
            print(f"Error: Transform file not found: {tf}")
            sys.exit(1)

    # Convert useInverse flags to booleans
    invert_flags = [bool(flag) for flag in args.useInverse]

    # Determine coordinate system
    coord_system = "RAS" if args.ras else "LPS"

    # Print header
    print("=" * 70)
    print("Transform ANTs Points CSV")
    print("=" * 70)
    print(f"\nInput CSV:         {args.input_csv}")
    print(f"Output CSV:        {args.output_csv}")
    print(f"Coordinate System: {coord_system}")
    print(f"\nTransforms ({len(args.transforms)} total):")
    for i, (tf, invert) in enumerate(zip(args.transforms, invert_flags)):
        invert_str = "inverted" if invert else "forward"
        print(f"  {i+1}. {tf.name} ({invert_str})")
    print()

    # Load points
    print("Loading points from CSV...")
    original_points, t_values = load_ants_points_csv(args.input_csv)
    print(f"  Loaded {len(original_points)} points")

    if len(original_points) > 0:
        print(f"  First point ({coord_system}): ({original_points[0, 0]:.2f}, "
              f"{original_points[0, 1]:.2f}, {original_points[0, 2]:.2f})")
    print()

    # Apply transforms
    print("Applying transforms...")
    print(f"NOTE: ANTs applies transforms in REVERSE order of the list")
    print(f"      (last transform in list is applied first)")
    print()

    try:
        transformed_points = apply_ants_transforms_to_points(
            points=original_points,
            transform_files=args.transforms,
            use_inverse=invert_flags,
            input_coordinate_system=coord_system
        )

        print(f"✓ Transformation complete! Transformed {len(transformed_points)} points")

        if len(transformed_points) > 0:
            print(f"\n  First point:")
            print(f"    Original ({coord_system}):    ({original_points[0, 0]:.2f}, "
                  f"{original_points[0, 1]:.2f}, {original_points[0, 2]:.2f})")
            print(f"    Transformed ({coord_system}): ({transformed_points[0, 0]:.2f}, "
                  f"{transformed_points[0, 1]:.2f}, {transformed_points[0, 2]:.2f})")

    except Exception as e:
        print(f"\n✗ Error during transformation: {e}")
        sys.exit(1)

    # Save results
    print(f"\nSaving transformed points to: {args.output_csv}")
    save_ants_points_csv(
        args.output_csv,
        transformed_points,
        t_values,
        original_points=original_points
    )

    print("=" * 70)
    print("✓ Done!")
    print("=" * 70)
    print(f"\nOutput file includes:")
    print(f"  - x, y, z: Transformed coordinates ({coord_system})")
    print(f"  - t: Original t values (preserved)")
    print(f"  - x_original, y_original, z_original: Original coordinates")

    if args.ras:
        print(f"\nNote: Coordinates were converted:")
        print(f"  1. RAS → LPS (for ANTs processing)")
        print(f"  2. Applied transforms")
        print(f"  3. LPS → RAS (back to original system)")


if __name__ == "__main__":
    main()
