#!/usr/bin/env python3
"""
ISP Version Comparator CLI
========================

Command-line interface for ISP version comparison.

Usage:
    python -m isp_compare.cli --help
    python -m isp_compare.cli compare --raw-a v1.raw --raw-b v2.raw --golden golden.jpg

Author: ISP Team
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from isp_compare import __version__
from isp_compare.core import (
    Comp12Parser, Comp12Config,
    CModelISP,
    ISPComparator, ComparisonConfig,
    ImageMetrics
)
from isp_compare.reports.pdf_generator import PDFReportGenerator
from isp_compare.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def cmd_compare(args):
    """Compare two ISP versions"""
    logger.info("=" * 60)
    logger.info("ISP Version Comparison")
    logger.info("=" * 60)
    
    # Build configuration
    config = ComparisonConfig(
        comp12_width=args.width,
        comp12_height=args.height,
        comp12_pattern=args.pattern,
        cmodel_path=args.cmodel,
        cmodel_threads=args.threads,
        cmodel_params={} if not args.params else dict(p.split("=") for p in args.params),
        traffic_light_roi=None,  # TODO: add ROI parsing
        output_dir=args.output,
        save_intermediate=args.save_intermediate
    )
    
    # Initialize comparator
    comparator = ISPComparator(config)
    
    # Run comparison
    result = comparator.compare_versions(
        version_a_raw=args.raw_a,
        version_b_raw=args.raw_b,
        golden_path=args.golden,
        version_a_label=args.label_a or "Version A",
        version_b_label=args.label_b or "Version B"
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON RESULT")
    print("=" * 60)
    print(f"Status: {result.overall_status}")
    print(f"Summary: {result.summary}")
    
    if result.recommendations:
        print("\nRecommendations:")
        for rec in result.recommendations:
            print(f"  • {rec}")
    
    if result.comparison:
        print("\nMetrics Comparison:")
        print(f"  Version A Score: {result.comparison.get('a_score', 0):.1f}")
        print(f"  Version B Score: {result.comparison.get('b_score', 0):.1f}")
        print(f"  Score Delta: {result.comparison.get('score_delta', 0):+.1f}")
    
    print(f"\nProcessing Time: {result.processing_time_ms:.1f} ms")
    
    # Save result
    result_path = comparator.save_result(result)
    print(f"\nResult saved: {result_path}")
    
    # Generate PDF report
    if args.pdf:
        generator = PDFReportGenerator()
        images = {}
        if result.version_a_result:
            images["version_a"] = result.version_a_result.output_path
        if result.version_b_result:
            images["version_b"] = result.version_b_result.output_path
        
        pdf_path = generator.generate(result, args.pdf, images)
        print(f"PDF Report: {pdf_path}")
    
    return 0 if result.overall_status != "error" else 1


def cmd_parse(args):
    """Parse Comp12 RAW file"""
    print(f"Parsing Comp12 RAW: {args.input}")
    
    parser = Comp12Parser(Comp12Config(
        width=args.width,
        height=args.height,
        pattern=args.pattern
    ))
    
    try:
        raw16 = parser.parse(args.input)
        print(f"Successfully parsed Comp12 RAW")
        print(f"  Shape: {raw16.shape}")
        print(f"  Dtype: {raw16.dtype}")
        print(f"  Range: [{raw16.min()}, {raw16.max()}]")
        
        if args.output:
            parser.save_for_cmodel(raw16, args.output)
            print(f"Saved to: {args.output}")
        
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_metrics(args):
    """Calculate image metrics"""
    print(f"Calculating metrics for: {args.image}")
    
    metrics = ImageMetrics()
    
    try:
        result = metrics.evaluate(
            args.image,
            reference_path=args.reference,
            traffic_light_roi=None  # TODO: add ROI
        )
        
        print(f"\nMetrics Result:")
        print(f"  Overall Score: {result.overall_score:.1f}")
        print(f"  Sharpness: {result.sharpness_score:.1f}")
        print(f"  Noise: {result.noise_score:.1f}")
        print(f"  Color: {result.color_score:.1f}")
        print(f"  Traffic Light: {result.traffic_light_score:.1f}")
        print(f"  Passed: {result.passed}")
        
        if result.issues:
            print(f"\nIssues:")
            for issue in result.issues:
                print(f"  • {issue}")
        
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_batch(args):
    """Batch process multiple RAW files"""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all RAW files
    raw_files = list(input_dir.glob("*.raw"))
    if not raw_files:
        print(f"No .raw files found in {input_dir}")
        return 1
    
    print(f"Found {len(raw_files)} RAW files")
    print(f"Output directory: {output_dir}")
    
    # Initialize CModel
    cmodel = CModelISP(args.cmodel, num_threads=args.threads)
    
    # Batch process
    print("Processing...")
    results = cmodel.batch_process(
        [str(f) for f in raw_files],
        str(output_dir),
        params={} if not args.params else dict(p.split("=") for p in args.params)
    )
    
    # Summary
    success = sum(1 for r in results if r.success)
    print(f"\nCompleted: {success}/{len(results)} successful")
    
    return 0 if success == len(results) else 1


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="ISP Version Comparator",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--version", action="version", version=f"isp-compare {__version__}")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two ISP versions")
    compare_parser.add_argument("--raw-a", required=True, help="Version A RAW file (Comp12)")
    compare_parser.add_argument("--raw-b", required=True, help="Version B RAW file (Comp12)")
    compare_parser.add_argument("--golden", help="Golden reference image")
    compare_parser.add_argument("--label-a", help="Label for Version A")
    compare_parser.add_argument("--label-b", help="Label for Version B")
    compare_parser.add_argument("--cmodel", required=True, help="Path to CModel executable")
    compare_parser.add_argument("--width", type=int, default=3840, help="RAW width (default: 3840)")
    compare_parser.add_argument("--height", type=int, default=2160, help="RAW height (default: 2160)")
    compare_parser.add_argument("--pattern", default="RGGB", choices=["RGGB", "BGGR", "GRBG", "GBRG"], help="Bayer pattern")
    compare_parser.add_argument("--threads", type=int, default=8, help="CPU threads (default: 8)")
    compare_parser.add_argument("--params", nargs="+", help="CModel parameters (key=value)")
    compare_parser.add_argument("--output", default="./output", help="Output directory")
    compare_parser.add_argument("--save-intermediate", action="store_true", help="Save intermediate files")
    compare_parser.add_argument("--pdf", help="Generate PDF report to specified path")
    compare_parser.set_defaults(func=cmd_compare)
    
    # Parse command
    parse_parser = subparsers.add_parser("parse", help="Parse Comp12 RAW file")
    parse_parser.add_argument("input", help="Input Comp12 RAW file")
    parse_parser.add_argument("--width", type=int, default=3840, help="RAW width")
    parse_parser.add_argument("--height", type=int, default=2160, help="RAW height")
    parse_parser.add_argument("--pattern", default="RGGB", choices=["RGGB", "BGGR", "GRBG", "GBRG"])
    parse_parser.add_argument("--output", "-o", help="Output file for RAW16")
    parse_parser.set_defaults(func=cmd_parse)
    
    # Metrics command
    metrics_parser = subparsers.add_parser("metrics", help="Calculate image metrics")
    metrics_parser.add_argument("image", help="Input image file")
    metrics_parser.add_argument("--reference", "-r", help="Reference image for PSNR/SSIM")
    metrics_parser.set_defaults(func=cmd_metrics)
    
    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Batch process RAW files")
    batch_parser.add_argument("input_dir", help="Input directory with RAW files")
    batch_parser.add_argument("output", help="Output directory")
    batch_parser.add_argument("--cmodel", required=True, help="Path to CModel executable")
    batch_parser.add_argument("--threads", type=int, default=8, help="CPU threads")
    batch_parser.add_argument("--params", nargs="+", help="CModel parameters (key=value)")
    batch_parser.set_defaults(func=cmd_batch)
    
    args = parser.parse_args()
    
    # Setup logging
    if args.debug:
        setup_logging(level=10)  # DEBUG
    
    # Execute command
    if hasattr(args, 'func'):
        return args.func(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
