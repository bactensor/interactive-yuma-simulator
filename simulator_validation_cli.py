#!/usr/bin/env python3
"""
Thin CLI wrapper for the modular simulator validation package.

This script preserves the original entrypoint while delegating all logic to
`project.simulator_validation` modules.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List

# Ensure Django app path is available
sys.path.insert(0, '/root/repos/interactive-yuma-simulator/app/src')

import django

# Configure Django for standalone execution of this integration script
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')
django.setup()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from project.simulator_validation.runner import validate_simulator, print_validation_results
from project.simulator_validation.hyperparams import (
    get_top_subnets_by_tao_emission,
    fetch_multiple_subnets_hyperparameters,
)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description='Validate Yuma simulator against real metagraph data')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--netuid', type=int, help='Single subnet ID to validate')
    group.add_argument('--netuids', type=str, help='Comma-separated list of subnet IDs (e.g., "1,3,9")')
    group.add_argument('--top-subnets', type=int, metavar='N', help='Validate top N subnets by TAO emission')
    parser.add_argument('--hours', type=float, default=None, help='Hours of data to fetch (default: based on num-epochs)')
    parser.add_argument('--use-epoch-time', action='store_true', help='Time range = num-epochs * 72 minutes (no slack)')
    parser.add_argument('--days-ago', type=int, default=1, help='Days ago to end data fetch (default: 1)')
    parser.add_argument('--tolerance', type=float, default=1e-4, help='Numerical tolerance for comparisons')
    parser.add_argument('--num-epochs', type=int, default=3, help='Number of epochs to validate/fetch')
    # Block-based controls
    parser.add_argument('--start-block', type=int, default=None, help='Start block (overrides start_date)')
    parser.add_argument('--end-block', type=int, default=None, help='End block (overrides end_date)')
    parser.add_argument('--no-diagnostics', action='store_true', help='Disable diagnostic artifact generation on failure')
    parser.add_argument('--bond-penalty-override', type=float, default=None, help='Override bond_penalty (e.g., 1.0)')
    args = parser.parse_args()

    # Resolve subnet IDs
    if args.netuid is not None:
        subnet_ids: List[int] = [args.netuid]
    elif args.top_subnets:
        logger.info(f"Fetching top {args.top_subnets} subnets by TAO emission...")
        subnet_ids = get_top_subnets_by_tao_emission(args.top_subnets)
        if not subnet_ids:
            logger.error("Failed to fetch top subnets from the chain")
            return 1
        logger.info(f"Selected top subnets: {subnet_ids}")
    else:
        try:
            subnet_ids = [int(v.strip()) for v in args.netuids.split(',') if v.strip()]
        except ValueError:
            logger.error(f"Invalid --netuids format: {args.netuids}. Expected comma-separated integers.")
            return 1

    # Determine fetch mode: block-based or date-based
    block_mode = any([args.start_block is not None, args.end_block is not None, args.num_epochs is not None])
    start_date = None
    end_date = None
    if not block_mode:
        # Date-based window
        end_date = datetime.now() - timedelta(days=args.days_ago)
        epoch_minutes = 72
        if args.use_epoch_time:
            total_minutes = args.num_epochs * epoch_minutes
            start_date = end_date - timedelta(minutes=total_minutes)
            logger.info(f"Epoch-based time: {args.num_epochs} * {epoch_minutes} = {total_minutes} min")
        elif args.hours is not None:
            start_date = end_date - timedelta(hours=args.hours)
            logger.info(f"Explicit time: {args.hours} hours")
        else:
            total_minutes = int(args.num_epochs * epoch_minutes * 1.2)
            start_date = end_date - timedelta(minutes=total_minutes)
            logger.info(f"Auto time range: {args.num_epochs} epochs * {epoch_minutes} min * 1.2 = {total_minutes} min")

        logger.info(f"Validating subnets {subnet_ids} from {start_date} to {end_date}")
    else:
        logger.info(
            f"Validating subnets {subnet_ids} using blocks: start_block={args.start_block}, end_block={args.end_block}, num_epochs={args.num_epochs}"
        )

    try:
        # Fetch hyperparams in one batch and reuse for each subnet (even single)
        all_hparams: Dict[int, Dict[str, Any]] = fetch_multiple_subnets_hyperparameters(subnet_ids)
        versions = {n: ('YUMA3' if all_hparams.get(n, {}).get('is_yuma3_on') else 'YUMA2') for n in subnet_ids if n in all_hparams}
        if versions:
            logger.info(f"Yuma versions: {versions}")

        all_results: Dict[int, Any] = {}
        all_success = True
        for nuid in subnet_ids:
            logger.info(f"\n{'='*60}\nVALIDATING SUBNET {nuid}\n{'='*60}")
            try:
                result = validate_simulator(
                    netuid=nuid,
                    tolerance=args.tolerance,
                    num_epochs=args.num_epochs,
                    generate_diagnostics=not args.no_diagnostics,
                    bond_penalty_override=args.bond_penalty_override,
                    hyperparams_data=all_hparams.get(nuid),
                    start_date=start_date,
                    end_date=end_date,
                    start_block=args.start_block,
                    end_block=args.end_block,
                )
                all_results[nuid] = result
                ok = all([
                    result['summary']['bonds_match'],
                    result['summary']['dividends_match'],
                    result['summary']['incentives_match'],
                ])
                if not ok:
                    all_success = False
            except Exception as e:
                logger.error(f"Validation failed for subnet {nuid}: {e}")
                import traceback; traceback.print_exc()
                all_results[nuid] = {'error': str(e), 'success': False}
                all_success = False

        print("\n" + "="*80)
        print("CONSOLIDATED VALIDATION RESULTS")
        print("="*80)
        for nuid, res in all_results.items():
            print(f"\nSUBNET {nuid}:")
            if 'error' in res:
                print(f"  ❌ FAILED: {res['error']}")
            else:
                print_validation_results(res)
        print(f"\n{'='*80}")
        print("OVERALL SUMMARY:")
        successful = [n for n, r in all_results.items() if 'error' not in r and all([
            r['summary']['bonds_match'], r['summary']['dividends_match'], r['summary']['incentives_match']
        ])]
        failed = [n for n in subnet_ids if n not in successful]
        print(f"Successful subnets: {successful}")
        print(f"Failed subnets: {failed}")
        print(f"Overall success: {'✓' if all_success else '✗'}")
        return 0 if all_success else 1
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        import traceback; traceback.print_exc()
        return 2


if __name__ == '__main__':
    sys.exit(main())
