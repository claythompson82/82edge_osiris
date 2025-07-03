from __future__ import annotations
import argparse
import asyncio
import logging

from . import meta_loop


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="DGM Kernel")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run the meta-loop only once.",
    )
    args = parser.parse_args()

    if args.once:
        logging.getLogger(__name__).info("Running DGM meta-loop once.")
        asyncio.run(meta_loop.loop_once())
        logging.getLogger(__name__).info("DGM meta-loop (once) finished.")
    else:
        logging.getLogger(__name__).info(
            "Starting DGM meta-loop to run continuously."
        )
        asyncio.run(meta_loop.meta_loop())


if __name__ == "__main__":
    main()
