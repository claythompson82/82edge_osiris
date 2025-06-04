import argparse
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Parses command-line arguments and executes tasks based on the --task-set argument.
    """
    parser = argparse.ArgumentParser(description="Nightly trainer driver script.")
    parser.add_argument(
        "--task-set",
        type=str,
        required=True,
        help="The set of tasks to run (e.g., 'marketforge')."
    )

    args = parser.parse_args()

    if args.task_set == "marketforge":
        logging.info("trainer loop stub")
    else:
        logging.warning(f"Unknown task set: {args.task_set}")

if __name__ == "__main__":
    main()
