import argparse


def main():
    parser = argparse.ArgumentParser(description="AZR module CLI.")
    # Add arguments here in the future
    parser.add_argument("--config", type=str, help="Path to config file.")
    args = parser.parse_args()
    print(f"AZR CLI stub. Config: {args.config}")


if __name__ == "__main__":
    main()
