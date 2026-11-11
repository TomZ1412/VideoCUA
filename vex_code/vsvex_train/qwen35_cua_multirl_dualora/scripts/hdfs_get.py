"""Copy a model, checkpoint, or dataset from HDFS into a local directory."""

import argparse
import os

from cruise.utilities.hdfs_io import hcopy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("local_parent")
    args = parser.parse_args()
    if not args.source.startswith("hdfs://"):
        raise ValueError("source must be an HDFS URI")
    os.makedirs(args.local_parent, exist_ok=True)
    if not hcopy(args.source, args.local_parent):
        raise RuntimeError(f"Could not copy {args.source} into {args.local_parent}")


if __name__ == "__main__":
    main()
