"""Upload one completed local checkpoint to HDFS and retain the newest ones."""

import argparse
import os
import re

from cruise.utilities.hdfs_io import hcopy, hlist_files, hmkdir, hrm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("remote_dir")
    parser.add_argument("--save_total_limit", type=int, default=5)
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)
    if not hmkdir(args.remote_dir):
        raise RuntimeError(f"Could not create HDFS directory {args.remote_dir}")
    if not hcopy(args.checkpoint, args.remote_dir):
        raise RuntimeError(f"Could not upload {args.checkpoint} to {args.remote_dir}")
    print(f"Uploaded {args.checkpoint} to {args.remote_dir}", flush=True)

    if args.save_total_limit > 0:
        checkpoints = []
        for path in hlist_files([args.remote_dir]):
            match = re.fullmatch(r"checkpoint-(\d+)", os.path.basename(path.rstrip("/")))
            if match:
                checkpoints.append((int(match.group(1)), path))
        checkpoints.sort()
        for _, path in checkpoints[:-args.save_total_limit]:
            if not hrm(path):
                raise RuntimeError(f"Could not remove old HDFS checkpoint {path}")
            print(f"Removed old HDFS checkpoint {path}", flush=True)


if __name__ == "__main__":
    main()
