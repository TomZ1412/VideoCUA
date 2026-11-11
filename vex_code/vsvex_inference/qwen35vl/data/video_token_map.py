"""Dataset-independent video -> token-count mappings with mean fallback."""

import json
import logging
import math
import os


def video_entries(sample):
    if "tos_key" in sample or "video" in sample:
        field = "tos_key" if "tos_key" in sample else "video"
        yield {field: sample[field]}
    else:
        for entry in sample.get("videos", []):
            field = "video" if "video" in entry else "tos_key"
            yield {field: entry[field]}


def map_header():
    return {"version": 2, "format": "video_token_map"}


def entry_key(entry):
    field = "video" if "video" in entry else "tos_key"
    location = entry[field]
    if not isinstance(location, str) or not location:
        raise ValueError("Video source must be a nonempty string")
    if field == "video":
        location = os.path.abspath(location)
    return field, location


def make_row(entry, tokens):
    field, location = entry_key(entry)
    return {field: location, "num_tokens": int(tokens)}


class VideoTokenMap:
    def __init__(self, header, rows):
        if header != map_header():
            raise ValueError("Expected a version 2 video_token_map cache")
        self.header = header
        self.rows = {}
        for row in rows:
            if ("video" in row) == ("tos_key" in row):
                raise ValueError("Each mapping must have exactly one video or tos_key")
            tokens = row.get("num_tokens")
            if type(tokens) is not int or tokens <= 0:
                raise ValueError("Video token counts must be positive integers")
            key = entry_key(row)
            if key in self.rows and self.rows[key]["num_tokens"] != tokens:
                raise ValueError(f"Conflicting token counts for {key}")
            self.rows[key] = make_row(row, tokens)
        self.mean = max(1, math.ceil(sum(row["num_tokens"] for row in self.rows.values()) / len(self.rows))) if self.rows else 1

    @classmethod
    def read(cls, path):
        with open(path) as handle:
            header = json.loads(next(handle))
            return cls(header, (json.loads(line) for line in handle if line.strip()))

    def lengths_for(self, samples):
        matched, missing, no_video = 0, 0, 0
        lengths = []
        for sample in samples:
            entries = list(video_entries(sample))
            length = 0
            for entry in entries:
                row = self.rows.get(entry_key(entry))
                if row is None:
                    missing += 1
                    length += self.mean
                else:
                    matched += 1
                    length += row["num_tokens"]
            if not entries:
                no_video += 1
                length = self.mean
            lengths.append(length)
        logging.warning("Video token map: %d matched occurrences, %d misses, %d samples without video; mean fallback=%d",
                        matched, missing, no_video, self.mean)
        return lengths
