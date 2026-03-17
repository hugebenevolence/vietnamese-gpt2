#!/usr/bin/env python3
"""Shared utility functions."""

import unicodedata


def normalize_text(text: str) -> str:
    """Apply Unicode NFC normalization, returning empty string for None."""
    if text is None:
        return ""
    return unicodedata.normalize("NFC", text)


def format_size(num_bytes: int) -> str:
    """Format byte count as human-readable string (B/KB/MB/GB/TB)."""
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"
