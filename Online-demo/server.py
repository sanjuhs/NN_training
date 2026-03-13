#!/usr/bin/env python3
"""
Serve the repository root so the Online-demo pages can load sibling assets.
"""

from __future__ import annotations

import argparse
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the NN_training repo for the Online-demo pages."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    server = ThreadingHTTPServer((args.host, args.port), SimpleHTTPRequestHandler)

    print("=== Online Demo Server ===")
    print(f"Serving repo root: {repo_root}")
    print(f"Address: http://{args.host}:{args.port}")
    print(
        f"Transformer page: http://{args.host}:{args.port}/Online-demo/transformer-model.html"
    )
    print(f"Comparison page: http://{args.host}:{args.port}/Online-demo/comparison.html")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
