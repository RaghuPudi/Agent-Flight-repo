"""
START SERVER — run this from anywhere inside atfm-agent/
Usage:  python start_server.py
        python start_server.py --port 8080
        python start_server.py --reload   (auto-reload on code changes)
"""

import sys
import os
import argparse

# Always set the working directory to atfm-agent/ before starting
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

# Ensure src/ is on the path
SRC = os.path.join(HERE, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Create required data directories
os.makedirs(os.path.join(HERE, "data", "audit_logs"), exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start ATFM Negotiation Agent OCC Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("uvicorn not installed. Run:  pip install uvicorn[standard]")
        sys.exit(1)

    print(f"\n  ✈  ATFM Negotiation Agent — OCC Dashboard")
    print(f"  ─────────────────────────────────────────")
    print(f"  URL     : http://{args.host}:{args.port}")
    print(f"  API docs: http://{args.host}:{args.port}/docs")
    print(f"  Reload  : {'yes' if args.reload else 'no'}")
    print(f"  Logs    : {os.path.join(HERE, 'data', 'audit_logs')}")
    print(f"\n  Press Ctrl+C to stop.\n")

    uvicorn.run(
        "api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        app_dir=SRC,          # uvicorn resolves 'api.server' relative to this directory
    )
