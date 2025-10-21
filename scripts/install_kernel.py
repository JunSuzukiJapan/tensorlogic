#!/usr/bin/env python3
"""
Install TensorLogic Jupyter Kernel

This script installs the TensorLogic kernel specification
into the Jupyter kernel directory.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def get_kernel_dir(user: bool = True) -> Path:
    """Get the Jupyter kernel installation directory."""
    if user:
        # User-level installation
        if sys.platform == 'win32':
            base = Path(os.environ.get('APPDATA', '~')).expanduser()
        else:
            base = Path('~/.local/share').expanduser()
        return base / 'jupyter' / 'kernels' / 'tensorlogic'
    else:
        # System-level installation
        return Path('/usr/local/share/jupyter/kernels/tensorlogic')


def install_kernel(user: bool = True, prefix: str = None) -> None:
    """
    Install the TensorLogic kernel.

    Args:
        user: Install for current user only
        prefix: Installation prefix (overrides user)
    """
    # Determine installation directory
    if prefix:
        kernel_dir = Path(prefix) / 'share' / 'jupyter' / 'kernels' / 'tensorlogic'
    else:
        kernel_dir = get_kernel_dir(user=user)

    print(f"Installing TensorLogic kernel to: {kernel_dir}")

    # Create kernel directory
    kernel_dir.mkdir(parents=True, exist_ok=True)

    # Get source files
    script_dir = Path(__file__).parent.parent
    jupyter_dir = script_dir / 'jupyter'

    # Copy kernel.json
    kernel_json_src = jupyter_dir / 'kernel.json'
    kernel_json_dst = kernel_dir / 'kernel.json'

    if not kernel_json_src.exists():
        print(f"Error: {kernel_json_src} not found", file=sys.stderr)
        sys.exit(1)

    shutil.copy(kernel_json_src, kernel_json_dst)
    print(f"✓ Copied kernel.json")

    # Copy logos if they exist
    for logo_file in ['logo-32x32.png', 'logo-64x64.png']:
        logo_src = jupyter_dir / logo_file
        if logo_src.exists():
            logo_dst = kernel_dir / logo_file
            shutil.copy(logo_src, logo_dst)
            print(f"✓ Copied {logo_file}")

    print(f"\n✓ TensorLogic kernel installed successfully!")
    print(f"\nTo use the kernel:")
    print(f"  1. Make sure TensorLogic is installed: pip install tensorlogic")
    print(f"  2. Start Jupyter: jupyter notebook")
    print(f"  3. Create a new notebook with TensorLogic kernel")
    print(f"\nTo list all kernels: jupyter kernelspec list")
    print(f"To uninstall: jupyter kernelspec uninstall tensorlogic")


def uninstall_kernel(user: bool = True, prefix: str = None) -> None:
    """
    Uninstall the TensorLogic kernel.

    Args:
        user: Uninstall from current user directory
        prefix: Installation prefix (overrides user)
    """
    if prefix:
        kernel_dir = Path(prefix) / 'share' / 'jupyter' / 'kernels' / 'tensorlogic'
    else:
        kernel_dir = get_kernel_dir(user=user)

    if not kernel_dir.exists():
        print(f"TensorLogic kernel not found at: {kernel_dir}")
        return

    print(f"Uninstalling TensorLogic kernel from: {kernel_dir}")
    shutil.rmtree(kernel_dir)
    print(f"✓ TensorLogic kernel uninstalled successfully!")


def main():
    parser = argparse.ArgumentParser(
        description='Install or uninstall TensorLogic Jupyter kernel'
    )
    parser.add_argument(
        '--user',
        action='store_true',
        default=True,
        help='Install for current user only (default)'
    )
    parser.add_argument(
        '--sys-prefix',
        action='store_true',
        help='Install to sys.prefix (for virtualenvs)'
    )
    parser.add_argument(
        '--prefix',
        help='Installation prefix'
    )
    parser.add_argument(
        '--uninstall',
        action='store_true',
        help='Uninstall the kernel'
    )

    args = parser.parse_args()

    # Determine prefix
    prefix = args.prefix
    if args.sys_prefix:
        prefix = sys.prefix

    # Install or uninstall
    if args.uninstall:
        uninstall_kernel(user=args.user, prefix=prefix)
    else:
        install_kernel(user=args.user, prefix=prefix)


if __name__ == '__main__':
    main()
