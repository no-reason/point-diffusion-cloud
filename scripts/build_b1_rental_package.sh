#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
destination="${1:-../badPVD-B1-minimal.tar.gz}"
# COPYFILE_DISABLE avoids macOS AppleDouble ``._*`` entries and xattr headers;
# the archive is intended for Linux AutoDL hosts.
COPYFILE_DISABLE=1 tar --no-xattrs -czf "$destination" \
  --exclude=.git --exclude=.venv --exclude='*.pyc' --exclude='__pycache__' --exclude='.pytest_cache' \
  --exclude='outputs*' --exclude='*.pth' --exclude='*.pt' --exclude='*.ckpt' \
  --exclude='*.h5' --exclude='*.hdf5' --exclude='*.npz' \
  --exclude='coordinate_audit/runtime_validation' --exclude='assets/*.gif' \
  --exclude='*.log' --exclude='*.pid' .
echo "$destination"
