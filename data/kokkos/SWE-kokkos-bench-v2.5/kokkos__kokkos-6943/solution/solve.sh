#!/usr/bin/env bash
set -euo pipefail
cd /workspace/repo
git apply --whitespace=nowarn /solution/gold.patch
