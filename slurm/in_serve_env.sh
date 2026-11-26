#!/bin/bash
export LD_LIBRARY_PATH=/opt/venvs/serve/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
exec "$@"
