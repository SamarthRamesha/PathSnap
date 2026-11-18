#!/bin/sh
set -eu
cd -- "$(dirname -- "$0")"

demo_python=.venv-demo/bin/python
if [ ! -x "$demo_python" ]; then
    echo 'Create .venv-demo with Python 3.12 and install requirements.txt. See DEMO_GUIDE.md.' >&2
    exit 1
fi

if [ "$(uname -s)" = Darwin ] && command -v brew >/dev/null 2>&1; then
    omp_prefix=$(brew --prefix libomp 2>/dev/null || :)
    if [ -f "$omp_prefix/lib/libomp.dylib" ]; then
        export DYLD_LIBRARY_PATH="$omp_prefix/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
    fi
fi

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PATHSNAP_EVALUATOR_DIR="${PATHSNAP_EVALUATOR_DIR:-data/student_test}"
exec "$demo_python" -m scripts.start_demo
