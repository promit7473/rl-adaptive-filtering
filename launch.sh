#!/bin/bash
set -e
cd "$(dirname "$0")"
source .venv/bin/activate

echo "=== preflight ==="
python3 -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None (CPU fallback)')"
mkdir -p logs results

if [ -f logs/run.pid ] && kill -0 "$(cat logs/run.pid)" 2>/dev/null; then
  echo "already running (PID $(cat logs/run.pid)). tail logs/pipeline.log to watch."
  exit 1
fi

echo "=== launching v3.1 production pipeline in background ==="
nohup bash -c '
set -eo pipefail
source .venv/bin/activate
PYTHONPATH=. python3 scripts/_run_pipeline.py 2>&1 | tee logs/pipeline.log
echo "ALL DONE $(date)" | tee logs/done.flag
' > logs/pipeline_outer.log 2>&1 &

echo $! > logs/run.pid
echo ""
echo "=== launched. PID $(cat logs/run.pid). ==="
echo "  watch live:        tail -f logs/pipeline.log"
echo "  current status:    tail -f logs/pipeline_outer.log"
echo "  GPU usage:         watch -n 2 nvidia-smi"
echo "  done flag:         ls logs/done.flag"
echo "  kill if needed:    kill \$(cat logs/run.pid)"
