#!/bin/bash
# Single-shot pipeline progress reporter. Reads logs/, prints one summary.
cd ~/rl-adaptive-filtering

if [ ! -f logs/run.pid ]; then echo "not launched"; exit 1; fi
PID=$(cat logs/run.pid)
if ! kill -0 "$PID" 2>/dev/null; then
  if [ -f logs/done.flag ]; then echo "DONE: $(cat logs/done.flag)"; exit 0
  else echo "process dead but no done.flag — check logs/pipeline.log"; exit 1; fi
fi

# Stage detection by which log file is currently being written to.
stages=("train_meta:Meta-RL training:1500000:5"
        "train_mlp:PPO-MLP training:800000:5"
        "train_meta_af:Meta-AF BPTT:4000:1"
        "ablate:Ablations:300000:7"
        "eval:Evaluation:1:1")

current=""
for s in "${stages[@]}"; do
  IFS=':' read -r tag name total seeds <<< "$s"
  f="logs/${tag}.log"
  if [ -f "$f" ] && [ "$(find "$f" -mmin -2 2>/dev/null)" ]; then
    current="$tag:$name:$total:$seeds"
    break
  fi
done

if [ -z "$current" ]; then
  # find the most recently modified log
  newest=$(ls -t logs/*.log 2>/dev/null | head -1)
  echo "PID $PID alive. last activity: $newest ($(stat -c %y "$newest" | cut -d. -f1))"
  exit 0
fi

IFS=':' read -r tag name total seeds <<< "$current"
log="logs/${tag}.log"

# How many seeds finished?
done_seeds=$(grep -c '\[seed .* saved\|saved ->' "$log" 2>/dev/null)
done_seeds=${done_seeds:-0}

# Current step inside the active seed: parse last "total_timesteps" line.
last_step=$(grep -oE 'total_timesteps *\| *[0-9]+' "$log" 2>/dev/null | tail -1 \
            | grep -oE '[0-9]+$')
last_step=${last_step:-0}

# Stage progress = (finished_seeds * total + current_step) / (seeds * total)
overall_total=$((seeds * total))
overall_done=$((done_seeds * total + last_step))
if [ "$overall_total" -gt 0 ]; then
  pct=$(awk "BEGIN {printf \"%.1f\", 100.0 * $overall_done / $overall_total}")
else
  pct="?"
fi

# Rough ETA from log start time
start_epoch=$(stat -c %W "$log" 2>/dev/null)
[ "$start_epoch" = "-" ] || [ -z "$start_epoch" ] || [ "$start_epoch" = "0" ] && \
  start_epoch=$(stat -c %Y logs/run.pid 2>/dev/null)
now_epoch=$(date +%s)
elapsed=$((now_epoch - start_epoch))
if [ "$overall_done" -gt 0 ] && [ "$elapsed" -gt 30 ]; then
  rate=$(awk "BEGIN {print $overall_done / $elapsed}")
  remaining=$((overall_total - overall_done))
  eta_sec=$(awk "BEGIN {printf \"%d\", $remaining / $rate}")
  eta_h=$((eta_sec / 3600))
  eta_m=$(((eta_sec % 3600) / 60))
  eta="~${eta_h}h ${eta_m}m"
else
  eta="warming up"
fi

# Stage index in pipeline
case "$tag" in
  train_meta)    idx="1/5" ;;
  train_mlp)     idx="2/5" ;;
  train_meta_af) idx="3/5" ;;
  ablate)        idx="4/5" ;;
  eval)          idx="5/5" ;;
esac

printf "stage %s: %-24s | seed %d/%d  step %d/%d  =  %s%%  | stage ETA %s\n" \
       "$idx" "$name" "$done_seeds" "$seeds" "$last_step" "$total" "$pct" "$eta"
