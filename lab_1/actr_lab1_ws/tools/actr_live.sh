#!/usr/bin/env bash
# Shows live which ACT-R production fires in each cycle of the agent.
# It reads the log of the newest agent run (~/actr_lab1_logs/<time>/events.jsonl).
# Start it after the agent has started (after "READY"). Ctrl+C to stop.
LOGS="$HOME/actr_lab1_logs"

echo "Waiting for the log of the agent in $LOGS ..."
while true; do
  dir="$(ls -td "$LOGS"/*/ 2>/dev/null | head -n 1)"
  if [ -n "$dir" ] && [ -f "${dir}events.jsonl" ]; then break; fi
  sleep 1
done
echo "Reading ${dir}events.jsonl   (Ctrl+C to stop)"
echo

tail -n +1 -f "${dir}events.jsonl" \
  | grep --line-buffered '"kind": "fire"' \
  | sed -u -E 's/.*"t": ([0-9.]+).*"production": "([^"]+)".*"utility": ([-0-9.]+).*/[\1 s]  FIRE  \2   (utility \3)/'
