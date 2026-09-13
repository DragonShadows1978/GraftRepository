#!/bin/bash
S=/tmp/claude-1000/-home-vader/9440b5e0-40c2-4d02-8b46-72aff61c5333/scratchpad
LOG=$S/gpu_queue23.log
idle_wait() { for k in $(seq 1 240); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${u:-99999}" -lt 600 ] && return 0; sleep 30; done; return 1; }
echo "QUEUE23 START $(date -Is) — XM2 Qwen final-channel cells (amendment 1)" >> $LOG
cd /mnt/ForgeRealm/wt/grm-xm2
grep -vE "^\s*#|^\s*$" artifacts/grm_xm2/lead_commands.txt | grep -- "--run" | grep -v "C3l__sup_harbor_restatement" | grep -v "C5__sup_praxis_fresh" | while read -r cmd; do
  idle_wait || { echo "card busy" >> $LOG; exit 1; }
  cell=$(echo "$cmd" | grep -oE "\-\-cell [A-Za-z0-9_+-]+" | cut -d' ' -f2)
  pre=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  timeout --kill-after=5s 590s bash -c "$cmd" > artifacts/grm_xm2/lead_a1_${cell}.log 2>&1; rc=$?
  st=$(grep -oE '"status": "[A-Z_]+"' artifacts/grm_xm2/lead_a1_${cell}.log | tail -1)
  echo "$(date +%H:%M:%S) $cell rc=$rc $st pre_mib=$pre" >> $LOG
  echo "$st" | grep -q ERROR && { echo "STOP on first ERROR; $(grep -oE '"error": "[^"]{0,160}' artifacts/grm_xm2/run/cells/$cell.json 2>/dev/null | tail -1)" >> $LOG; break; }
  sleep 175
done
echo "QUEUE23 DONE $(date -Is)" >> $LOG
