#!/bin/bash
# Resumable LoRA queue: each line = protocol fold seed groups; a finished run (json exists) is skipped.
cd $(dirname "$0")/..
LOG=results/lora_queue.log
while read -r p f s g; do
  [ -z "$p" ] && continue
  gt=$([ "$g" = "group_strict" ] && echo strict || echo phash); ft=$([ "$f" = "x" ] && echo x || echo $f)
  out="results/lora_${p}_${gt}_f${ft}_s${s}.json"
  if [ -f "$out" ]; then echo "SKIP $out" >> $LOG; continue; fi
  fa=$([ "$f" = "x" ] && echo "" || echo "--fold $f")
  echo "=== START $p fold=$f seed=$s groups=$g $(date '+%F %T')" >> $LOG
  python -I -u code/train_lora.py $p $fa --seed $s --groups $g --epochs 8 --bs 16 2>&1 \
    | grep --line-buffered -E "epoch [0-9]+/|Error|Traceback|MemoryError|Killed" >> $LOG
  [ -f "$out" ] && echo "=== END $p f=$f s=$s OK $(date '+%F %T')" >> $LOG || echo "=== END $p f=$f s=$s FAILED $(date '+%F %T')" >> $LOG
done < code/queue.txt
echo "QUEUE_FINISHED $(date '+%F %T')" >> $LOG
