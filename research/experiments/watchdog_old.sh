#!/usr/bin/env bash
# Runs on the always-up second box. Waits for the calibrated host to come back,
# then resumes its queue there (pipeline validation first, then campaign A).
set -u
OLD="root@cpod-1wqzm4msan2q-s3.podtcp.compshare.cn"
OPTS="-i /root/.ssh/hostlink -o StrictHostKeyChecking=no -o ConnectTimeout=12 -p 22304"
LOG=${LOG:-/root/watchdog.log}
while true; do
  if ssh $OPTS $OLD 'echo up' >/dev/null 2>&1; then
    echo "$(date -Is) old host is back" >>"$LOG"
    ssh $OPTS $OLD 'cd /workspace/organizer-projects/FFSplat-Hackathon && rm -rf records/artifacts-20261030 && git pull -q --ff-only && mkdir -p outputs/experiments && nohup setsid bash research/experiments/round8_pipeline.sh > outputs/pipeline-launch.log 2>&1 < /dev/null & nohup setsid bash research/experiments/campaign_a.sh > outputs/campaignA-launch.log 2>&1 < /dev/null & echo relaunched' >>"$LOG" 2>&1
    break
  fi
  sleep 60
done
