#!/usr/bin/env bash
# Resumable, idempotent downloader for Abel et al. datasets (OSF). Re-run anytime; it resumes.
# Run detached:  setsid nohup ./download_datasets.sh >> data/download.log 2>&1 &
cd "$(dirname "$0")/data" || exit 1
declare -A FILES=(
  [Empkins_Dataset.zip]="sh3xn"
  [Guardian_Dataset.zip]="gyh75"
)
for name in "${!FILES[@]}"; do
  node=${FILES[$name]}
  meta=$(until curl -gfsSL --max-time 60 "https://api.osf.io/v2/nodes/$node/files/osfstorage/?filter[name]=$name"; do sleep 10; done)
  read -r url size md5 < <(python3 -I -c '
import sys,json
a=json.load(sys.stdin)["data"][0]
print(a["links"]["download"],a["attributes"]["size"],a["attributes"]["extra"]["hashes"]["md5"])' <<<"$meta")
  until [ "$(stat -c%s "$name" 2>/dev/null || echo 0)" -eq "$size" ]; do
    echo "$(date) downloading $name ($size bytes)"
    curl -fL -C - --retry 20 --retry-delay 10 --retry-all-errors --connect-timeout 30 -o "$name" "$url" || sleep 15
  done
  if [ "$(md5sum "$name" | cut -d' ' -f1)" = "$md5" ]; then echo "$(date) OK $name md5 verified"
  else echo "$(date) MD5 MISMATCH $name, redownloading"; rm -f "$name"; exec "$0"; fi
  [ -d "${name%.zip}" ] || { unzip -q -o "$name" -d "${name%.zip}" && echo "$(date) unzipped $name"; }
done
echo "$(date) ALL DONE"
