#!/usr/bin/env bash
# dropcache_loop.sh LOG | --now [LOG]: Inkling's page-cache dropper on one node. serve.sh starts it while the
# server runs, as the user units inkling-serve-dropcache (head) and inkling-serve-dropcache-r1 (worker); `dropcache start|stop`.
#   LOG      every DROP_EVERY_S (30): when the node's reclaimable file cache -- Active(file) + Inactive(file) of
#            /proc/meminfo; Shmem cannot be dropped -- exceeds DROP_MIN_MIB (512), drop it; one line a drop to LOG
#   --now    one drop now, whatever the cache holds (the serve's drops before the load and right after READY): its
#            line on stdout, and appended to LOG when given
# A drop runs DROP_HELPER with sudo -n, /usr/local/sbin/dropcache-helper, root-owned
# and argument-less (sync; echo 3 > /proc/sys/vm/drop_caches), allowed NOPASSWD on both nodes. Where sudo -n refuses,
# drop_cache.py runs instead (the sudoless fadvise of every file of at least 1 MiB in the models and artifacts; ~25 s).
set -u
export LC_ALL=C
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
EVERY=${DROP_EVERY_S:-30}
MIN=${DROP_MIN_MIB:-512}
HELPER=${DROP_HELPER:-/usr/local/sbin/dropcache-helper}

mem() {  # MemFree, MemAvailable, Active(file) + Inactive(file), in MiB
  awk '/^MemFree:/ {f = $2} /^MemAvailable:/ {a = $2} /^Active\(file\):/ {x = $2} /^Inactive\(file\):/ {y = $2}
       END {printf "%d %d %d\n", f / 1024, a / 1024, (x + y) / 1024}' /proc/meminfo
}

drop() {  # one drop and its line
  local f0 a0 c0 f1 a1 c1 t0 how
  read -r f0 a0 c0 < <(mem)
  t0=$EPOCHREALTIME
  if sudo -n "$HELPER" > /dev/null 2>&1; then
    how=helper
  else
    python3 -I "$HERE/drop_cache.py" > /dev/null 2>&1
    how="drop_cache.py: sudo -n refused"
  fi
  read -r f1 a1 c1 < <(mem)
  printf '%s %s drop (%s, %.2f s): file cache %d -> %d MiB, MemFree %d -> %d MiB, MemAvailable %d -> %d MiB\n' \
    "$(date '+%F %T')" "$(hostname -s)" "$how" "$(awk -v a="$t0" -v b="$EPOCHREALTIME" 'BEGIN {print b - a}')" \
    "$c0" "$c1" "$f0" "$f1" "$a0" "$a1"
}

if [ "${1:-}" = --now ]; then
  line=$(drop)
  echo "$line"
  if [ -n "${2:-}" ]; then mkdir -p "$(dirname "$2")" && echo "$line" >> "$2"; fi
  exit 0
fi
LOG=${1:?usage: dropcache_loop.sh LOG | --now [LOG]}
mkdir -p "$(dirname "$LOG")"
echo "$(date '+%F %T') $(hostname -s) dropper: every ${EVERY} s when the file cache exceeds ${MIN} MiB ($HELPER)" >> "$LOG"
while :; do
  read -r _ _ cache < <(mem)
  if [ "$cache" -gt "$MIN" ]; then drop >> "$LOG"; fi
  sleep "$EVERY"
done
