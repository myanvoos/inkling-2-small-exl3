#!/usr/bin/env bash
# serve.sh — Inkling-2-Small (EXL3 2.59bpw) on the native TensorFold inkling2 server, TP=2:
# rank 0 on the head node (127.0.0.1:8888), rank 1 on the worker (:8889), with the inkling gateway on :8080 in front.
#   scripts/serve.sh start | stop | restart | status | logs | gateway | check | dropcache start|stop|status
# start: refuses at once while anything else holds the GPUs or the ports (another engine, another gateway, a GPU
#   process outside the lock), then runs the serve as the systemd user unit inkling-serve, so it outlives the shell
#   or agent session that started it, and follows its log until it serves or fails (Ctrl-C stops following, not the
#   start). The serve takes the GPU pair lock (the turnstile, then gpu.lock; queued jobs run first) and HOLDS it
#   while the server runs, so no GPU job or build starts beside it; drops the page cache on both nodes; checks idle
#   GPUs, memory and the files on both nodes; starts both ranks (rank 1 as the worker's user unit inkling-serve-r1);
#   waits for /v1/models; then (re)creates the gateway.
# stop: the gateway (only when it is this stack's), both ranks, then the lock, then the units. It never stops another
#   model or a GPU job, and kills only processes running this stack's exact binary.
# gateway: (re)creates only the gateway in front of the serving engine (after a gateway/ change).
# check: start's preflight alone (read-only): what a start would refuse right now, and why.
# dropcache start|stop|status: the page-cache dropper units by hand (the serve starts them after READY, stop stops them).
# Knobs: config/inkling.env (for one start: CONTEXT=..., MIN_AVAIL_GIB=..., LOCK_WAIT_S=..., READY_S=...).
set -uo pipefail
DEP=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
ENV_FILE=${ENV_FILE:-$DEP/config/inkling.env}
set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a
RECIPE=${TF_RECIPE:-inkling-2-small-exl3}
LABEL=com.inkling.serve
STATE=$LOGDIR/state
PID0=$LOGDIR/r0.pid
HOLDER=$GPU_LOCK_DIR/.inkling-serve-holder      # the lock holder's pid from its start (waiting or holding)
HELD=$GPU_LOCK_DIR/.inkling-serve-lock-held     # the holder's pid while it holds the pair lock
RELEASE=$GPU_LOCK_DIR/.inkling-serve-release    # asks the holder to let go
API=http://127.0.0.1:$PORT/v1
GW_API=http://127.0.0.1:$GATEWAY_PORT/v1
ENVS="${NCCL_LIB:+LD_LIBRARY_PATH=$NCCL_LIB }NCCL_COMM_ID=$NCCL_COMM_ID NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME NCCL_IB_HCA=$NCCL_IB_HCA NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX NCCL_IB_ADDR_FAMILY=AF_INET NCCL_DEBUG=WARN NCCL_SOCKET_RETRY_CNT=600 TF_INKLING2_FLOOR_GIB=$TF_INKLING2_FLOOR_GIB $MIX_ENV $ROWS_ENV $EXTRA_ENV"
ARGS="--chat-template $CHAT_TEMPLATE --context $CONTEXT --max-tokens $MAX_TOKENS --parallel $PARALLEL --name $SERVED_NAME --no-update-check --tp 2 --master $HEAD_IP"
mkdir -p "$LOGDIR" "$DEPLOY/state"   # the state dir holds the rows' shards and the mixed store's bank history

log() { echo "[inkling $(date +%H:%M:%S)] $*"; }
state() { echo "$* ($(date '+%F %T'))" > "$STATE"; }
wssh() { ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "$@"; }
on_worker() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" bash -s; }   # a script on stdin
listening() { ss -ltnH "sport = :$1" 2>/dev/null | grep -q .; }
served() { curl -fsS --max-time 5 "$1/models" 2>/dev/null | grep -q "\"id\": *\"$2\""; }
engine_ok() { served "$API" "$SERVED_NAME"; }
gateway_ok() { served "$GW_API" "$GATEWAY_MODEL"; }
ours() { [ -n "${1:-}" ] && [ "$(readlink "/proc/$1/exe" 2>/dev/null)" = "$(readlink -f "$BIN")" ]; }
unit_state() { systemctl --user show -p ActiveState --value "$1" 2>/dev/null; }
alive_holder() {  # $1: a marker; prints the pid when it names a live holder of this recipe
  local p
  p=$(cat "$1" 2>/dev/null) && [ -n "$p" ] && { tr '\0' ' ' < "/proc/$p/cmdline"; } 2>/dev/null | grep -q inkling-serve-lock-held && echo "$p"
}
gateway_recipe() { docker inspect -f "{{index .Config.Labels \"$LABEL\"}}" "$GATEWAY_NAME" 2>/dev/null; }
gateway_running() { [ "$(docker inspect -f '{{.State.Running}}' "$GATEWAY_NAME" 2>/dev/null)" = true ]; }
lock_free() { mkdir -p "$GPU_LOCK_DIR" 2>/dev/null; touch "$GPU_LOCK_DIR/gpu.lock" 2>/dev/null; ( flock -n -x 9 ) 9<"$GPU_LOCK_DIR/gpu.lock"; }   # true only when nothing holds the pair lock at all

# One node's files, as OK / FAIL lines (run with bash -s on either node).
node_checks() {
  cat <<EOF
if [ -n "$ENGINE_MD5" ]; then
  m=\$(md5sum < "$BIN" 2>/dev/null | cut -d' ' -f1)
  [ "\$m" = "$ENGINE_MD5" ] && echo "OK engine $BIN" || echo "FAIL engine $BIN md5 \${m:-missing} (want $ENGINE_MD5): run scripts/install_engine.sh"
else
  [ -x "$BIN" ] && echo "OK engine $BIN (ENGINE_MD5 unset: not pinned)" || echo "FAIL engine $BIN missing or not executable: run scripts/install_engine.sh"
fi
t=\$(md5sum < "$CHAT_TEMPLATE" 2>/dev/null | cut -d' ' -f1)
[ "\$t" = "$TEMPLATE_MD5" ] && echo "OK template C" || echo "FAIL template $CHAT_TEMPLATE md5 \${t:-missing} (want $TEMPLATE_MD5)"
if [ -n "$VIEW_SHA" ]; then
  s=\$(sha256sum < "$MODEL_VIEW/checkpoint.json" 2>/dev/null | cut -d' ' -f1)
  [ "\$s" = "$VIEW_SHA" ] && echo "OK view $MODEL_VIEW" || echo "FAIL view: checkpoint.json \${s:-missing} under $MODEL_VIEW (want $VIEW_SHA)"
else
  [ -e "$MODEL_VIEW/checkpoint.json" ] && echo "OK view $MODEL_VIEW (VIEW_SHA unset: not pinned)" || echo "FAIL view: no checkpoint.json under $MODEL_VIEW on this node: build the view here too (scripts/make_native_view.py; rank 1 reads the same path on the worker)"
fi
if [ -n "$PACK_SHA" ]; then
  p=\$(sha256sum < "$MODEL_VIEW/dense-exl3/manifest.json" 2>/dev/null | cut -d' ' -f1)
  [ "\$p" = "$PACK_SHA" ] && echo "OK dense pack" || echo "FAIL dense pack: dense-exl3/manifest.json \${p:-missing} under $MODEL_VIEW (want $PACK_SHA)"
else
  [ -e "$MODEL_VIEW/dense-exl3/manifest.json" ] && echo "OK dense pack (PACK_SHA unset: not pinned)" || echo "WARN no dense pack at $MODEL_VIEW/dense-exl3 (the engine falls back to the export's bf16 dense tensors)"
fi
if [ -n "${NCCL_LIB:-}" ]; then
  [ -e "$NCCL_LIB/libnccl.so.2" ] && echo "OK NCCL ($NCCL_LIB)" || echo "FAIL no $NCCL_LIB/libnccl.so.2 (NCCL_LIB is set to that directory)"
else
  ldconfig -p 2>/dev/null | grep -q 'libnccl\.so\.2' && echo "OK NCCL (system, ldconfig)" || echo "FAIL no system libnccl.so.2 and NCCL_LIB is unset: set NCCL_LIB to a directory holding libnccl.so.2 (the engine dlopens it; ldconfig does not list one)"
fi
sudo -n -l "$DROP_HELPER" >/dev/null 2>&1 && echo "OK page-cache helper: sudo -n $DROP_HELPER" || echo "WARN no sudo -n $DROP_HELPER: drops fall back to drop_cache.py (~25 s each)"
if [ -n "$MIX_ENV" ]; then
  r=\$(python3 -I -c 'import json,os,sys; d=sys.argv[1]; f=sorted(set(json.load(open(os.path.join(d,"model.safetensors.index.json")))["weight_map"].values())); m=[x for x in f if not os.path.isfile(os.path.join(d,x))]; print(len(f), len(m), (m or [""])[0])' "$NATIVE_DIR" 2>/dev/null)
  set -- \$r
  [ "\${1:-0}" -gt 0 ] && [ "\${2:-1}" = 0 ] && echo "OK native weights: \$1 shards under $NATIVE_DIR -> \$(readlink -f "$NATIVE_DIR")" || echo "FAIL native weights $NATIVE_DIR: \${2:-?} of \${1:-?} shards missing \${3:-} (the mixed store streams them; MIX_ENV= boots the EXL3 store alone)"
fi
n=\$(find -L "$MODEL_VIEW" -name '*.safetensors' 2>/dev/null | wc -l); need=\$((n + 1024)); [ "\$need" -lt 4096 ] && need=4096
got=\$(systemd-run --user --wait --pipe --quiet -p LimitNOFILE=$NOFILE sh -c 'ulimit -Sn' 2>/dev/null)
if [ "\${got:-0}" -ge "\$need" ] 2>/dev/null; then echo "OK open files: a unit started as the runner starts one gets \$got (the load opens \$n shards; need \$need)"
else echo "FAIL open files: a unit started as the runner starts one gets \${got:-?} (the load opens \$n shards; need \$need): raise NOFILE in config/inkling.env, at most the user manager's DefaultLimitNOFILE \$(systemctl --user show -p DefaultLimitNOFILE --value 2>/dev/null)"; fi
EOF
}

checks_both() {  # prints both nodes' lines; fails when any line FAILs
  local bad=0 line
  while IFS= read -r line; do echo "  head: $line"; case "$line" in FAIL*) bad=1 ;; esac; done < <(node_checks | bash -s)
  while IFS= read -r line; do echo "  worker: $line"; case "$line" in FAIL*) bad=1 ;; esac; done < <(node_checks | on_worker)
  return $bad
}

gpu_apps() {  # $1: head|worker; the GPU compute processes there
  local q="nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null"
  if [ "$1" = head ]; then bash -c "$q"; else wssh "$q"; fi
}

# --- the GPU pair lock (turnstile + gpu.lock), held by a detached holder until stop asks ---
take_lock() {
  local p waited=0 shown=-57      # the first "waiting" line after 3 s, then one a minute
  if p=$(alive_holder "$HELD"); then log "the GPU pair lock is already this recipe's (holder pid $p)"; return 0; fi
  rm -f "$HELD" "$RELEASE"
  mkdir -p "$GPU_LOCK_DIR"
  if ! alive_holder "$HOLDER" >/dev/null; then
    rm -f "$HOLDER"
    setsid nohup bash -c '
      L=$1; HELD=$2; HOLDER=$3; RELEASE=$4
      echo $$ > "$HOLDER"
      exec 8>"$L/gpu.turnstile" 9>"$L/gpu.lock"
      echo "$(date +%T) WAIT  $$ inkling-serve serve.sh (the inkling serve holds the pair lock while it serves)" >> "$L/lock.log"
      flock -x 8; flock -x 9; flock -u 8; exec 8>&-
      echo "$(date +%T) HOLD  $$ inkling-serve" >> "$L/lock.log"
      echo $$ > "$HELD"
      while [ ! -f "$RELEASE" ]; do sleep 3; done
      rm -f "$RELEASE" "$HELD" "$HOLDER"
      echo "$(date +%T) FREE  $$ inkling-serve" >> "$L/lock.log"
    ' inkling-serve-lock-held "$GPU_LOCK_DIR" "$HELD" "$HOLDER" "$RELEASE" >> "$LOGDIR/lock-holder.log" 2>&1 < /dev/null &
  fi
  while ! p=$(alive_holder "$HELD"); do
    if [ "$waited" -ge 3 ] && ! alive_holder "$HOLDER" >/dev/null; then
      log "the lock holder died while waiting ($LOGDIR/lock-holder.log)"; return 1
    fi
    if [ "$LOCK_WAIT_S" -gt 0 ] && [ "$waited" -ge "$LOCK_WAIT_S" ]; then
      log "the GPU pair lock is still busy after ${LOCK_WAIT_S} s (LOCK_WAIT_S); giving up. lock.log:"
      tail -4 "$GPU_LOCK_DIR/lock.log" | cut -c1-200
      release_lock; return 1
    fi
    if [ $((waited - shown)) -ge 60 ]; then
      log "waiting for the GPU pair lock (jobs queued before this one run first); lock.log: $(tail -1 "$GPU_LOCK_DIR/lock.log" 2>/dev/null | cut -c1-160)"
      shown=$waited
    fi
    sleep 3; waited=$((waited + 3))
  done
  log "GPU pair lock held (holder pid $p, after ${waited} s)"
}

release_lock() {
  local p
  if p=$(alive_holder "$HELD"); then
    touch "$RELEASE"
    for _ in $(seq 1 20); do alive_holder "$HELD" >/dev/null || break; sleep 1; done
    if alive_holder "$HELD" >/dev/null; then kill -- "-$p" 2>/dev/null; log "the lock holder did not let go in 20 s: killed (pid $p)"
    else log "GPU pair lock released"; fi
  elif p=$(alive_holder "$HOLDER"); then
    kill -- "-$p" 2>/dev/null; log "stopped the waiting lock holder (pid $p)"   # its group: the bash and its flock
  fi
  for marker in "$HELD" "$HOLDER"; do   # a holder that died (killed with its unit) leaves its markers
    if [ -e "$marker" ] && ! alive_holder "$marker" >/dev/null; then rm -f "$marker"; log "removed the stale marker $marker"; fi
  done
  rm -f "$RELEASE"
}

# --- the ranks ---
start_rank0() {  # $1: stamp
  ( cd /tmp && setsid bash -c 'echo $$ > "$1"; shift; exec "$@"' _ "$PID0" env $ENVS "$BIN" serve "$MODEL_VIEW" $ARGS \
      --rank 0 --host 127.0.0.1 --port "$PORT" > "$LOGDIR/r0-$1.log" 2>&1 < /dev/null & )
  ln -sfn "r0-$1.log" "$LOGDIR/r0.log"
}

start_rank1() {  # $1: stamp; rank 1 as worker's user unit
  on_worker <<EOF
set -u
mkdir -p '$LOGDIR'
systemctl --user stop '$PEER_UNIT' 2>/dev/null; systemctl --user reset-failed '$PEER_UNIT' 2>/dev/null
systemd-run --user --quiet -p LimitNOFILE=$NOFILE --unit='$PEER_UNIT' -p TimeoutStopSec=30 --working-directory=/tmp \
  --description='Inkling rank 1 (head node: serve.sh; its stop stops this)' \
  /bin/bash -c 'exec env $ENVS $BIN serve $MODEL_VIEW $ARGS --rank 1 --host 127.0.0.1 --port $PEER_PORT > $LOGDIR/r1-$1.log 2>&1 < /dev/null'
ln -sfn 'r1-$1.log' '$LOGDIR/r1.log'
EOF
}

stop_rank0() {
  local p q
  p=$(cat "$PID0" 2>/dev/null)
  if ours "$p"; then
    kill "$p" 2>/dev/null
    for _ in $(seq 1 30); do ours "$p" || break; sleep 1; done
    ours "$p" && kill -9 "$p" 2>/dev/null
  fi
  for q in $(pgrep -x tensorfold-nati); do ours "$q" && kill "$q" 2>/dev/null; done   # strays of this exact binary
  sleep 1
  for q in $(pgrep -x tensorfold-nati); do ours "$q" && kill -9 "$q" 2>/dev/null; done
  rm -f "$PID0"
  local left=""; for q in $(pgrep -x tensorfold-nati); do ours "$q" && left="$left $q"; done
  [ -z "$left" ] && log "rank 0: stopped" || log "rank 0: STILL RUNNING:$left"
}

stop_rank1() {
  on_worker <<EOF | sed 's/^/  /'
systemctl --user stop '$PEER_UNIT' 2>/dev/null; systemctl --user reset-failed '$PEER_UNIT' 2>/dev/null
b=\$(readlink -f '$BIN')
for q in \$(pgrep -x tensorfold-nati); do [ "\$(readlink /proc/\$q/exe)" = "\$b" ] && kill \$q; done
sleep 1
for q in \$(pgrep -x tensorfold-nati); do [ "\$(readlink /proc/\$q/exe)" = "\$b" ] && kill -9 \$q; done
left=""; for q in \$(pgrep -x tensorfold-nati); do [ "\$(readlink /proc/\$q/exe)" = "\$b" ] && left="\$left \$q"; done
[ -z "\$left" ] && echo "rank 1 (worker): stopped" || echo "rank 1 (worker): STILL RUNNING:\$left"
true
EOF
}

# --- page caches: the root helper (or drop_cache.py) now, and the dropper units while serving ---
sync_dropper() {  # the dropper's scripts onto the worker (same path)
  wssh "mkdir -p '$DEP/scripts' '$LOGDIR'" && rsync -a "$DEP/scripts/dropcache_loop.sh" "$DEP/scripts/drop_cache.py" "$WORKER_SSH:$DEP/scripts/"
}

drop_now() {  # one drop on both nodes now (before the load, right after READY): each node's before / after line
  local env="DROP_HELPER=$DROP_HELPER"
  sync_dropper || log "could not copy the dropper's scripts to the worker"
  env $env bash "$DEP/scripts/dropcache_loop.sh" --now "$LOGDIR/dropcache.log" | sed 's/^/  head: /'
  wssh "env $env bash '$DEP/scripts/dropcache_loop.sh' --now '$LOGDIR/dropcache-r1.log'" | sed 's/^/  worker: /'
}

dropper_start() {  # the dropper units on both nodes (replacing any running)
  local env=(-E "DROP_EVERY_S=$DROP_EVERY_S" -E "DROP_MIN_MIB=$DROP_MIN_MIB" -E "DROP_HELPER=$DROP_HELPER")
  sync_dropper || { log "could not copy the dropper's scripts to the worker"; return 1; }
  systemctl --user stop "$DROP_UNIT" 2>/dev/null; systemctl --user reset-failed "$DROP_UNIT" 2>/dev/null
  systemd-run --user --quiet --unit="$DROP_UNIT" -p TimeoutStopSec=10 "${env[@]}" \
    --description="Inkling page-cache dropper (head node; serve.sh)" \
    /bin/bash "$DEP/scripts/dropcache_loop.sh" "$LOGDIR/dropcache.log" || return 1
  wssh "systemctl --user stop '$PEER_DROP_UNIT' 2>/dev/null; systemctl --user reset-failed '$PEER_DROP_UNIT' 2>/dev/null; \
    systemd-run --user --quiet --unit='$PEER_DROP_UNIT' -p TimeoutStopSec=10 ${env[*]} \
      --description='Inkling page-cache dropper (worker node; serve.sh on the head)' \
      /bin/bash '$DEP/scripts/dropcache_loop.sh' '$LOGDIR/dropcache-r1.log'" || return 1
  log "page-cache droppers: $DROP_UNIT (head), $PEER_DROP_UNIT (worker): every ${DROP_EVERY_S} s above ${DROP_MIN_MIB} MiB of file cache"
}

dropper_stop() {
  systemctl --user stop "$DROP_UNIT" 2>/dev/null; systemctl --user reset-failed "$DROP_UNIT" 2>/dev/null
  wssh "systemctl --user stop '$PEER_DROP_UNIT' 2>/dev/null; systemctl --user reset-failed '$PEER_DROP_UNIT' 2>/dev/null; true"
  log "page-cache droppers: stopped"
}

dropper_status() {
  echo "page-cache dropper head ($DROP_UNIT): $(unit_state "$DROP_UNIT"); last: $(tail -1 "$LOGDIR/dropcache.log" 2>/dev/null | cut -c1-150)"
  echo "page-cache dropper worker ($PEER_DROP_UNIT): $(wssh "systemctl --user show -p ActiveState --value '$PEER_DROP_UNIT'; tail -1 '$LOGDIR/dropcache-r1.log' 2>/dev/null | cut -c1-150" 2>/dev/null | paste -sd ';' | sed 's/;/; last: /')"
}

# --- the gateway ---
gateway() {
  local ctx owner
  engine_ok || { log "the engine does not serve $SERVED_NAME on :$PORT: no gateway"; return 1; }
  owner=$(gateway_recipe)
  if gateway_running && [ "$owner" != "$RECIPE" ]; then
    log "the gateway container serves another recipe (${owner:-unlabeled}): stop that recipe first"; return 1
  fi
  mkdir -p "$GATEWAY_STATE"
  if ! docker image inspect "$GATEWAY_IMAGE" >/dev/null 2>&1; then
    log "gateway image $GATEWAY_IMAGE not found: building it from $DEP/gateway"
    docker build -t "$GATEWAY_IMAGE" "$DEP/gateway" >/dev/null || { log "gateway image build failed"; return 1; }
  fi
  if docker inspect "$GATEWAY_NAME" >/dev/null 2>&1; then
    docker stop -t 10 "$GATEWAY_NAME" >/dev/null 2>&1       # SIGTERM: the gateway writes its memory file compact
    if ! docker inspect -f '{{range .Mounts}}{{.Destination}} {{end}}' "$GATEWAY_NAME" | grep -qw /var/lib/inkling-gateway \
        && docker cp "$GATEWAY_NAME:/var/lib/inkling-gateway/inkling-memory.jsonl" "$GATEWAY_STATE/" >/dev/null 2>&1; then
      log "gateway: its Inkling memory carried over to $GATEWAY_STATE"
    fi
  fi
  docker rm -f "$GATEWAY_NAME" >/dev/null 2>&1
  if listening "$GATEWAY_PORT"; then log "port $GATEWAY_PORT is in use: $(ss -ltnpH "sport = :$GATEWAY_PORT")"; return 1; fi
  ctx=$(curl -fsS --max-time 5 "$API/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["context_length"])' 2>/dev/null) || ctx=$CONTEXT
  # GATEWAY_HIDDEN_ALIAS: a comma- or space-separated list, one --hidden-alias each (the gateway's option repeats)
  local hidden=() h
  for h in ${GATEWAY_HIDDEN_ALIAS//,/ }; do hidden+=(--hidden-alias "$h"); done
  # shellcheck disable=SC2086
  docker run -d --name "$GATEWAY_NAME" --label "$LABEL=$RECIPE" --network host $GATEWAY_ENV \
    --tmpfs /tmp:rw,size=256m,mode=1777 -v "$GATEWAY_STATE:/var/lib/inkling-gateway" \
    -v /etc/ssl/certs:/etc/ssl/certs:ro -v /etc/ssl/certs/ca-certificates.crt:/usr/local/lib/python3.12/dist-packages/certifi/cacert.pem:ro \
    -e SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt -e REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    "$GATEWAY_IMAGE" --listen "0.0.0.0:$GATEWAY_PORT" --upstream "http://127.0.0.1:$PORT" \
    --served-model "$GATEWAY_MODEL" ${GATEWAY_ALIAS_MODEL:+--alias-model "$GATEWAY_ALIAS_MODEL"} \
    "${hidden[@]}" ${GATEWAY_DISPLAY_NAME:+--display-name "$GATEWAY_DISPLAY_NAME"} --context "$ctx" \
    --max-tokens "$GATEWAY_MAX_TOKENS" --root "$GATEWAY_ROOT" --family "$GATEWAY_FAMILY" >/dev/null || return 1
  for _ in $(seq 1 30); do gateway_ok && break; sleep 1; done
  gateway_ok || { log "the gateway does not answer on :$GATEWAY_PORT"; docker logs --tail 30 "$GATEWAY_NAME" 2>&1; return 1; }
  log "gateway: $GATEWAY_MODEL${GATEWAY_DISPLAY_NAME:+ ($GATEWAY_DISPLAY_NAME)}${GATEWAY_ALIAS_MODEL:+, alias $GATEWAY_ALIAS_MODEL}${GATEWAY_HIDDEN_ALIAS:+, also answers $GATEWAY_HIDDEN_ALIAS} on :$GATEWAY_PORT -> :$PORT (root $GATEWAY_ROOT, context $ctx, family $GATEWAY_FAMILY)"
}

# --- start: the foreground checks, then the unit ---
preflight() {
  local bad=0 g owner
  local self=""                      # this recipe already serves: its own engine, GPUs and lock are no refusal
  if ours "$(cat "$PID0" 2>/dev/null)" && alive_holder "$HELD" >/dev/null; then
    self=1; echo "  OK this recipe is serving (rank 0 pid $(cat "$PID0"); a start reports it; restart replaces it)"
  fi
  if listening "$PORT" && [ -z "$self" ]; then echo "  FAIL port $PORT serves another engine: $(ss -ltnpH "sport = :$PORT" | cut -c1-150) (scripts/serve.sh stop)"; bad=1; fi
  owner=$(gateway_recipe)
  if gateway_running && [ "$owner" != "$RECIPE" ]; then
    echo "  FAIL $GATEWAY_NAME serves another recipe (${owner:-unlabeled}): stop that recipe first"; bad=1
  elif ! gateway_running && listening "$GATEWAY_PORT"; then
    echo "  FAIL port $GATEWAY_PORT is in use: $(ss -ltnpH "sport = :$GATEWAY_PORT" | cut -c1-150)"; bad=1
  fi
  for node in head worker; do
    g=$(gpu_apps "$node" | tr '\n' ' ')
    if [ -n "${g// }" ] && [ -n "$self" ] && [ -z "$(gpu_apps "$node" | grep -v ", $(readlink -f "$BIN"),")" ]; then
      echo "  OK $node GPU: this recipe's rank"
    elif [ -n "${g// }" ]; then
      if lock_free; then echo "  FAIL $node GPU busy outside the pair lock: $g"; bad=1
      else echo "  WAIT $node GPU busy under the pair lock ($g): the serve queues behind it (LOCK_WAIT_S=$LOCK_WAIT_S)"; fi
    fi
  done
  checks_both || bad=1
  return $bad
}

follow() {  # the unit's log until it serves or fails
  local tp rc=1 st
  tail -n +1 -F "$LOGDIR/serve.log" 2>/dev/null & tp=$!
  trap 'kill $tp 2>/dev/null; echo; log "stopped following: the start goes on in the unit $UNIT (status: scripts/serve.sh status; cancel: scripts/serve.sh stop)"; exit 130' INT
  while sleep 2; do
    st=$(cat "$STATE" 2>/dev/null)
    case "$st" in serving*) rc=0; break ;; failed*) rc=1; break ;; esac
    case "$(unit_state "$UNIT")" in
      active|activating|reloading) ;;
      *) sleep 1; st=$(cat "$STATE" 2>/dev/null); case "$st" in serving*) rc=0 ;; *) log "the unit $UNIT ended without serving (${st:-no state})" ;; esac; break ;;
    esac
  done
  sleep 1; kill "$tp" 2>/dev/null; wait "$tp" 2>/dev/null
  trap - INT
  return $rc
}

start() {
  local envs=() v
  if [ "$(unit_state "$UNIT")" = active ]; then
    case "$(cat "$STATE" 2>/dev/null)" in
      serving*)
        if engine_ok && gateway_ok; then log "already serving $GATEWAY_MODEL on :$GATEWAY_PORT"; status; return 0; fi
        if engine_ok; then log "the engine serves; the gateway does not: recreating it"; gateway && return 0; return 1; fi
        log "the unit says serving, but the engine does not answer: stop, then start"; return 1 ;;
      failed*) ;;
      *) log "a start is in progress: following it"; follow; return ;;
    esac
  fi
  log "preflight"
  preflight || { log "preflight failed: not starting"; return 1; }
  for v in CONTEXT MAX_TOKENS PARALLEL TF_INKLING2_FLOOR_GIB MIX_ENV ROWS_ENV EXTRA_ENV BIN MODEL_VIEW CHAT_TEMPLATE PORT PEER_PORT GPU_LOCK_DIR \
           LOCK_WAIT_S MIN_AVAIL_GIB READY_S GATEWAY_MODEL GATEWAY_ALIAS_MODEL GATEWAY_HIDDEN_ALIAS GATEWAY_DISPLAY_NAME ENV_FILE NOFILE; do
    envs+=(-E "$v=${!v}")
  done
  systemctl --user stop "$UNIT" 2>/dev/null; systemctl --user reset-failed "$UNIT" 2>/dev/null
  state "launching"; : > "$LOGDIR/serve.log"
  systemd-run --user --quiet -p LimitNOFILE="$NOFILE" --unit="$UNIT" -p RemainAfterExit=yes -p TimeoutStopSec=60 --working-directory="$DEP" \
    --description="Inkling serve ($RECIPE: the GPU pair lock, rank 0, the gateway; serve.sh)" \
    -E "TF_RECIPE=$RECIPE" "${envs[@]}" \
    /bin/bash -c "exec /bin/bash '$DEP/scripts/serve.sh' serve >> '$LOGDIR/serve.log' 2>&1" ||
    { log "systemd-run failed"; return 1; }
  log "launched the user unit $UNIT (log $LOGDIR/serve.log); following"
  follow
}

# --- serve: the unit's body (never run it from a shell: it would die with the session) ---
abort() {
  trap - TERM
  log "FAILED: $*"
  if [ -n "${RANKS:-}" ]; then stop_rank0; stop_rank1; fi
  release_lock
  state "failed: $*"
  exit 1
}

serve() {
  local stamp g a0 a1
  [ -n "${INVOCATION_ID:-}" ] || [ "${FOREGROUND:-0}" = 1 ] || { echo "serve runs inside the unit $UNIT: use start" >&2; exit 2; }
  trap 'abort "stopped (SIGTERM)"' TERM
  stamp=$(date +%Y%m%dT%H%M%S)
  log "serve $RECIPE: engine $BIN, view $MODEL_VIEW, window $CONTEXT, template $(basename "$CHAT_TEMPLATE"), floor $TF_INKLING2_FLOOR_GIB GiB${MIX_ENV:+, mixed store ($MIX_ENV)}${EXTRA_ENV:+, extra env $EXTRA_ENV}"
  state "waiting for the GPU pair lock"
  take_lock || { state "failed: the GPU pair lock was not taken"; exit 1; }
  state "loading"
  log "dropping the page cache on both nodes before the load ($DROP_HELPER; drop_cache.py where sudo -n refuses)"
  drop_now
  g=$(gpu_apps head | tr '\n' ' '); [ -z "${g// }" ] || abort "the head GPU is busy outside the pair lock: $g"
  g=$(gpu_apps worker | tr '\n' ' '); [ -z "${g// }" ] || abort "the worker GPU is busy outside the pair lock: $g"
  a0=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  a1=$(wssh "awk '/MemAvailable/ {print int(\$2/1048576)}' /proc/meminfo")
  log "MemAvailable: head $a0 GiB, worker ${a1:-?} GiB (need $MIN_AVAIL_GIB)"
  [ "${a0:-0}" -ge "$MIN_AVAIL_GIB" ] && [ "${a1:-0}" -ge "$MIN_AVAIL_GIB" ] ||
    abort "MemAvailable head $a0 / worker ${a1:-?} GiB is under $MIN_AVAIL_GIB GiB: another process holds the memory"
  listening "$PORT" && abort "port $PORT is in use"
  wssh "mkdir -p '$(dirname "$BIN")' '$(dirname "$CHAT_TEMPLATE")' '$LOGDIR'" || abort "the worker is unreachable"
  rsync -a "$CHAT_TEMPLATE" "$WORKER_SSH:$CHAT_TEMPLATE" && rsync -a "$BIN" "$WORKER_SSH:$BIN" || abort "the copy to the worker failed"
  # the mixed store reads its prior on every rank and writes its demand history at each request's end; neither
  # is in this checkout's cargo on the worker (rank 1 gets only the engine and the template above), so copy the
  # prior at its absolute path and make the history's directory there too. A miss there is fatal (the engine
  # refuses to load); a missing history only starts the bank cold.
  if [ -n "$MIX_ENV" ]; then
    prior=$(printf '%s' "$MIX_ENV" | tr ' ' '\n' | sed -n 's/^TF_INKLING2_PRIOR=//p' | head -1)
    history=$(printf '%s' "$MIX_ENV" | tr ' ' '\n' | sed -n 's/^TF_INKLING2_MIX_HISTORY=//p' | head -1)
    if [ -n "$prior" ] && [ -f "$prior" ]; then
      wssh "mkdir -p '$(dirname "$prior")'" && rsync -a "$prior" "$WORKER_SSH:$prior" || abort "the copy of the mixed store's prior ($prior) to the worker failed"
    fi
    [ -n "$history" ] && { wssh "mkdir -p '$(dirname "$history")'" || abort "could not make $(dirname "$history") on the worker"; }
  fi
  checks_both || abort "a node lacks the engine, template or view (lines above)"
  log "starting rank 0 (head 127.0.0.1:$PORT) and rank 1 (worker 127.0.0.1:$PEER_PORT); logs $LOGDIR/r{0,1}-$stamp.log"
  RANKS=1
  start_rank0 "$stamp"
  sleep 3
  start_rank1 "$stamp" || abort "rank 1 did not start"
  for _ in $(seq 1 $((READY_S / 5))); do
    if engine_ok; then break; fi
    if ! ours "$(cat "$PID0" 2>/dev/null)"; then tail -30 "$LOGDIR/r0-$stamp.log"; abort "rank 0 exited ($LOGDIR/r0-$stamp.log)"; fi
    if ! wssh "systemctl --user is-active --quiet '$PEER_UNIT'"; then
      wssh "tail -30 '$LOGDIR/r1-$stamp.log'"; abort "rank 1 exited (worker $LOGDIR/r1-$stamp.log)"
    fi
    sleep 5
  done
  engine_ok || abort "not serving after $READY_S s (READY_S)"
  log "engine READY: $(curl -fsS --max-time 5 "$API/models" | head -c 160)"
  log "dropping the page cache on both nodes after the load"
  drop_now
  gateway || abort "the gateway did not come up"
  dropper_start || log "WARNING: the page-cache droppers did not start (scripts/serve.sh dropcache start)"
  state "serving $GATEWAY_MODEL on :$GATEWAY_PORT (engine :$PORT)"
  log "READY: $GATEWAY_MODEL through the gateway on :$GATEWAY_PORT"
}

stop() {
  local owner
  owner=$(gateway_recipe)
  if [ "$owner" = "$RECIPE" ]; then
    docker stop "$GATEWAY_NAME" >/dev/null 2>&1 && log "gateway: stopped"
  elif gateway_running; then
    log "gateway: left alone (it serves ${owner:-another recipe})"
  fi
  # a start in progress stops first (its own SIGTERM trap stops what it started)
  if [ -n "$(systemctl --user show -p MainPID --value "$UNIT" 2>/dev/null | grep -v '^0$')" ]; then
    log "stopping the start in progress"; systemctl --user stop "$UNIT" 2>/dev/null
  fi
  dropper_stop
  stop_rank0
  stop_rank1
  release_lock
  systemctl --user stop "$UNIT" 2>/dev/null; systemctl --user reset-failed "$UNIT" 2>/dev/null
  state "stopped"
  log "stopped"
}

status() {
  local rc=0 p
  if engine_ok; then echo "engine :$PORT: serving $SERVED_NAME (rank 0 pid $(cat "$PID0" 2>/dev/null))"
  else echo "engine :$PORT: not serving"; rc=3; fi
  echo "rank 1 (worker, unit $PEER_UNIT): $(wssh "systemctl --user show -p ActiveState,MainPID --value '$PEER_UNIT'" 2>/dev/null | tr '\n' ' ')"
  if gateway_ok; then echo "gateway :$GATEWAY_PORT: serving $GATEWAY_MODEL${GATEWAY_DISPLAY_NAME:+ ($GATEWAY_DISPLAY_NAME)}${GATEWAY_ALIAS_MODEL:+ + $GATEWAY_ALIAS_MODEL}${GATEWAY_HIDDEN_ALIAS:+, also answers $GATEWAY_HIDDEN_ALIAS} (container label: $(gateway_recipe))"
  else echo "gateway :$GATEWAY_PORT: not serving $GATEWAY_MODEL"; rc=3; fi
  if p=$(alive_holder "$HELD"); then echo "GPU pair lock: held by this recipe (holder pid $p)"
  elif p=$(alive_holder "$HOLDER"); then echo "GPU pair lock: waiting for it (holder pid $p)"
  else echo "GPU pair lock: not this recipe's (lock.log: $(tail -1 "$GPU_LOCK_DIR/lock.log" 2>/dev/null | cut -c1-100))"; fi
  echo "unit $UNIT: $(unit_state "$UNIT"); $(cat "$STATE" 2>/dev/null || echo 'no state')"
  dropper_status
  return $rc
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  logs) exec tail -n 40 -F "$LOGDIR/serve.log" "$LOGDIR/r0.log" ;;
  gateway) gateway ;;
  check) preflight && log "preflight: a start would proceed" || { log "preflight: a start would refuse"; exit 1; } ;;
  dropcache)
    case "${2:-status}" in
      start) dropper_start ;; stop) dropper_stop ;; status) dropper_status ;;
      *) echo "usage: scripts/serve.sh dropcache start|stop|status" >&2; exit 2 ;;
    esac ;;
  serve) serve ;;
  *) sed -n '2,19p' "$0"; exit 2 ;;
esac
