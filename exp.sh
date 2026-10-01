#!/usr/bin/env bash
# Experiments on the project's working tree, recorded in lx-work. The project repo gets no branch
# and no commit: each change is saved as a patch, and the tree must always equal the baseline
# commit plus the kept patches, applied in number order.
# Night trials (rules/night.md) live in lx-work/night/ and never stay in the tree: each is parked.
#   exp.sh new <1 to 4 words>          start the next experiment; checks the tree first
#   exp.sh try <1 to 4 words>          start the next night trial, on the baseline plus all kept patches
#   exp.sh save                        write the running item's change.patch, new files included
#   exp.sh keep                        end the experiment and keep the change (needs run.sh and result.json)
#   exp.sh park                        end the night trial: save its patch, reverse it, confirm the tree is back
#   exp.sh drop                        end either, reverse the change, and confirm the tree is back
#   exp.sh adopt <NNN>                 turn parked night trial NNN into the next experiment
#   exp.sh delivered                   the user committed the kept patches: HEAD becomes the baseline
#   exp.sh check                       confirm the tree equals the baseline plus the kept patches; fails while an item runs
#   exp.sh export <NNN|night/NNN|best|baseline>  write that code to lx-work/local/replay/<name>/
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project="$(dirname "$here")"
work="$project/lx-work"
exp="$work/exp"
night="$work/night"
die() { echo "error: $*" >&2; exit 1; }
[ -d "$exp" ] || die "$exp not found; run lx-e2espeed/setup.sh project first"
cd "$project"
top="$(git rev-parse --show-toplevel 2>/dev/null)" || die "$project is not a git repo"
[ "$(cd "$top" && pwd -P)" = "$project" ] || die "lx-e2espeed must sit at the top of the project repo"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

diff_() {  # plain patches whatever the user's git config says
  git -c diff.noprefix=false -c diff.mnemonicPrefix=false diff --no-color --no-ext-diff --no-textconv "$@"
}

baseline() {
  local c
  c="$(sed -n 's/^- Baseline commit: *\([0-9a-f]\{7,40\}\) *$/\1/p' "$work/PROJECT.md")"
  c="${c%%$'\n'*}"
  [ -n "$c" ] || die "lx-work/PROJECT.md has no '- Baseline commit: <hash>' line"
  git rev-parse -q --verify "$c^{commit}" || die "baseline commit $c is not in the project repo"
}

snapshot() {  # tree id of the working tree; the project's own index is never touched
  local index
  index="$(git rev-parse --git-path index)"
  if [ -f "$index" ]; then command cp -f "$index" "$tmp/snap"; fi
  GIT_INDEX_FILE="$tmp/snap" git add -A .
  GIT_INDEX_FILE="$tmp/snap" git write-tree
}

replay_base=""  # set by export: rebuild an item from the baseline in its own Base line

replay() {  # tree id of the baseline commit plus the patches of the given experiment dirs
  local base d
  base="${replay_base:-$(baseline)}"
  rm -f "$tmp/replay"
  GIT_INDEX_FILE="$tmp/replay" git read-tree "$base"
  for d in "$@"; do
    [ -s "$d/change.patch" ] || continue
    GIT_INDEX_FILE="$tmp/replay" git apply --cached --binary --whitespace=nowarn "$d/change.patch" \
      || die "$(basename "$d")/change.patch does not apply to the replayed tree"
  done
  GIT_INDEX_FILE="$tmp/replay" git write-tree
}

items() {  # dirs under $1 (exp or night) in number order, with status $2 (any status when empty)
  local d
  for d in "$1"/[0-9][0-9][0-9]-*/; do
    [ -f "$d/NOTE.md" ] || continue
    if [ -z "${2:-}" ] || grep -qx -- "- Status: $2" "$d/NOTE.md"; then echo "${d%/}"; fi
  done
}

exp_dirs() { items "$exp" "${1:-}"; }

num_of() { local n; n="$(basename "$1")"; echo "$((10#${n%%-*}))"; }

kept_below() {  # kept experiment dirs numbered below $1
  local d
  while IFS= read -r d; do
    if [ "$(num_of "$d")" -lt "$1" ]; then echo "$d"; fi
  done < <(exp_dirs kept)
}

running() { local all; all="$(items "$exp" running; items "$night" running)"; printf '%s' "${all%%$'\n'*}"; }

is_trial() { [[ "$1" == "$night"/* ]]; }

kept_for() {  # kept experiments the item $1 builds on: all of them for a night trial
  if is_trial "$1"; then exp_dirs kept; else kept_below "$(num_of "$1")"; fi
}

kept_of() {  # experiment dirs named in the Base line of NOTE.md $1; a missing one prints its pattern
  local n m
  for n in $(sed -n 's/^- Base: baseline [0-9a-f]* + kept \([0-9 ]*[0-9]\).*$/\1/p' "$1"); do
    m=("$exp/$n"-*/)
    echo "${m[0]%/}"
  done
}

must_match() {  # $1 expected tree, $2 actual tree, $3 message
  if [ "$1" != "$2" ]; then
    diff_ --stat "$1" "$2" >&2
    die "$3"
  fi
}

warn_ignored() {  # ignored files written since the experiment started are in no patch
  local files
  [ -f "$work/local/exp-start" ] || return 0
  files="$(find . \( -path ./.git -o -path ./lx-work -o -path ./lx-e2espeed -o -name __pycache__ \) -prune \
             -o -type f -newer "$work/local/exp-start" -print 2>/dev/null \
           | git check-ignore --stdin 2>/dev/null | head -n 10 || true)"
  [ -n "$files" ] || return 0
  echo "warning: ignored files written during this experiment are not in its patch (up to 10):" >&2
  echo "$files" >&2
  echo "Build outputs and caches can change the next bench: make run.sh rebuild them, and after a drop remove the ones this experiment made." >&2
}

set_status() {  # $1 experiment or trial dir, $2 new status, in NOTE.md and its INDEX.md
  local name
  name="$(basename "$1")"
  sed -i "s/^- Status: [a-z]*\$/- Status: $2/" "$1/NOTE.md"
  sed -i "s/^| $name | [a-z]* |/| $name | $2 |/" "$(dirname "$1")/INDEX.md"
}

start() {  # $1 exp or night, then 1 to 4 words: check the tree and create the next dir there
  local root="$1" name base now want list="" last=0 n num dir d cells
  shift
  name="$(printf '%s' "$*" | tr 'A-Z ' 'a-z-' | tr -s '-')"
  name="${name#-}"; name="${name%-}"
  [[ "$name" =~ ^[a-z0-9]+(-[a-z0-9]+){0,3}$ ]] || die "usage: exp.sh new|try <1 to 4 words of letters and digits>"
  [ -z "$(running)" ] || die "$(basename "$(running)") is still running; end it first"
  base="$(baseline)"
  [ "$(git rev-parse HEAD)" = "$base" ] \
    || die "the project HEAD is not the baseline commit ${base:0:7}; see 'When the project changes underneath' in rules/experiment.md"
  mapfile -t kept < <(exp_dirs kept)
  now="$(snapshot)"
  want="$(replay ${kept[@]+"${kept[@]}"})"
  must_match "$want" "$now" "the project tree differs from the baseline plus the kept patches (files above); ask the user before going on"
  for d in ${kept[@]+"${kept[@]}"}; do list+="$(basename "$d" | cut -d- -f1) "; done
  list="${list% }"
  # Numbers are never reused, so take the highest one seen in directories or indexes.
  if [ ! -f "$root/INDEX.md" ]; then
    mkdir -p "$root"
    command cp -f "$here/templates/work-skeleton/$(basename "$root")/INDEX.md" "$root/INDEX.md"
  fi
  for n in $( { ls "$root"; cat "$root/INDEX.md" "$root/ARCHIVE.md" 2>/dev/null; } \
              | grep -oE '(^|[^0-9])[0-9]{3}-[a-z0-9]' | grep -oE '[0-9]{3}' || true); do
    if [ $((10#$n)) -gt "$last" ]; then last=$((10#$n)); fi
  done
  [ "$last" -lt 999 ] || die "numbers in $(basename "$root")/ are exhausted"
  num="$(printf '%03d' $((last + 1)))"
  dir="$root/$num-$name"
  mkdir "$dir"
  sed -e "s/^# NNN-name\$/# $num-$name/" \
      -e "s/^- Date: <YYYY-MM-DD>\$/- Date: $(date +%F)/" \
      -e "s/^- Base: .*\$/- Base: baseline ${base:0:7} + kept ${list:-none}/" \
      -e "s/^- Status: running | kept | delivered | dropped | parked | adopted\$/- Status: running/" \
      "$here/templates/exp-note.md" > "$dir/NOTE.md"
  if grep -qE '^(# NNN-name|- (Date|Base|Status): .*[<|])' "$dir/NOTE.md"; then
    rm -rf -- "$dir"
    die "templates/exp-note.md no longer matches exp.sh; fix one of them"
  fi
  # A new row has as many cells as the index's header row.
  cells="$(grep -m1 '^| [A-Z]' "$root/INDEX.md" | tr -cd '|' | wc -c)"
  printf '| %s | running |%s\n' "$num-$name" "$(printf ' |%.0s' $(seq 3 "$((cells - 1))"))" >> "$root/INDEX.md"
  mkdir -p "$work/local" && touch "$work/local/exp-start"
  echo "Started $(basename "$root")/$num-$name on baseline ${base:0:7} + kept ${list:-none}."
}

cmd_new() { start "$exp" "$@"; }

cmd_try() { start "$night" "$@"; }

cmd_save() {
  local dir start now stat
  dir="$(running)"
  [ -n "$dir" ] || die "no running experiment"
  mapfile -t kept < <(kept_for "$dir")
  start="$(replay ${kept[@]+"${kept[@]}"})"
  now="$(snapshot)"
  diff_ --binary --full-index "$start" "$now" > "$dir/change.patch"
  stat="$(diff_ --shortstat "$start" "$now")"
  echo "Saved $(basename "$dir")/change.patch:${stat:- no changes}"
}

cmd_keep() {
  local dir f want now
  dir="$(running)"
  [ -n "$dir" ] || die "no running experiment"
  ! is_trial "$dir" || die "night trials end with park or drop; the user adopts them into experiments"
  for f in run.sh result.json; do
    [ -f "$dir/$f" ] || die "save $f in lx-work/exp/$(basename "$dir")/ first"
  done
  cmd_save
  mapfile -t kept < <(kept_below "$(num_of "$dir")")
  want="$(replay ${kept[@]+"${kept[@]}"} "$dir")"
  now="$(snapshot)"
  must_match "$want" "$now" "the saved patch does not reproduce the tree; nothing was changed"
  set_status "$dir" kept
  warn_ignored
  echo "Kept $(basename "$dir"). Replay check passed. Fill in the results, re-run the stage timing for MAP.md if the patch changed a file it points to, then commit lx-work."
}

cmd_drop() {
  local dir start now
  dir="$(running)"
  [ -n "$dir" ] || die "no running experiment"
  cmd_save
  if [ -s "$dir/change.patch" ]; then
    git apply -R --binary --whitespace=nowarn "$dir/change.patch" || die "could not reverse the change; the tree is unchanged"
  fi
  mapfile -t kept < <(kept_for "$dir")
  start="$(replay ${kept[@]+"${kept[@]}"})"
  now="$(snapshot)"
  must_match "$start" "$now" "after reversing, the tree still differs from the start of the experiment"
  set_status "$dir" dropped
  warn_ignored
  echo "Dropped $(basename "$dir"); the tree is back to the baseline plus the kept patches. Record why, then commit lx-work."
}

cmd_park() {
  local dir f want now start
  dir="$(running)"
  [ -n "$dir" ] && is_trial "$dir" || die "no running night trial"
  for f in run.sh result.json; do
    [ -f "$dir/$f" ] || die "save $f in lx-work/night/$(basename "$dir")/ first"
  done
  cmd_save
  [ -s "$dir/change.patch" ] || die "the tree holds no change; land the smallest change and verify it before parking"
  mapfile -t kept < <(exp_dirs kept)
  want="$(replay ${kept[@]+"${kept[@]}"} "$dir")"
  now="$(snapshot)"
  must_match "$want" "$now" "the saved patch does not reproduce the tree; nothing was changed"
  git apply -R --binary --whitespace=nowarn "$dir/change.patch" || die "could not reverse the change; the tree is unchanged"
  start="$(replay ${kept[@]+"${kept[@]}"})"
  now="$(snapshot)"
  must_match "$start" "$now" "after reversing, the tree still differs from the base"
  set_status "$dir" parked
  warn_ignored
  echo "Parked night/$(basename "$dir"): its patch rebuilds the change, and the tree is back to the base. Record the result, then commit lx-work."
}

cmd_adopt() {
  local match trial was dir now words
  [[ "${1:-}" =~ ^[0-9]{3}$ ]] || die "usage: exp.sh adopt <night trial number>"
  match=("$night/$1"-*/)
  [ -d "${match[0]}" ] || die "no night trial $1"
  trial="${match[0]%/}"
  grep -qx -- "- Status: parked" "$trial/NOTE.md" || die "night/$(basename "$trial") is not parked"
  was="$(sed -n 's/^- Base: //p' "$trial/NOTE.md")"
  words="$(basename "$trial" | cut -d- -f2- | tr '-' ' ')"
  start "$exp" "$words" > /dev/null
  dir="$(running)"
  now="$(sed -n 's/^- Base: //p' "$dir/NOTE.md")"
  # The experiment carries the trial's record from Evidence on, and its candidates.
  { sed -n '1,/^## Evidence$/p' "$dir/NOTE.md" | sed '$d'; sed -n '/^## Evidence$/,$p' "$trial/NOTE.md"; } > "$tmp/note"
  sed -i "s|^- Base: .*\$|&; adopted from night/$(basename "$trial")|" "$tmp/note"
  command cp -f "$tmp/note" "$dir/NOTE.md"
  if [ -d "$trial/candidate" ]; then command cp -R "$trial/candidate" "$dir/candidate"; fi
  if [ "$was" = "$now" ]; then
    git apply --binary --whitespace=nowarn "$trial/change.patch" \
      || die "the patch did not apply; run exp.sh drop, and the trial stays parked"
    command cp -f "$trial/run.sh" "$trial/result.json" "$dir/"
    set_status "$trial" adopted
    echo "Adopted night/$(basename "$trial") as $(basename "$dir"), change applied. The base is unchanged, so its numbers stand: run verify once, then exp.sh keep."
    return
  fi
  set_status "$trial" adopted
  if git apply --check --binary --whitespace=nowarn "$trial/change.patch" 2>/dev/null; then
    echo "Adopted night/$(basename "$trial") as $(basename "$dir"), nothing applied: the base moved ($was -> $now). A/B the candidate against the current tree, then land it; night/$(basename "$trial")/change.patch still applies."
  else
    echo "Adopted night/$(basename "$trial") as $(basename "$dir"), nothing applied: the base moved ($was -> $now) and its patch no longer applies. A/B the candidate against the current tree, then land it by hand."
  fi
}

cmd_check() {
  local dir want now
  dir="$(running)"
  if [ -n "$dir" ]; then
    mapfile -t kept < <(kept_for "$dir")
  else
    mapfile -t kept < <(exp_dirs kept)
  fi
  want="$(replay ${kept[@]+"${kept[@]}"})"
  now="$(snapshot)"
  if [ -n "$dir" ]; then
    echo "$(basename "$dir") is running; its unsaved change against the start:" >&2
    diff_ --stat "$want" "$now" >&2
    die "the tree is not at the base until $(basename "$dir") ends with keep, park, or drop"
  fi
  must_match "$want" "$now" "the project tree differs from the baseline plus the kept patches (files above)"
  echo "OK: the project tree equals baseline $(git rev-parse --short "$(baseline)") plus ${#kept[@]} kept patch(es)."
}

cmd_export() {
  local target="${1:-}" label tree out match d
  case "$target" in
    baseline) kept=(); label=baseline ;;
    best) mapfile -t kept < <(exp_dirs kept); label=best ;;
    [0-9][0-9][0-9]|night/[0-9][0-9][0-9])
      if [[ "$target" == night/* ]]; then match=("$work/$target"-*/); else match=("$exp/$target"-*/); fi
      [ -d "${match[0]}" ] || die "no $target"
      # From the item's own Base line, so later experiments and deliveries do not change what it rebuilds.
      replay_base="$(sed -n 's/^- Base: baseline \([0-9a-f]*\) .*$/\1/p' "${match[0]}NOTE.md")"
      [ -n "$replay_base" ] || die "${match[0]}NOTE.md has no Base line"
      mapfile -t kept < <(kept_of "${match[0]}NOTE.md")
      for d in ${kept[@]+"${kept[@]}"}; do [ -d "$d" ] || die "$d, named in its Base line, is missing"; done
      kept+=("${match[0]%/}")
      label="$(basename "${match[0]}")"
      [[ "$target" != night/* ]] || label="night-$label" ;;
    *) die "usage: exp.sh export <NNN|night/NNN|best|baseline>" ;;
  esac
  tree="$(replay ${kept[@]+"${kept[@]}"})"
  out="$work/local/replay/$label"
  rm -rf -- "$out"
  mkdir -p "$out"
  git archive "$tree" | tar -x -C "$out"
  echo "Exported $label to lx-work/local/replay/$label/"
}

cmd_delivered() {
  local head want d
  [ -z "$(running)" ] || die "$(basename "$(running)") is still running; end it first"
  mapfile -t kept < <(exp_dirs kept)
  [ "${#kept[@]}" -gt 0 ] || die "no kept experiments"
  want="$(replay "${kept[@]}")"
  head="$(git rev-parse --short HEAD)"
  [ "$(git rev-parse "HEAD^{tree}")" = "$want" ] \
    || die "HEAD $head is not the baseline plus the kept patches; re-baseline with the user (rules/experiment.md)"
  must_match "$want" "$(snapshot)" "the working tree differs from HEAD $head"
  for d in "${kept[@]}"; do set_status "$d" delivered; done
  sed -i "s/^- Baseline commit: .*\$/- Baseline commit: $head/" "$work/PROJECT.md"
  # The tree did not change, so the time breakdown stays as current as it was; only its baseline is renamed.
  sed -i "s/^- Measured on: baseline [0-9a-f]*/- Measured on: baseline $head/" "$work/MAP.md"
  echo "Marked ${#kept[@]} kept experiment(s) delivered; HEAD $head is the new baseline. Record it in DECISIONS.md, then commit lx-work."
}

case "${1:-}" in
  new) shift; cmd_new "$@" ;;
  try) shift; cmd_try "$@" ;;
  save) cmd_save ;;
  keep) cmd_keep ;;
  park) cmd_park ;;
  drop) cmd_drop ;;
  adopt) cmd_adopt "${2:-}" ;;
  delivered) cmd_delivered ;;
  check) cmd_check ;;
  export) cmd_export "${2:-}" ;;
  *) sed -n '2,15p' "$0"; exit 1 ;;
esac
