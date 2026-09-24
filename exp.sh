#!/usr/bin/env bash
# Experiments on the project's working tree, recorded in lx-work. The project repo gets no branch
# and no commit: each change is saved as a patch, and the tree must always equal the baseline
# commit plus the kept patches, applied in number order.
#   exp.sh new <1 to 4 words>          start the next experiment; checks the tree first
#   exp.sh save                        write the running experiment's change.patch, new files included
#   exp.sh keep                        end it and keep the change (needs run.sh and result.json)
#   exp.sh drop                        end it, reverse the change, and confirm the tree is back
#   exp.sh check                       confirm the tree equals the baseline plus the kept patches
#   exp.sh export <NNN|best|baseline>  write that code to lx-work/local/replay/<name>/
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project="$(dirname "$here")"
work="$project/lx-work"
exp="$work/exp"
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

replay() {  # tree id of the baseline commit plus the patches of the given experiment dirs
  local base d
  base="$(baseline)"
  rm -f "$tmp/replay"
  GIT_INDEX_FILE="$tmp/replay" git read-tree "$base"
  for d in "$@"; do
    [ -s "$d/change.patch" ] || continue
    GIT_INDEX_FILE="$tmp/replay" git apply --cached --binary --whitespace=nowarn "$d/change.patch" \
      || die "$(basename "$d")/change.patch does not apply to the replayed tree"
  done
  GIT_INDEX_FILE="$tmp/replay" git write-tree
}

exp_dirs() {  # experiment dirs in number order, with status $1 (any status when empty)
  local d
  for d in "$exp"/[0-9][0-9][0-9]-*/; do
    [ -f "$d/NOTE.md" ] || continue
    if [ -z "${1:-}" ] || grep -qx -- "- Status: $1" "$d/NOTE.md"; then echo "${d%/}"; fi
  done
}

num_of() { local n; n="$(basename "$1")"; echo "$((10#${n%%-*}))"; }

kept_below() {  # kept experiment dirs numbered below $1
  local d
  while IFS= read -r d; do
    if [ "$(num_of "$d")" -lt "$1" ]; then echo "$d"; fi
  done < <(exp_dirs kept)
}

running() { local all; all="$(exp_dirs running)"; printf '%s' "${all%%$'\n'*}"; }

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

set_status() {  # $1 experiment dir, $2 new status, in NOTE.md and exp/INDEX.md
  local name
  name="$(basename "$1")"
  sed -i "s/^- Status: running\$/- Status: $2/" "$1/NOTE.md"
  sed -i "s/^| $name | running |/| $name | $2 |/" "$exp/INDEX.md"
}

cmd_new() {
  local name base now want list="" last=0 n num dir d
  name="$(printf '%s' "$*" | tr 'A-Z ' 'a-z-' | tr -s '-')"
  name="${name#-}"; name="${name%-}"
  [[ "$name" =~ ^[a-z0-9]+(-[a-z0-9]+){0,3}$ ]] || die "usage: exp.sh new <1 to 4 words of letters and digits>"
  [ -z "$(running)" ] || die "$(basename "$(running)") is still running; keep or drop it first"
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
  for n in $( { ls "$exp"; cat "$exp/INDEX.md" "$exp/ARCHIVE.md" 2>/dev/null; } \
              | grep -oE '(^|[^0-9])[0-9]{3}-[a-z0-9]' | grep -oE '[0-9]{3}' || true); do
    if [ $((10#$n)) -gt "$last" ]; then last=$((10#$n)); fi
  done
  [ "$last" -lt 999 ] || die "experiment numbers are exhausted"
  num="$(printf '%03d' $((last + 1)))"
  dir="$exp/$num-$name"
  mkdir "$dir"
  sed -e "s/^# NNN-name\$/# $num-$name/" \
      -e "s/^- Date: <YYYY-MM-DD>\$/- Date: $(date +%F)/" \
      -e "s/^- Base: .*\$/- Base: baseline ${base:0:7} + kept ${list:-none}/" \
      -e "s/^- Status: running | kept | dropped\$/- Status: running/" \
      "$here/templates/exp-note.md" > "$dir/NOTE.md"
  if grep -qE '^(# NNN-name|- (Date|Base|Status): .*[<|])' "$dir/NOTE.md"; then
    rm -rf -- "$dir"
    die "templates/exp-note.md no longer matches exp.sh; fix one of them"
  fi
  echo "| $num-$name | running | | | | | |" >> "$exp/INDEX.md"
  mkdir -p "$work/local" && touch "$work/local/exp-start"
  echo "Started $num-$name on baseline ${base:0:7} + kept ${list:-none}."
}

cmd_save() {
  local dir start now stat
  dir="$(running)"
  [ -n "$dir" ] || die "no running experiment"
  mapfile -t kept < <(kept_below "$(num_of "$dir")")
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
  echo "Kept $(basename "$dir"). Replay check passed. Fill in the results, then commit lx-work."
}

cmd_drop() {
  local dir start now
  dir="$(running)"
  [ -n "$dir" ] || die "no running experiment"
  cmd_save
  if [ -s "$dir/change.patch" ]; then
    git apply -R --binary --whitespace=nowarn "$dir/change.patch" || die "could not reverse the change; the tree is unchanged"
  fi
  mapfile -t kept < <(kept_below "$(num_of "$dir")")
  start="$(replay ${kept[@]+"${kept[@]}"})"
  now="$(snapshot)"
  must_match "$start" "$now" "after reversing, the tree still differs from the start of the experiment"
  set_status "$dir" dropped
  warn_ignored
  echo "Dropped $(basename "$dir"); the tree is back to the baseline plus the kept patches. Record why, then commit lx-work."
}

cmd_check() {
  local dir want now
  dir="$(running)"
  if [ -n "$dir" ]; then
    mapfile -t kept < <(kept_below "$(num_of "$dir")")
  else
    mapfile -t kept < <(exp_dirs kept)
  fi
  want="$(replay ${kept[@]+"${kept[@]}"})"
  now="$(snapshot)"
  if [ -n "$dir" ]; then
    echo "$(basename "$dir") is running; its unsaved change against the start:"
    diff_ --stat "$want" "$now"
    return
  fi
  must_match "$want" "$now" "the project tree differs from the baseline plus the kept patches (files above)"
  echo "OK: the project tree equals baseline $(git rev-parse --short "$(baseline)") plus ${#kept[@]} kept patch(es)."
}

cmd_export() {
  local target="${1:-}" label tree out match
  case "$target" in
    baseline) kept=(); label=baseline ;;
    best) mapfile -t kept < <(exp_dirs kept); label=best ;;
    [0-9][0-9][0-9])
      match=("$exp/$target"-*/)
      [ -d "${match[0]}" ] || die "no experiment $target"
      mapfile -t kept < <(kept_below "$((10#$target))")
      kept+=("${match[0]%/}")
      label="$(basename "${match[0]}")" ;;
    *) die "usage: exp.sh export <NNN|best|baseline>" ;;
  esac
  tree="$(replay ${kept[@]+"${kept[@]}"})"
  out="$work/local/replay/$label"
  rm -rf -- "$out"
  mkdir -p "$out"
  git archive "$tree" | tar -x -C "$out"
  echo "Exported $label to lx-work/local/replay/$label/"
}

case "${1:-}" in
  new) shift; cmd_new "$@" ;;
  save) cmd_save ;;
  keep) cmd_keep ;;
  drop) cmd_drop ;;
  check) cmd_check ;;
  export) cmd_export "${2:-}" ;;
  *) sed -n '2,10p' "$0"; exit 1 ;;
esac
