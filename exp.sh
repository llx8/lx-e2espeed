#!/usr/bin/env bash
# Experiments on the project's working tree, recorded in lx-work. The project repo gets no branch
# and no commit: each change is saved as a patch, and the tree must always equal the baseline
# commit plus the kept patches, applied in number order.
# Linked repos are git checkouts at the project top that the project ignores, such as a sibling
# algorithm repo or its symlink, named in lx-work/PROJECT.md as '- Linked repos: <dir> <commit>, ...'
# or none. Each is handled like the project tree, from that commit (its base commit), and an item's
# change to it is saved as change.<dir>.patch beside change.patch.
# Trials of a self-iteration run (rules/self-iteration.md) live in lx-work/trials/ and never stay in the tree: each is parked.
#   exp.sh new <1 to 4 words>          start the next experiment; checks the tree first
#   exp.sh try <1 to 4 words>          start the next trial, on best (the baseline commit plus all kept patches)
#   exp.sh save                        write the running item's change.patch, new files included, and change.<dir>.patch per changed linked repo
#   exp.sh keep                        end the experiment and keep the change (needs run.sh and result.json)
#   exp.sh park                        end the trial: save its patch, reverse it, confirm the tree is back
#   exp.sh drop                        end either, reverse the change, and confirm the tree is back
#   exp.sh adopt <NNN>                 turn parked trial NNN into the next experiment
#   exp.sh delivered                   the user committed the kept patches: HEAD becomes the baseline commit, each linked repo's HEAD its base commit
#   exp.sh check                       confirm the tree equals the baseline commit plus the kept patches; fails while an item runs
#   exp.sh export <NNN|trials/NNN|best|baseline-commit>  write that code to lx-work/local/replay/<name>/, each linked repo in <name>/<dir>/
# The baseline commit is only where experiments start; the baseline that results compare with is in PROJECT.md (Metric).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
die() { echo "error: $*" >&2; exit 1; }
# Every project shares this rules directory, so the project is the git repo this runs in, not a parent of
# this script; from inside lx-work, its own repo, the project is one level up.
project="$(git rev-parse --show-toplevel 2>/dev/null)" || die "run exp.sh from inside the project repo"
if [ "$(basename "$project")" = lx-work ]; then project="$(dirname "$project")"; fi
[ "$project" != "$here" ] || die "run exp.sh from the project repo, not from lx-e2espeed"
work="$project/lx-work"
exp="$work/exp"
trials="$work/trials"
[ -d "$exp" ] || die "$exp not found; run $here/setup.sh project first"
cd "$project"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

repos=(.)            # "." is the project; load_linked adds the linked repos
declare -A lbase=()  # linked repo -> its base commit

git_in() { local r="$1"; shift; git -C "$project/$r" "$@"; }  # $1 repo, then a git command

patch_of() { if [ "$1" = . ]; then echo change.patch; else echo "change.$1.patch"; fi; }

name_of() { if [ "$1" = . ]; then echo "the project tree"; else echo "linked repo $1/"; fi; }

load_linked() {  # fills repos and lbase from '- Linked repos:' in PROJECT.md; a missing line means none
  local line entry dir c rest t
  local -a entries
  line="$(sed -n 's/^- Linked repos: *//p' "$work/PROJECT.md")"
  line="${line%%$'\n'*}"
  line="${line%"${line##*[! ]}"}"
  [ -n "$line" ] && [ "$line" != none ] || return 0
  IFS=, read -ra entries <<< "$line"
  for entry in "${entries[@]}"; do
    read -r dir c rest <<< "$entry" || true
    [[ "${dir:-}" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]*$ && "${c:-}" =~ ^[0-9a-f]{7,40}$ && -z "${rest:-}" ]] \
      || die "lx-work/PROJECT.md: '- Linked repos:' must be none or '<dir> <commit>' pairs separated by commas, not '$entry'"
    case "$dir" in lx-work|lx-e2espeed) die "$dir is not a linked repo" ;; esac
    [ -z "${lbase[$dir]:-}" ] || die "linked repo $dir/ is named twice in PROJECT.md"
    [ -d "$dir" ] || die "linked repo $dir/ is not a directory at the project top"
    t="$(git_in "$dir" rev-parse --show-toplevel 2>/dev/null)" && [ "$(cd "$t" && pwd -P)" = "$(cd "$dir" && pwd -P)" ] \
      || die "linked repo $dir/ is not the top of its own git checkout"
    # Otherwise the project's own snapshot would hold it too.
    git check-ignore -q "$dir" || die "the project does not ignore linked repo $dir/; add it to its .gitignore or .git/info/exclude with the user's okay"
    lbase[$dir]="$(git_in "$dir" rev-parse -q --verify "$c^{commit}")" || die "commit $c is not in linked repo $dir/"
    repos+=("$dir")
  done
}

diff_() {  # $1 repo, then diff arguments; plain patches whatever the user's git config says
  local r="$1"
  shift
  git_in "$r" -c diff.noprefix=false -c diff.mnemonicPrefix=false diff --no-color --no-ext-diff --no-textconv "$@"
}

baseline() {
  local c
  c="$(sed -n 's/^- Baseline commit: *\([0-9a-f]\{7,40\}\) *$/\1/p' "$work/PROJECT.md")"
  c="${c%%$'\n'*}"
  [ -n "$c" ] || die "lx-work/PROJECT.md has no '- Baseline commit: <hash>' line"
  git rev-parse -q --verify "$c^{commit}" || die "baseline commit $c is not in the project repo"
}

snapshot() {  # $1 repo: tree id of its working tree; the repo's own index is never touched
  local snap
  snap="$tmp/snap.$(patch_of "$1")"
  (
    cd "$project/$1"
    index="$(git rev-parse --git-path index)"
    rm -f "$snap"
    # -p keeps the index's mtime, which git's racy-file check compares with: with a fresh mtime, a file changed
    # to the same size in the second it was staged would be read from the stale index entry.
    if [ -f "$index" ]; then command cp -fp "$index" "$snap"; fi
    # Python bytecode is a cache, never a change: a linked repo without a .gitignore once put 14 .pyc files into a
    # kept patch after a run that imported its code.
    GIT_INDEX_FILE="$snap" git add -A -- . ':(exclude,glob)**/__pycache__/**' ':(exclude,glob)**/*.pyc'
    GIT_INDEX_FILE="$snap" git write-tree
  )
}

declare -A replay_bases=()  # set by export: rebuild an item from the commits in its own Base line

base_of() {  # $1 repo: the commit its replay starts from
  if [ -n "${replay_bases[$1]:-}" ]; then echo "${replay_bases[$1]}"
  elif [ "$1" = . ]; then baseline
  else echo "${lbase[$1]}"; fi
}

replay() {  # $1 repo, then experiment dirs: tree id of its base commit plus those dirs' patches for it
  local r="$1" base p index d
  shift
  base="$(base_of "$r")"
  p="$(patch_of "$r")"
  index="$tmp/replay.$p"
  rm -f "$index"
  GIT_INDEX_FILE="$index" git_in "$r" read-tree "$base"
  for d in "$@"; do
    [ -s "$d/$p" ] || continue
    GIT_INDEX_FILE="$index" git_in "$r" apply --cached --binary --whitespace=nowarn "$d/$p" \
      || die "$(basename "$d")/$p does not apply to the replayed tree"
  done
  GIT_INDEX_FILE="$index" git_in "$r" write-tree
}

items() {  # dirs under $1 (exp or trials) in number order, with status $2 (any status when empty)
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

running() { local all; all="$(items "$exp" running; items "$trials" running)"; printf '%s' "${all%%$'\n'*}"; }

is_trial() { [[ "$1" == "$trials"/* ]]; }

kept_for() {  # kept experiments the item $1 builds on: all of them for a trial
  if is_trial "$1"; then exp_dirs kept; else kept_below "$(num_of "$1")"; fi
}

kept_of() {  # experiment dirs named in the Base line of NOTE.md $1; a missing one prints its pattern
  local n m
  # Notes written before 2026-10-03 say "baseline <hash>" without "commit".
  for n in $(sed -n 's/^- Base: baseline \(commit \)\{0,1\}[0-9a-f]* + kept \([0-9 ]*[0-9]\).*$/\2/p' "$1"); do
    m=("$exp/$n"-*/)
    echo "${m[0]%/}"
  done
}

must_match() {  # $1 repo, $2 expected tree, $3 actual tree, $4 message
  if [ "$2" != "$3" ]; then
    diff_ "$1" --stat "$2" "$3" >&2
    die "$4"
  fi
}

match_all() {  # $1 message after the repo's name, then experiment dirs: every repo must equal its replay of them
  local msg="$1" r want now
  shift
  for r in "${repos[@]}"; do
    # Assignments, so a failing replay stops the script instead of passing an empty tree id on.
    want="$(replay "$r" "$@")"
    now="$(snapshot "$r")"
    must_match "$r" "$want" "$now" "$(name_of "$r") $msg"
  done
}

has_change() {  # whether item dir $1 holds a non-empty patch for any repo
  local r
  for r in "${repos[@]}"; do
    if [ -s "$1/$(patch_of "$r")" ]; then return 0; fi
  done
  return 1
}

patches_check() {  # $1 item dir, then git apply flags (-R to reverse): whether each repo's patch from it applies
  local dir="$1" r p
  shift
  for r in "${repos[@]}"; do
    p="$dir/$(patch_of "$r")"
    if [ -s "$p" ]; then git_in "$r" apply --check "$@" --binary --whitespace=nowarn "$p" || return 1; fi
  done
}

apply_all() {  # the same arguments: apply every repo's patch, or none when any of them would fail
  local dir="$1" r p
  patches_check "$@" || return 1
  shift
  for r in "${repos[@]}"; do
    p="$dir/$(patch_of "$r")"
    if [ -s "$p" ]; then
      git_in "$r" apply "$@" --binary --whitespace=nowarn "$p" \
        || die "$(basename "$dir")/$(patch_of "$r") passed the check but did not apply; look at $(name_of "$r") with the user"
    fi
  done
}

warn_ignored() {  # ignored files written since the experiment started are in no patch
  local files="" found r l
  local -a prune
  [ -f "$work/local/exp-start" ] || return 0
  for r in "${repos[@]}"; do
    prune=(-path ./.git -o -name __pycache__)
    if [ "$r" = . ]; then
      prune+=(-o -path ./lx-work -o -path ./lx-e2espeed)
      for l in "${repos[@]:1}"; do prune+=(-o -path "./$l"); done
    fi
    found="$(cd "$project/$r" && find . \( "${prune[@]}" \) -prune -o -type f -newer "$work/local/exp-start" -print 2>/dev/null \
             | git check-ignore --stdin 2>/dev/null | sed "s|^\./|$r/|" || true)"
    if [ -n "$found" ]; then files+="$found"$'\n'; fi
  done
  [ -n "$files" ] || return 0
  echo "warning: ignored files written during this experiment are not in its patch (up to 10):" >&2
  printf '%s' "$files" | head -n 10 >&2
  echo "Build outputs and caches can change the next bench: make run.sh rebuild them, and after a drop remove the ones this experiment made." >&2
}

set_status() {  # $1 experiment or trial dir, $2 new status, in NOTE.md and its INDEX.md
  local name
  name="$(basename "$1")"
  sed -i "s/^- Status: [a-z]*\$/- Status: $2/" "$1/NOTE.md"
  sed -i "s/^| $name | [a-z]* |/| $name | $2 |/" "$(dirname "$1")/INDEX.md"
}

start() {  # $1 exp or trials, then 1 to 4 words: check the tree and create the next dir there
  local root="$1" name base list="" linked="" last=0 n num dir d r cells
  shift
  name="$(printf '%s' "$*" | tr 'A-Z ' 'a-z-' | tr -s '-')"
  name="${name#-}"; name="${name%-}"
  [[ "$name" =~ ^[a-z0-9]+(-[a-z0-9]+){0,3}$ ]] || die "usage: exp.sh new|try <1 to 4 words of letters and digits>"
  [ -z "$(running)" ] || die "$(basename "$(running)") is still running; end it first"
  base="$(baseline)"
  [ "$(git rev-parse HEAD)" = "$base" ] \
    || die "the project HEAD is not the baseline commit ${base:0:7}; see 'When the project changes underneath' in rules/experiment.md"
  for r in "${repos[@]:1}"; do
    [ "$(git_in "$r" rev-parse HEAD)" = "${lbase[$r]}" ] \
      || die "the HEAD of linked repo $r/ is not its base commit ${lbase[$r]:0:7} (PROJECT.md, Linked repos); see 'When the project changes underneath' in rules/experiment.md"
    linked+=" + $r ${lbase[$r]:0:7}"
  done
  mapfile -t kept < <(exp_dirs kept)
  match_all "differs from its base commit plus the kept patches (files above); ask the user before going on" ${kept[@]+"${kept[@]}"}
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
      -e "s/^- Base: .*\$/- Base: baseline commit ${base:0:7} + kept ${list:-none}$linked/" \
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
  echo "Started $(basename "$root")/$num-$name on baseline commit ${base:0:7} + kept ${list:-none}$linked."
}

cmd_new() { start "$exp" "$@"; }

cmd_try() { start "$trials" "$@"; }

cmd_save() {
  local dir r p start now stat
  dir="$(running)"
  [ -n "$dir" ] || die "no running experiment"
  mapfile -t kept < <(kept_for "$dir")
  for r in "${repos[@]}"; do
    p="$(patch_of "$r")"
    start="$(replay "$r" ${kept[@]+"${kept[@]}"})"
    now="$(snapshot "$r")"
    stat="$(diff_ "$r" --shortstat "$start" "$now")"
    if [ "$r" != . ] && [ -z "$stat" ]; then
      rm -f "$dir/$p"  # a linked repo the item does not change gets no patch file
      continue
    fi
    diff_ "$r" --binary --full-index "$start" "$now" > "$dir/$p"
    echo "Saved $(basename "$dir")/$p:${stat:- no changes}"
  done
}

cmd_keep() {
  local dir f
  dir="$(running)"
  [ -n "$dir" ] || die "no running experiment"
  ! is_trial "$dir" || die "trials end with park or drop; the user adopts them into experiments"
  for f in run.sh result.json; do
    [ -f "$dir/$f" ] || die "save $f in lx-work/exp/$(basename "$dir")/ first"
  done
  cmd_save
  mapfile -t kept < <(kept_below "$(num_of "$dir")")
  match_all "is not what the saved patches rebuild; nothing was changed" ${kept[@]+"${kept[@]}"} "$dir"
  set_status "$dir" kept
  warn_ignored
  echo "Kept $(basename "$dir"). Replay check passed. Fill in the results, re-run the stage timing for MAP.md if the patch changed a file it points to, then commit lx-work."
}

cmd_drop() {
  local dir
  dir="$(running)"
  [ -n "$dir" ] || die "no running experiment"
  cmd_save
  apply_all "$dir" -R || die "could not reverse the change; the tree is unchanged"
  mapfile -t kept < <(kept_for "$dir")
  match_all "still differs from the start of the experiment after reversing" ${kept[@]+"${kept[@]}"}
  set_status "$dir" dropped
  warn_ignored
  echo "Dropped $(basename "$dir"); the tree is back to the baseline commit plus the kept patches. Record why, then commit lx-work."
}

cmd_park() {
  local dir f
  dir="$(running)"
  [ -n "$dir" ] && is_trial "$dir" || die "no running trial"
  for f in run.sh result.json; do
    [ -f "$dir/$f" ] || die "save $f in lx-work/trials/$(basename "$dir")/ first"
  done
  cmd_save
  has_change "$dir" || die "the tree holds no change; land the smallest change and verify it before parking"
  mapfile -t kept < <(exp_dirs kept)
  match_all "is not what the saved patches rebuild; nothing was changed" ${kept[@]+"${kept[@]}"} "$dir"
  apply_all "$dir" -R || die "could not reverse the change; the tree is unchanged"
  match_all "still differs from the base after reversing" ${kept[@]+"${kept[@]}"}
  set_status "$dir" parked
  warn_ignored
  echo "Parked trials/$(basename "$dir"): its patches rebuild the change, and the tree is back to the base. Record the result, then commit lx-work."
}

cmd_adopt() {
  local match trial was dir now words
  [[ "${1:-}" =~ ^[0-9]{3}$ ]] || die "usage: exp.sh adopt <trial number>"
  match=("$trials/$1"-*/)
  [ -d "${match[0]}" ] || die "no trial $1"
  trial="${match[0]%/}"
  grep -qx -- "- Status: parked" "$trial/NOTE.md" || die "trials/$(basename "$trial") is not parked"
  was="$(sed -n 's/^- Base: //p' "$trial/NOTE.md")"
  words="$(basename "$trial" | cut -d- -f2- | tr '-' ' ')"
  start "$exp" "$words" > /dev/null
  dir="$(running)"
  now="$(sed -n 's/^- Base: //p' "$dir/NOTE.md")"
  # The experiment carries the trial's record from Evidence on, and its candidates.
  { sed -n '1,/^## Evidence$/p' "$dir/NOTE.md" | sed '$d'; sed -n '/^## Evidence$/,$p' "$trial/NOTE.md"; } > "$tmp/note"
  sed -i "s|^- Base: .*\$|&; adopted from trials/$(basename "$trial")|" "$tmp/note"
  command cp -f "$tmp/note" "$dir/NOTE.md"
  if [ -d "$trial/candidate" ]; then command cp -R "$trial/candidate" "$dir/candidate"; fi
  if [ "$was" = "$now" ]; then
    apply_all "$trial" || die "the patches did not apply; run exp.sh drop, and the trial stays parked"
    command cp -f "$trial/run.sh" "$trial/result.json" "$dir/"
    set_status "$trial" adopted
    echo "Adopted trials/$(basename "$trial") as $(basename "$dir"), change applied. The base is unchanged, so its numbers stand: run verify once, then exp.sh keep."
    return
  fi
  set_status "$trial" adopted
  if patches_check "$trial" 2>/dev/null; then
    echo "Adopted trials/$(basename "$trial") as $(basename "$dir"), nothing applied: the base moved ($was -> $now). A/B the candidate against the current tree, then land it; the patches in trials/$(basename "$trial")/ still apply."
  else
    echo "Adopted trials/$(basename "$trial") as $(basename "$dir"), nothing applied: the base moved ($was -> $now) and its patches no longer apply. A/B the candidate against the current tree, then land it by hand."
  fi
}

cmd_check() {
  local dir r want now
  dir="$(running)"
  if [ -n "$dir" ]; then
    mapfile -t kept < <(kept_for "$dir")
    echo "$(basename "$dir") is running; its unsaved change against the start:" >&2
    for r in "${repos[@]}"; do
      want="$(replay "$r" ${kept[@]+"${kept[@]}"})"
      now="$(snapshot "$r")"
      if [ "$want" != "$now" ]; then echo "$(name_of "$r"):" >&2; diff_ "$r" --stat "$want" "$now" >&2; fi
    done
    die "the tree is not at the base until $(basename "$dir") ends with keep, park, or drop"
  fi
  mapfile -t kept < <(exp_dirs kept)
  match_all "differs from its base commit plus the kept patches (files above)" ${kept[@]+"${kept[@]}"}
  echo "OK: the project tree equals baseline commit $(git rev-parse --short "$(baseline)") plus ${#kept[@]} kept patch(es)."
  for r in "${repos[@]:1}"; do
    echo "OK: linked repo $r/ equals its base commit $(git_in "$r" rev-parse --short "${lbase[$r]}") plus the kept patches for it."
  done
}

cmd_export() {
  local target="${1:-}" label out match d r c note
  local -a parts=("${repos[@]}")
  local -A trees=()
  case "$target" in
    baseline-commit) kept=(); label=baseline-commit ;;
    best) mapfile -t kept < <(exp_dirs kept); label=best ;;
    [0-9][0-9][0-9]|trials/[0-9][0-9][0-9])
      if [[ "$target" == trials/* ]]; then match=("$work/$target"-*/); else match=("$exp/$target"-*/); fi
      [ -d "${match[0]}" ] || die "no $target"
      note="${match[0]}NOTE.md"
      label="$(basename "${match[0]}")"
      [[ "$target" != trials/* ]] || label="trial-$label"
      # From the item's own Base line, so later experiments and deliveries do not change what it rebuilds.
      replay_bases[.]="$(sed -n 's/^- Base: baseline \(commit \)\{0,1\}\([0-9a-f]*\) .*$/\2/p' "$note")"
      [ -n "${replay_bases[.]}" ] || die "${note} has no Base line"
      parts=(.)
      for r in "${repos[@]:1}"; do
        c="$(sed -n "s/^- Base: baseline .* + ${r//./\\.} \([0-9a-f]\{7,40\}\)\([^0-9a-f].*\)\{0,1\}\$/\1/p" "$note")"
        if [ -z "$c" ]; then
          echo "warning: the Base line of $label names no commit of linked repo $r/, so the export leaves $r/ out" >&2
          continue
        fi
        replay_bases[$r]="$(git_in "$r" rev-parse -q --verify "$c^{commit}")" || die "commit $c of linked repo $r/, named in its Base line, is not in that repo"
        parts+=("$r")
      done
      mapfile -t kept < <(kept_of "$note")
      for d in ${kept[@]+"${kept[@]}"}; do [ -d "$d" ] || die "$d, named in its Base line, is missing"; done
      kept+=("${match[0]%/}") ;;
    *) die "usage: exp.sh export <NNN|trials/NNN|best|baseline-commit>" ;;
  esac
  for r in "${parts[@]}"; do trees[$r]="$(replay "$r" ${kept[@]+"${kept[@]}"})"; done
  out="$work/local/replay/$label"
  rm -rf -- "$out"
  for r in "${parts[@]}"; do
    mkdir -p "$out/$r"
    git_in "$r" archive "${trees[$r]}" | tar -x -C "$out/$r"
  done
  echo "Exported $label to lx-work/local/replay/$label/${parts[1]+, linked repos included}"
}

cmd_delivered() {
  local r head want now d linked=""
  local -A heads=()
  [ -z "$(running)" ] || die "$(basename "$(running)") is still running; end it first"
  mapfile -t kept < <(exp_dirs kept)
  [ "${#kept[@]}" -gt 0 ] || die "no kept experiments"
  for r in "${repos[@]}"; do
    want="$(replay "$r" "${kept[@]}")"
    head="$(git_in "$r" rev-parse --short HEAD)"
    [ "$(git_in "$r" rev-parse "HEAD^{tree}")" = "$want" ] \
      || die "the HEAD $head of $(name_of "$r") is not its base commit plus the kept patches; agree on a new baseline commit with the user (rules/experiment.md)"
    now="$(snapshot "$r")"
    must_match "$r" "$want" "$now" "the working tree of $(name_of "$r") differs from its HEAD $head"
    heads[$r]="$head"
  done
  for d in "${kept[@]}"; do set_status "$d" delivered; done
  sed -i "s/^- Baseline commit: .*\$/- Baseline commit: ${heads[.]}/" "$work/PROJECT.md"
  for r in "${repos[@]:1}"; do linked+="${linked:+, }$r ${heads[$r]}"; done
  if [ -n "$linked" ]; then sed -i "s/^- Linked repos: .*\$/- Linked repos: $linked/" "$work/PROJECT.md"; fi
  # The tree did not change, so the time breakdown stays as current as it was; only its baseline commit is renamed.
  sed -i "s/^- Measured on: baseline \(commit \)\{0,1\}[0-9a-f]*/- Measured on: baseline commit ${heads[.]}/" "$work/MAP.md"
  echo "Marked ${#kept[@]} kept experiment(s) delivered; HEAD ${heads[.]} is the new baseline commit${linked:+ and the linked repos are at their new base commits ($linked)}, and the baseline (PROJECT.md, Metric) stays. Record it in DECISIONS.md, then commit lx-work."
}

case "${1:-}" in
  new|try|save|keep|park|drop|adopt|delivered|check|export) load_linked ;;
  *) sed -n '/^set -euo pipefail$/q; 2,$p' "$0"; exit 1 ;;
esac
case "$1" in
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
esac
