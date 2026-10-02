#!/usr/bin/env bash
# Set up lx-e2espeed.
#   setup.sh global   once per machine, and after hooks/dispatch.py, skills/, or vendor/skills/ change:
#                     install the Claude Code hooks and skills into ~/.claude
#   setup.sh project  once per project, from inside the project repo: exclude the lx dirs, create lx-work
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

install_global() {
  mkdir -p "$HOME/.claude/hooks"
  cp "$here/hooks/dispatch.py" "$HOME/.claude/hooks/lx_dispatch.py"
  # Merge into settings.json without touching existing hooks; keep a backup first.
  python3 - <<'PY'
import json
import shutil
from pathlib import Path

path = Path.home() / ".claude" / "settings.json"
data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
if path.exists():
    shutil.copy(path, path.with_name("settings.json.bak"))
hooks = data.setdefault("hooks", {})


def ensure(event, matcher, mode):
    command = f'python3 "$HOME/.claude/hooks/lx_dispatch.py" {mode}'
    groups = hooks.setdefault(event, [])
    if any(h.get("command") == command for g in groups for h in g.get("hooks", [])):
        return
    group = {"hooks": [{"type": "command", "command": command, "timeout": 60}]}
    if matcher:
        group["matcher"] = matcher
    groups.append(group)


ensure("SessionStart", None, "session-start")
ensure("PostToolUse", "Edit|Write|MultiEdit|NotebookEdit|Bash", "post-edit")
path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
  # Skills go to the personal skills directory: Claude Code does not look for skills inside
  # git-ignored directories such as <project>/lx-e2espeed/. A marker file tells ours apart, so
  # a skill of the same name from anywhere else is never overwritten.
  local src dst name
  mkdir -p "$HOME/.claude/skills"
  for dst in "$HOME/.claude/skills"/*/; do
    name="$(basename "$dst")"
    if [ -f "$dst.lx-e2espeed" ] && [ ! -d "$here/skills/$name" ] && [ ! -d "$here/vendor/skills/$name" ]; then
      rm -rf -- "$dst"
    fi
  done
  for src in "$here"/skills/*/ "$here"/vendor/skills/*/; do
    name="$(basename "$src")"
    dst="$HOME/.claude/skills/$name"
    if [ -e "$dst" ] && [ ! -f "$dst/.lx-e2espeed" ]; then
      echo "warning: ~/.claude/skills/$name is not from lx-e2espeed; left unchanged" >&2
      continue
    fi
    rm -rf -- "$dst"
    cp -R "$src" "$dst"
    find "$dst" -name __pycache__ -prune -exec rm -rf {} +
    touch "$dst/.lx-e2espeed"
  done
  # The SessionStart hook compares this with the clone, so an outdated install is noticed.
  python3 "$here/hooks/dispatch.py" stamp "$here" > "$HOME/.claude/hooks/lx_installed"
  echo "Installed ~/.claude/hooks/lx_dispatch.py, registered its hooks, and installed the skills in skills/ and vendor/skills/."
  echo "Backup of the previous settings: ~/.claude/settings.json.bak"
}

setup_project() {
  local root exclude entry version
  root="$(git rev-parse --show-toplevel 2>/dev/null)" || {
    echo "error: run this from inside the project's git repo; without one, see onboarding step 1" >&2; exit 1; }
  if [ "$here" != "$(cd "$root" && pwd -P)/lx-e2espeed" ]; then
    echo "error: this clone must sit at $root/lx-e2espeed" >&2; exit 1
  fi
  cd "$root"
  exclude="$(git rev-parse --git-path info/exclude)"
  mkdir -p "$(dirname "$exclude")"
  # The lx dirs stay out of the project repo. Python caches appear after any run; excluding them
  # keeps exp.sh's tree check from reporting them as changes. The exclude file is local and never pushed.
  for entry in /lx-e2espeed/ /lx-work/ __pycache__/ '*.py[cod]'; do
    grep -qxF "$entry" "$exclude" 2>/dev/null || echo "$entry" >> "$exclude"
  done
  if [ -e "$here/.git" ]; then
    git -C "$here" config core.hooksPath hooks/git
  fi
  if [ -e lx-work ]; then
    echo "lx-work already exists; left unchanged."
  else
    cp -R "$here/templates/work-skeleton" lx-work
    mkdir -p lx-work/local/logs lx-work/local/profiles lx-work/local/scratch
    # Record which rules this project started with, so its results can be traced to them, and which
    # skeleton lx-work matches, so the sync after a pull knows where to diff from (rules/naming.md).
    version="$(git -C "$here" rev-parse --short HEAD 2>/dev/null || echo uncommitted)"
    sed -i "s/<rules-version>/$version/" lx-work/PROJECT.md lx-work/STATUS.md
    git -C lx-work init -q -b main
    git -C lx-work config core.hooksPath ../lx-e2espeed/hooks/git
    echo "Created lx-work."
  fi
  echo "Done: $root is set up (onboarding step 1). Step 2 starts once the user says so."
}

case "${1:-}" in
  global) install_global ;;
  project) setup_project ;;
  *) sed -n '2,4p' "$0"; exit 1 ;;
esac
