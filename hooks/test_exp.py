#!/usr/bin/env python3
"""Run exp.sh through the life of experiments and trials on a throwaway project.

Usage: test_exp.py
Builds, in a temporary directory, a project repo with one linked repo and an lx-work from the skeleton, then
drives every exp.sh subcommand (new, try, save, keep, park, drop, adopt, export, check, delivered) in the order
a project would, and checks exit codes, messages, the saved patches, the statuses in NOTE.md and INDEX.md, and
that the tree is back where exp.sh says. The pre-commit hook runs this file. Exits 1 at the first failed
expectation, with that command's output.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HOOKS = Path(__file__).resolve().parent
RULES = HOOKS.parent
EXP = RULES / "exp.sh"

APP = "def run(x):\n    return x\n"
CORE = "def core(x):\n    return x\n"
RUN_SH = "#!/usr/bin/env bash\n"


class Failure(Exception):
    pass


def read(path):
    return path.read_text(encoding="utf-8")


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append(path, text):
    write(path, read(path) + text)


def set_line(path, field, value):
    text, n = re.subn(rf"^- {field}: .*$", f"- {field}: {value}", read(path), flags=re.M)
    if not n:
        raise Failure(f"fixture: no '- {field}:' line in {path.name}; the skeleton changed")
    write(path, text)


def check(condition, what):
    if not condition:
        raise Failure(what)


class Fixture:
    """A project repo with app.py, a linked repo lib/ with core.py, and an lx-work from the skeleton."""

    def __init__(self, tmp):
        self.env = {**os.environ, "GIT_CONFIG_GLOBAL": str(tmp / "gitconfig"), "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@example.com",
                    "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@example.com"}
        write(tmp / "gitconfig", "")
        self.project = tmp / "project"
        self.lib = self.project / "lib"
        self.work = self.project / "lx-work"
        write(self.project / "app.py", APP)
        write(self.project / ".gitignore", "__pycache__/\nbuild/\n")
        write(self.lib / "core.py", CORE)
        self.git(self.lib, "init", "-q", "-b", "main")
        self.git(self.lib, "add", "-A")
        self.git(self.lib, "commit", "-qm", "lib base")
        self.git(self.project, "init", "-q", "-b", "main")
        write(self.project / ".git" / "info" / "exclude", "/lx-work/\n/lib/\n")
        self.git(self.project, "add", "-A")
        self.git(self.project, "commit", "-qm", "base")
        self.base = self.head(self.project)
        self.lbase = self.head(self.lib)
        shutil.copytree(RULES / "templates" / "work-skeleton", self.work)
        self.git(self.work, "init", "-q", "-b", "main")
        set_line(self.work / "PROJECT.md", "Baseline commit", self.base)
        set_line(self.work / "PROJECT.md", "Linked repos", f"lib {self.lbase}")
        set_line(self.work / "MAP.md", "Measured on", f"baseline commit {self.base}")
        self.runs = 0

    def git(self, cwd, *args):
        done = subprocess.run(["git", *args], cwd=cwd, env=self.env, check=True, capture_output=True, text=True)
        return done.stdout.strip()

    def head(self, repo):
        return self.git(repo, "rev-parse", "HEAD")[:7]

    def commit_all(self, repo, message):
        self.git(repo, "add", "-A")
        self.git(repo, "commit", "-qm", message)
        return self.head(repo)

    def run(self, args, code, *needles, cwd=None):
        """exp.sh with these arguments must exit with `code` and print every needle."""
        self.runs += 1
        done = subprocess.run(["bash", str(EXP), *args], cwd=cwd or self.project, env=self.env,
                              capture_output=True, text=True)
        out = done.stdout + done.stderr
        problems = [f"exit {done.returncode}, expected {code}"] if done.returncode != code else []
        problems += [f"output lacks {needle!r}" for needle in needles if needle not in out]
        if problems:
            raise Failure(f"exp.sh {' '.join(args)}: {'; '.join(problems)}\n{out.strip() or '(no output)'}")
        return out


def status_is(item, status):
    """NOTE.md and the INDEX.md beside the item both say so."""
    return f"- Status: {status}\n" in read(item / "NOTE.md") and f"| {item.name} | {status} |" in read(item.parent / "INDEX.md")


def scenario(f):
    project, lib, work = f.project, f.lib, f.work
    exp, trials, replay = work / "exp", work / "trials", work / "local" / "replay"

    # Usage and where it runs.
    f.run([], 1, "exp.sh new <1 to 4 words>")
    f.run(["check"], 1, "run exp.sh from inside the project repo", cwd=project.parent)
    f.run(["check"], 0, f"OK: the project tree equals baseline commit {f.base} plus 0 kept patch(es).",
          f"OK: linked repo lib/ equals its base commit {f.lbase} plus the kept patches for it.", cwd=work)

    # An experiment that is kept, changing both repos and adding a file.
    f.run(["new", "one", "two", "three", "four", "five"], 1, "usage: exp.sh new|try <1 to 4 words")
    f.run(["new", "First Change"], 0, f"Started exp/001-first-change on baseline commit {f.base} + kept none + lib {f.lbase}.")
    first = exp / "001-first-change"
    note = read(first / "NOTE.md")
    check("# 001-first-change\n" in note and f"- Base: baseline commit {f.base} + kept none + lib {f.lbase}\n" in note,
          f"NOTE.md of 001 is wrong:\n{note}")
    check(status_is(first, "running"), "001 is not marked running")
    f.run(["new", "again"], 1, "001-first-change is still running; end it first")
    append(project / "app.py", "# first\n")
    append(lib / "core.py", "# first lib\n")
    write(project / "new.py", "VALUE = 1\n")
    f.run(["save"], 0, "Saved 001-first-change/change.patch:", "Saved 001-first-change/change.lib.patch:")
    check("new.py" in read(first / "change.patch") and "# first lib" in read(first / "change.lib.patch"),
          "the saved patches of 001 lack its changes")
    f.run(["check"], 1, "the tree is not at the base until 001-first-change ends with keep, park, or drop")
    f.run(["keep"], 1, "save run.sh in lx-work/exp/001-first-change/ first")
    write(first / "run.sh", RUN_SH)
    write(first / "result.json", "{}\n")
    f.run(["keep"], 0, "Kept 001-first-change. Replay check passed.")
    check(status_is(first, "kept"), "001 is not marked kept")
    after_first = {"app": read(project / "app.py"), "core": read(lib / "core.py")}
    f.run(["check"], 0, "plus 1 kept patch(es).")

    # An experiment that is dropped; an ignored file written meanwhile gets a warning.
    f.run(["new", "second"], 0, "Started exp/002-second on baseline commit", "+ kept 001 + lib")
    second = exp / "002-second"
    append(project / "app.py", "# second\n")
    write(project / "build" / "out.bin", "x")  # ignored, so in no patch; __pycache__ is pruned from the warning
    earlier = time.time() - 10
    os.utime(work / "local" / "exp-start", (earlier, earlier))
    f.run(["drop"], 0, "Dropped 002-second; the tree is back to the baseline commit plus the kept patches.",
          "ignored files written during this experiment are not in its patch", "build/out.bin")
    shutil.rmtree(project / "build")
    check(read(project / "app.py") == after_first["app"], "drop did not restore app.py")
    check(status_is(second, "dropped"), "002 is not marked dropped")
    check((second / "change.patch").stat().st_size > 0, "drop saved no patch for 002")

    # A trial: parked with its change in the linked repo only, then adopted.
    f.run(["try", "Quick Idea"], 0, "Started trials/001-quick-idea on baseline commit", "+ kept 001 + lib")
    trial = trials / "001-quick-idea"
    f.run(["keep"], 1, "trials end with park or drop")
    f.run(["park"], 1, "save run.sh in lx-work/trials/001-quick-idea/ first")
    write(trial / "run.sh", RUN_SH)
    write(trial / "result.json", "{}\n")
    f.run(["park"], 1, "the tree holds no change; land the smallest change and verify it before parking")
    append(lib / "core.py", "# trial\n")
    f.run(["park"], 0, "Parked trials/001-quick-idea: its patches rebuild the change, and the tree is back to the base.")
    check(read(lib / "core.py") == after_first["core"], "park did not restore lib/core.py")
    check((trial / "change.lib.patch").stat().st_size > 0 and (trial / "change.patch").stat().st_size == 0,
          "the parked trial's patches are not one empty project patch and one linked-repo patch")
    check(status_is(trial, "parked"), "trial 001 is not marked parked")
    f.run(["park"], 1, "no running trial")
    f.run(["adopt", "999"], 1, "no trial 999")
    f.run(["adopt", "1"], 1, "usage: exp.sh adopt <trial number>")
    f.run(["adopt", "001"], 0, "Adopted trials/001-quick-idea as 003-quick-idea, change applied.")
    adopted = exp / "003-quick-idea"
    note = read(adopted / "NOTE.md")
    check("# 003-quick-idea\n" in note and "; adopted from trials/001-quick-idea\n" in note, f"NOTE.md of 003 is wrong:\n{note}")
    check(status_is(adopted, "running") and status_is(trial, "adopted"), "adopt did not set the statuses")
    check(read(lib / "core.py") == after_first["core"] + "# trial\n", "adopt did not apply the trial's change")
    check((adopted / "run.sh").is_file() and (adopted / "result.json").is_file(), "adopt did not copy run.sh and result.json")
    f.run(["adopt", "001"], 1, "trials/001-quick-idea is not parked")
    f.run(["keep"], 0, "Kept 003-quick-idea.")
    best = {"app": read(project / "app.py"), "core": read(lib / "core.py")}
    f.run(["check"], 0, "plus 2 kept patch(es).")

    # Exports rebuild each item from its own Base line.
    f.run(["export", "best"], 0, "Exported best to lx-work/local/replay/best/, linked repos included")
    check(read(replay / "best" / "app.py") == best["app"] and read(replay / "best" / "lib" / "core.py") == best["core"]
          and read(replay / "best" / "new.py") == "VALUE = 1\n", "export best differs from the tree")
    f.run(["export", "baseline-commit"], 0, "Exported baseline-commit")
    check(read(replay / "baseline-commit" / "app.py") == APP and read(replay / "baseline-commit" / "lib" / "core.py") == CORE
          and not (replay / "baseline-commit" / "new.py").exists(), "export baseline-commit is not the base")
    f.run(["export", "001"], 0, "Exported 001-first-change")
    check(read(replay / "001-first-change" / "app.py") == after_first["app"]
          and read(replay / "001-first-change" / "lib" / "core.py") == after_first["core"], "export 001 differs from the first change")
    f.run(["export", "002"], 0, "Exported 002-second")
    check(read(replay / "002-second" / "app.py") == after_first["app"] + "# second\n", "export 002 lacks the dropped change")
    f.run(["export", "trials/001"], 0, "Exported trial-001-quick-idea")
    check(read(replay / "trial-001-quick-idea" / "lib" / "core.py") == best["core"]
          and read(replay / "trial-001-quick-idea" / "app.py") == after_first["app"], "export trials/001 differs from the parked trial")
    f.run(["export", "777"], 1, "no 777")
    f.run(["export"], 1, "usage: exp.sh export")

    # Changes outside an experiment are caught in either repo.
    append(project / "app.py", "# stray\n")
    f.run(["check"], 1, "the project tree differs from its base commit plus the kept patches")
    write(project / "app.py", best["app"])
    append(lib / "core.py", "# stray\n")
    f.run(["check"], 1, "linked repo lib/ differs from its base commit plus the kept patches")
    write(lib / "core.py", best["core"])
    f.run(["check"], 0, "OK:")

    # Delivery: the user commits, exp.sh moves the base commits.
    f.run(["delivered"], 1, "is not its base commit plus the kept patches; agree on a new baseline commit")
    head = f.commit_all(project, "deliver")
    lhead = f.commit_all(lib, "deliver lib")
    f.run(["delivered"], 0, f"Marked 2 kept experiment(s) delivered; HEAD {head} is the new baseline commit "
          f"and the linked repos are at their new base commits (lib {lhead})")
    project_md = read(work / "PROJECT.md")
    check(f"- Baseline commit: {head}\n" in project_md and f"- Linked repos: lib {lhead}\n" in project_md,
          "delivered did not move the base commits in PROJECT.md")
    check(f"- Measured on: baseline commit {head}\n" in read(work / "MAP.md"), "delivered did not rename the baseline commit in MAP.md")
    check(status_is(first, "delivered") and status_is(adopted, "delivered"), "the kept experiments are not marked delivered")
    f.run(["check"], 0, f"OK: the project tree equals baseline commit {head} plus 0 kept patch(es).")
    f.run(["delivered"], 1, "no kept experiments")

    # A HEAD that is not the baseline commit stops a new experiment; numbers continue after delivery.
    write(project / "README.md", "x\n")
    f.commit_all(project, "unrelated")
    f.run(["new", "late"], 1, f"the project HEAD is not the baseline commit {head}")
    f.git(project, "reset", "-q", "--hard", "HEAD~1")
    f.run(["new", "after delivery"], 0, f"Started exp/004-after-delivery on baseline commit {head} + kept none + lib {lhead}.")
    f.run(["drop"], 0, "Dropped 004-after-delivery")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        fixture = None
        try:
            fixture = Fixture(Path(tmp))
            scenario(fixture)
        except Failure as e:
            print(f"FAIL {e}")
            return 1
        except subprocess.CalledProcessError as e:
            print(f"FAIL fixture git command {e.cmd}:\n{e.stderr}")
            return 1
    print(f"exp.sh: {fixture.runs} invocations behaved as expected on the throwaway project.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
