---
tier: on-demand
read_when: before reading project code to learn what the model is for, who calls it, its input and output contract, or how production builds and runs it; then open only the file that answers the question
---
# Knowledge

What reading the code and the talks with the user established, kept here so later sessions read it instead of the code.
Other facts have their own home: the timed path, its shapes and time shares in MAP.md; leads and known issues in BACKLOG.md; agreed parameters in PROJECT.md, which points here for details; choices and their reasons in DECISIONS.md.

| File | Answers | Read at |
|---|---|---|
| business.md | Who calls the model and how, what each output means, the rules the code applies to results, traffic | <commits> |
| contract.md | Exact input and output formats of the entry points, and what downstream can observe | <commits> |
| deployment.md | How production gets the code, weights, and built artifacts, and how it builds, starts, and runs | <commits> |

## Writing here
- One conclusion per line, ending with its tag and source. No pasted code.
- Sources: `path:function` or a path in the project repo; <name>/<path> in a sibling repo, each named here with its location: <none>.
- Tags: [code] read in a repo file (code, config, or the authors' docs, named by the source); [run] seen in command output; [guess] inferred, not confirmed; [user YYYY-MM-DD] stated by the user.
- Write while reading. When a fact changes, edit its line instead of adding a second one.
- Open questions stay under "Open questions" in their topic file; STATUS.md only points to them. An answer replaces its question as a [user] line.
- After the project or a sibling repo moves to another commit, re-read what the diff touches, then update the Read at column and each file's Read at line.
- A new topic gets its own kebab-case file and a row in the table.
