---
tier: on-demand
read_when: deciding the target hardware or build settings, planning how a change ships, or anything about CI, the image, or service start-up
---
# Deployment

- Read at: <project commit, plus the name and commit of each sibling repo read> (<YYYY-MM-DD>)

## Code and weights
<How production gets the code and weights: pinned commits, submodules, Git LFS, downloads.>

## Build
<What CI builds (extensions, exported models, engines), on which machine and GPU, with which settings.>

## Runtime
<The image or environment and its versions, the start command, and what runs at start-up before the first request.>

## Differences from this machine
<GPU, driver, and versions, and anything built here that production builds differently.>

## Open questions
- <Where production's built artifacts come from, the GPUs and workers per GPU, how a release reaches production.>
