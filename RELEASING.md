# SReWAM public-release checklist

This repository contains the code for **SReWAM: Sparse Representation-Driven
Memory for One-Step World-Action Models**. The upstream JEPA-WAM paper and
project page are not SReWAM publications.

Before publishing a public repository or archive:

1. **Rotate/revoke the old W&B API key.** Earlier Git revisions contained a
   literal key in Slurm launchers. Removing it from current files does not
   remove it from Git history. Publish from a clean history/new repository
   after rotating the key; do not push the old history to a public remote.
2. Keep `.wandb_api_key`, model weights, RLDS data, simulator assets, logs,
   rollout outputs, and latency results outside Git. Check `git status` and
   the resulting archive before upload.
3. Provide a verified SReWAM checkpoint URL and checksum (including its
   `config.json` and `dataset_statistics.json`) before claiming that the
   checkpoint is publicly reproducible. The current README intentionally
   leaves the SReWAM weight URL unspecified.
4. Verify a fresh environment can resolve the Qvv2.5 model/tokenizer, V-JEPA
   weights, JEPA-WAM base VLM/policy, and LIBERO-Plus assets. The Qvv alias is
   not a download URL.
5. Run the regression tests in the README and a real H100 model-load/simulator
   smoke test. Do not publish latency or success-rate numbers from a pending,
   failed, or partial evaluation job.
6. The upstream JEPA-WAM static website is excluded from the source release.
   Create and review a separate SReWAM site before enabling GitHub Pages.
7. Review third-party model, dataset, and simulator licenses separately from
   this repository's MIT code license.

Slurm submission files and Newton-specific job helpers are local-only and
excluded from the source release. Allocate resources with your site's
scheduler, then invoke the portable launchers in `vla-scripts/` inside that
allocation. The historical Slurm files are not included in a fresh clone.
