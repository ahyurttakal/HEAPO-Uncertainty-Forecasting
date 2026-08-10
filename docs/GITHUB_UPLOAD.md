# Publishing this repository on GitHub

Create an empty GitHub repository without a generated README, license, or
`.gitignore`, then run from this project's root:

```powershell
git init
git add .
git status --short
git commit -m "Initial reproducible HEAPO forecasting release"
git branch -M main
git remote add origin https://github.com/OWNER/REPOSITORY.git
git push -u origin main
```

Before `git add`, replace `OWNER/REPOSITORY` only in commands or metadata you
choose to add; do not put credentials in files. Verify that `data/`,
`outputs/`, local configs, archives, Parquet files, and checkpoints are absent.

Recommended repository settings:

- keep the default branch named `main`;
- enable branch protection after CI passes;
- require the `test` workflow for pull requests;
- enable Issues only if they can be monitored;
- create releases from immutable version tags such as `v1.0.0`;
- archive the exact config and run manifest with the manuscript, not in a
  public repository if they contain local paths.
