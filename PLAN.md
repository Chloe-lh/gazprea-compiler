Commit **Frequently**. More small commits > huge unreadable commits.


## Dividing features -> branches

1. Send a message in Discord to ensure no one else is working on said feature.
2. Pull all new changes.
3. Create a new issue and branch for your feature.

## Merging to main

1. Pull from main, deal with any merge conflicts locally.
2. Ensure CI job run passes, i.e. code builds.
3. Submit a pull request.

## How to name commits
Follow the [Conventional Commits](https://gist.github.com/qoomon/5dfcdf8eec66a051ecd85625518cfd13).

Quick example:

```
git commit -m"<type>(<optional scope>): <description>" \
  -m"<optional body>" \
  -m"<optional footer>"
```

Types: `feat`, `fix`, `refactor`, `style` (e.g. whitespace), `test` (test cases), `docs`