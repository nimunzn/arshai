# PyPI Deployment Quick Start

**For Arshai Package Maintainers Only**

## Prerequisites Checklist

- [ ] Repository write access
- [ ] PyPI token configured as `PYPI_TOKEN` secret
- [ ] GitHub CLI (`gh`) installed
- [ ] Local git setup with push permissions

## Release Process (5 Steps)

### 1. Update Version Numbers

```bash
# Edit pyproject.toml
version = "X.Y.Z"

# Edit arshai/_version.py  
__version__ = "X.Y.Z"
__version_info__ = (X, Y, Z)
```

### 2. Commit and Push

```bash
git add pyproject.toml arshai/_version.py
git commit -m "chore: bump version to X.Y.Z"
git push origin main
```

### 3. Create Git Tag

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

### 4. Create GitHub Release

```bash
gh release create vX.Y.Z --title "Release vX.Y.Z" --notes "Release notes here"
```

### 5. Monitor Deployment

```bash
# Check workflow status
gh run list --limit 5

# Verify on PyPI (after a few minutes)
pip install arshai==X.Y.Z
```

## Version Numbering

- **Patch (X.Y.Z)**: Bug fixes → increment Z
- **Minor (X.Y.0)**: New features → increment Y  
- **Major (X.0.0)**: Breaking changes → increment X

## Troubleshooting

| Error | Solution |
|-------|----------|
| Version already exists | Increment version and retry |
| 403 Forbidden | Regenerate PyPI token |
| Build failed | Test `poetry build` locally |
| Import errors | Check dependencies in pyproject.toml |

## Emergency Contact

For deployment issues:
1. Check [detailed deployment guide](./deployment/pypi-deployment.md)
2. Open GitHub issue
3. Contact repository maintainers

---

**⚠️ Important**: Only maintainers should deploy. Contributors should submit PRs.