# PyPI Deployment Guide

This guide provides detailed instructions for maintainers on how to deploy new versions of the Arshai package to PyPI.

## Overview

Arshai uses a fully automated CI/CD pipeline for PyPI deployment. The process is triggered by creating a GitHub release, which automatically builds and publishes the package to PyPI.

## Quick Reference

```bash
# 1. Update versions
# Edit pyproject.toml: version = "X.Y.Z"
# Edit arshai/_version.py: __version__ = "X.Y.Z"

# 2. Commit and push
git add pyproject.toml arshai/_version.py
git commit -m "chore: bump version to X.Y.Z"
git push origin main

# 3. Create tag and release
git tag vX.Y.Z
git push origin vX.Y.Z
gh release create vX.Y.Z --title "Release vX.Y.Z" --notes "Release notes"

# 4. Monitor deployment
gh run list --limit 5
```

## Prerequisites

### For Repository Maintainers

1. **Repository Access**: Write access to the `nimunzn/arshai` repository
2. **PyPI Account**: Account with permissions to manage the `arshai` package
3. **GitHub CLI**: Install `gh` CLI tool for release management
4. **Git Access**: Local git setup with push permissions

### PyPI Token Configuration

The deployment requires a PyPI API token configured as a repository secret:

1. **Generate PyPI Token**:
   - Go to https://pypi.org/manage/account/token/
   - Click "Add API token"
   - Token name: `arshai-github-actions`
   - Scope: "Project: arshai" (recommended) or "Entire account"
   - Copy the token (starts with `pypi-`)

2. **Add to GitHub Secrets**:
   - Go to: https://github.com/nimunzn/arshai/settings/secrets/actions
   - Click "New repository secret"
   - Name: `PYPI_TOKEN`
   - Value: [Your PyPI token]

## Deployment Process

### Step 1: Prepare Release

#### 1.1 Ensure Clean State

```bash
# Ensure you're on main branch and up to date
git checkout main
git pull origin main

# Check for uncommitted changes
git status
```

#### 1.2 Run Pre-release Checks

```bash
# Install dependencies
poetry install

# Run tests
poetry run pytest

# Check code quality
poetry run black --check .
poetry run isort --check-only .
poetry run mypy arshai/

# Build package locally to verify
poetry build
```

#### 1.3 Determine Version Number

Follow [Semantic Versioning](https://semver.org/):

- **Major (X.0.0)**: Breaking API changes
- **Minor (X.Y.0)**: New features, backward compatible
- **Patch (X.Y.Z)**: Bug fixes, backward compatible

Examples:
- `0.2.2` → `0.2.3` (bug fix)
- `0.2.2` → `0.3.0` (new feature)
- `0.2.2` → `1.0.0` (breaking change)

### Step 2: Update Version Numbers

#### 2.1 Update pyproject.toml

```bash
# Edit pyproject.toml
nano pyproject.toml

# Change this line:
version = "0.2.2"  # old version
# To:
version = "0.3.0"  # new version
```

#### 2.2 Update arshai/_version.py

```bash
# Edit arshai/_version.py
nano arshai/_version.py

# Change these lines:
__version__ = "0.2.2"          # old version
__version_info__ = (0, 2, 2)   # old version
# To:
__version__ = "0.3.0"          # new version
__version_info__ = (0, 3, 0)   # new version
```

#### 2.3 Verify Version Update

```bash
# Check that Poetry picks up the new version
poetry version
# Should output: arshai 0.3.0

# Check that Python module reports correct version
python -c "import arshai; print(arshai.__version__)"
# Should output: 0.3.0
```

### Step 3: Commit and Push Changes

```bash
# Stage version files
git add pyproject.toml arshai/_version.py

# Commit with standard message format
git commit -m "chore: bump version to 0.3.0"

# Push to main branch
git push origin main
```

### Step 4: Create Git Tag

```bash
# Create annotated tag
git tag v0.3.0 -m "Release v0.3.0"

# Push tag to GitHub
git push origin v0.3.0
```

### Step 5: Create GitHub Release

#### 5.1 Generate Release Notes

Prepare release notes covering:
- **New Features**: Major additions
- **Improvements**: Enhancements to existing features
- **Bug Fixes**: Fixed issues
- **Dependencies**: Updated dependencies
- **Breaking Changes**: Any backward incompatible changes

#### 5.2 Create Release via GitHub CLI

```bash
gh release create v0.3.0 \
  --title "Release v0.3.0" \
  --notes "## What's New in v0.3.0

🚀 **New Features:**
- Added new agent type for specialized tasks
- Enhanced memory management capabilities

🔧 **Improvements:**
- Better error handling in LLM clients
- Performance optimizations

🐛 **Bug Fixes:**
- Fixed memory leak in conversation agent
- Resolved timeout issues with external APIs

📦 **Dependencies:**
- Updated OpenAI client to latest version
- Added support for new embedding models

## Installation

\`\`\`bash
pip install arshai==0.3.0
\`\`\`

## Full Changelog
https://github.com/nimunzn/arshai/compare/v0.2.2...v0.3.0"
```

#### 5.3 Alternative: Create Release via GitHub UI

1. Go to: https://github.com/nimunzn/arshai/releases
2. Click "Create a new release"
3. Choose tag: `v0.3.0`
4. Release title: `Release v0.3.0`
5. Add release notes
6. Click "Publish release"

### Step 6: Monitor Deployment

#### 6.1 Check GitHub Actions

```bash
# Monitor workflow status
gh run list --limit 5

# View specific run details
gh run view [RUN_ID]

# View logs if there are issues
gh run view [RUN_ID] --log-failed
```

#### 6.2 Verify PyPI Publication

```bash
# Check PyPI page (may take a few minutes)
open https://pypi.org/project/arshai/

# Test installation
pip install arshai==0.3.0
python -c "import arshai; print(f'Installed version: {arshai.__version__}')"
```

## GitHub Actions Workflow

### Workflow File

The deployment is handled by `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  build-and-publish:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
    
    - name: Install Poetry
      uses: snok/install-poetry@v1
      with:
        version: latest
        virtualenvs-create: true
        virtualenvs-in-project: true
    
    - name: Install dependencies
      run: poetry install --no-interaction --no-root
    
    - name: Build package
      run: poetry build
    
    - name: Publish to PyPI
      env:
        POETRY_PYPI_TOKEN_PYPI: ${{ secrets.PYPI_TOKEN }}
      run: poetry publish
```

### Workflow Triggers

- **Automatic**: GitHub release creation
- **Manual**: Repository maintainers can manually trigger via GitHub Actions UI

### Workflow Steps

1. **Checkout**: Gets the code at the tagged commit
2. **Setup Python**: Installs Python 3.11
3. **Install Poetry**: Sets up Poetry for package management
4. **Install Dependencies**: Installs project dependencies
5. **Build Package**: Creates wheel and source distribution
6. **Publish**: Uploads to PyPI using the configured token

## Troubleshooting

### Common Issues

#### 1. Version Already Exists on PyPI

**Error**: `HTTPError: 400 Bad Request from https://upload.pypi.org/legacy/`

**Solution**: 
- Cannot upload the same version twice to PyPI
- Increment version number and create new release

#### 2. PyPI Token Invalid

**Error**: `HTTPError: 403 Forbidden`

**Solution**: 
- Regenerate PyPI token
- Update `PYPI_TOKEN` secret in GitHub repository settings

#### 3. Build Failures

**Error**: Package build fails during workflow

**Solution**: 
- Test build locally: `poetry build`
- Check for syntax errors or missing dependencies
- Ensure all required files are included in version control

#### 4. Missing Dependencies

**Error**: Import errors during package build

**Solution**: 
- Verify all dependencies are listed in `pyproject.toml`
- Test installation in clean environment locally

### Recovery Procedures

#### Failed Deployment After Release

If the GitHub Actions workflow fails after a release is created:

1. **Fix the issue** in the codebase
2. **Update version** to next patch version (e.g., 0.3.0 → 0.3.1)
3. **Follow normal release process**
4. **Delete failed release** from GitHub (optional)

#### Hotfix Deployment

For critical bug fixes:

1. **Create hotfix branch** from main
2. **Apply minimal fix**
3. **Update to patch version** (e.g., 0.3.0 → 0.3.1)
4. **Fast-track through review process**
5. **Follow normal deployment process**

## Security Considerations

### PyPI Token Management

- **Scope**: Use project-scoped tokens when possible
- **Rotation**: Rotate tokens periodically (every 6-12 months)
- **Access**: Limit token access to essential maintainers
- **Storage**: Store only in GitHub Secrets, never in code

### Release Verification

- **Code Review**: All changes should be reviewed before release
- **Testing**: Comprehensive testing before version bump
- **Signing**: Consider signing releases for additional security

## Automation Improvements

### Future Enhancements

1. **Automated Version Bumping**: Use conventional commits to auto-determine version
2. **Changelog Generation**: Auto-generate changelogs from commit messages
3. **Test PyPI**: Add Test PyPI deployment for pre-release testing
4. **Release Validation**: Automated post-release testing

### Monitoring

- **PyPI Download Stats**: Monitor package adoption
- **Issue Tracking**: Monitor for post-release issues
- **Performance**: Track deployment success rates

## Contact and Support

For deployment issues or questions:

1. **GitHub Issues**: Open an issue in the repository
2. **Maintainer Contact**: Reach out to repository maintainers
3. **Documentation**: Check this guide and CONTRIBUTING.md

## Changelog

- **2024-12-07**: Initial deployment guide created
- **2024-12-07**: Added troubleshooting section
- **2024-12-07**: Added security considerations