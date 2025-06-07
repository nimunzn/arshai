=================
PyPI Deployment
=================

Complete guide for deploying Arshai package versions to PyPI.

.. note::
   This guide is for **maintainers only**. Contributors should submit pull requests with their changes, and maintainers will handle releases.

Overview
========

Arshai uses a fully automated CI/CD pipeline for PyPI deployment:

- **Trigger**: Creating a GitHub release
- **Process**: Automated via GitHub Actions
- **Result**: Package published to PyPI automatically

Quick Reference
===============

.. code-block:: bash

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

Prerequisites
=============

Repository Access
-----------------

- Write access to https://github.com/nimunzn/arshai
- Ability to create releases and manage repository secrets

PyPI Configuration
------------------

1. **PyPI Account** with permissions for the ``arshai`` package
2. **API Token** configured as repository secret:

   - Generate at: https://pypi.org/manage/account/token/
   - Token name: ``arshai-github-actions``
   - Scope: ``Project: arshai`` (recommended)
   - Add to GitHub: Repository Settings → Secrets → Actions → ``PYPI_TOKEN``

Tools Required
--------------

- Git with push access to the repository
- GitHub CLI (``gh``) for release management
- Python and Poetry for local testing

Deployment Process
==================

Step 1: Prepare Release
-----------------------

Ensure Clean State
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git checkout main
   git pull origin main
   git status  # Should show no uncommitted changes

Run Pre-release Checks
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Install dependencies
   poetry install

   # Run tests
   poetry run pytest

   # Check code quality
   poetry run black --check .
   poetry run isort --check-only .

   # Test build
   poetry build

Step 2: Update Version Numbers
------------------------------

Choose Version Number
~~~~~~~~~~~~~~~~~~~~~

Follow `Semantic Versioning <https://semver.org/>`_:

- **Major (X.0.0)**: Breaking API changes
- **Minor (X.Y.0)**: New features, backward compatible  
- **Patch (X.Y.Z)**: Bug fixes, backward compatible

Update pyproject.toml
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: toml

   [tool.poetry]
   name = "arshai"
   version = "X.Y.Z"  # Update this line

Update arshai/_version.py
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   __version__ = "X.Y.Z"
   __version_info__ = (X, Y, Z)

Verify Updates
~~~~~~~~~~~~~~

.. code-block:: bash

   # Check Poetry version
   poetry version
   # Should output: arshai X.Y.Z

   # Check Python module version
   python -c "import arshai; print(arshai.__version__)"
   # Should output: X.Y.Z

Step 3: Commit and Push
-----------------------

.. code-block:: bash

   # Stage version files
   git add pyproject.toml arshai/_version.py

   # Commit with standard message
   git commit -m "chore: bump version to X.Y.Z"

   # Push to main
   git push origin main

Step 4: Create Git Tag
----------------------

.. code-block:: bash

   # Create annotated tag
   git tag vX.Y.Z -m "Release vX.Y.Z"

   # Push tag
   git push origin vX.Y.Z

Step 5: Create GitHub Release
-----------------------------

Using GitHub CLI
~~~~~~~~~~~~~~~~

.. code-block:: bash

   gh release create vX.Y.Z \
     --title "Release vX.Y.Z" \
     --notes "## What's New in vX.Y.Z

   🚀 **New Features:**
   - Feature 1 description
   - Feature 2 description

   🔧 **Improvements:**
   - Improvement 1
   - Improvement 2

   🐛 **Bug Fixes:**
   - Bug fix 1
   - Bug fix 2

   ## Installation

   \`\`\`bash
   pip install arshai==X.Y.Z
   \`\`\`

   ## Full Changelog
   https://github.com/nimunzn/arshai/compare/vPREV...vX.Y.Z"

Using GitHub Web Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Go to: https://github.com/nimunzn/arshai/releases
2. Click "Create a new release"
3. Choose tag: ``vX.Y.Z``
4. Release title: ``Release vX.Y.Z``
5. Add detailed release notes
6. Click "Publish release"

Step 6: Monitor Deployment
--------------------------

Check GitHub Actions
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # List recent workflow runs
   gh run list --limit 5

   # View specific run
   gh run view [RUN_ID]

   # Check logs if there are issues
   gh run view [RUN_ID] --log-failed

Verify PyPI Publication
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Check PyPI page (may take a few minutes)
   open https://pypi.org/project/arshai/

   # Test installation
   pip install arshai==X.Y.Z
   python -c "import arshai; print(f'Version: {arshai.__version__}')"

GitHub Actions Workflow
========================

The deployment uses ``.github/workflows/publish.yml``:

.. code-block:: yaml

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

Workflow Details
----------------

**Trigger**: Automatically runs when a GitHub release is published

**Steps**:
1. Checkout code at the tagged commit
2. Set up Python 3.11 environment
3. Install Poetry package manager
4. Install project dependencies
5. Build package (wheel and source distribution)
6. Publish to PyPI using configured token

Troubleshooting
===============

Common Issues
-------------

Version Already Exists
~~~~~~~~~~~~~~~~~~~~~~~

**Error**: ``HTTPError: 400 Bad Request from https://upload.pypi.org/legacy/``

**Cause**: Cannot upload the same version twice to PyPI

**Solution**: 
- Increment version number
- Create new release with updated version

Invalid PyPI Token
~~~~~~~~~~~~~~~~~~~

**Error**: ``HTTPError: 403 Forbidden``

**Cause**: PyPI token is invalid or expired

**Solution**:
1. Generate new token at https://pypi.org/manage/account/token/
2. Update ``PYPI_TOKEN`` secret in GitHub repository settings

Build Failures
~~~~~~~~~~~~~~~

**Error**: Package build fails during workflow

**Solutions**:
- Test build locally: ``poetry build``
- Check for syntax errors
- Verify all dependencies are in ``pyproject.toml``
- Ensure all required files are committed

Import Errors
~~~~~~~~~~~~~

**Error**: Module import failures during build

**Solutions**:
- Verify all dependencies in ``pyproject.toml``
- Test in clean virtual environment
- Check for circular imports

Recovery Procedures
===================

Failed Deployment
-----------------

If GitHub Actions fails after release creation:

1. **Identify and fix** the issue
2. **Increment version** to next patch (e.g., 0.3.0 → 0.3.1)
3. **Follow normal release process**
4. **Optionally delete** failed release from GitHub

Hotfix Deployment
-----------------

For critical bug fixes:

1. **Create hotfix branch** from main
2. **Apply minimal fix**
3. **Update to patch version**
4. **Expedite review process**
5. **Deploy using normal process**

Security Considerations
=======================

Token Management
----------------

- **Scope**: Use project-scoped tokens when possible
- **Rotation**: Rotate tokens every 6-12 months
- **Access**: Limit to essential maintainers only
- **Storage**: Only in GitHub Secrets, never in code

Release Security
----------------

- **Code Review**: All changes reviewed before release
- **Testing**: Comprehensive testing before version bump
- **Verification**: Post-release testing and monitoring

Best Practices
==============

Version Management
------------------

- Follow semantic versioning strictly
- Document breaking changes clearly
- Provide migration guides for major versions

Release Notes
-------------

- Include all user-facing changes
- Categorize changes (features, fixes, improvements)
- Provide installation and upgrade instructions
- Link to full changelog on GitHub

Testing
-------

- Test locally before releasing
- Verify installation in clean environment
- Check documentation builds correctly
- Validate all examples still work

For additional help, see the `main deployment documentation <https://github.com/nimunzn/arshai/blob/main/docs/deployment/pypi-deployment.md>`_ or open an issue in the repository.