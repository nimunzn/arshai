==========
Deployment
==========

This section covers deployment processes and procedures for the Arshai package.

.. toctree::
   :maxdepth: 2
   :caption: Deployment Guides

   pypi-deployment
   production
   scaling
   monitoring

PyPI Deployment
===============

The primary deployment target for Arshai is the Python Package Index (PyPI). This allows users to install the package via pip.

**For Contributors:**
- See :doc:`pypi-deployment` for complete deployment instructions
- Only maintainers can deploy to PyPI
- Process is fully automated via GitHub Actions

**For Users:**
- Install the latest version: ``pip install arshai``
- Install specific version: ``pip install arshai==X.Y.Z``
- Upgrade existing installation: ``pip install --upgrade arshai``

Quick Start for Maintainers
============================

.. code-block:: bash

   # 1. Update versions in pyproject.toml and arshai/_version.py
   # 2. Commit and push changes
   git add pyproject.toml arshai/_version.py
   git commit -m "chore: bump version to X.Y.Z"
   git push origin main
   
   # 3. Create tag and release
   git tag vX.Y.Z
   git push origin vX.Y.Z
   gh release create vX.Y.Z --title "Release vX.Y.Z" --notes "Release notes"

Deployment Status
=================

Current deployment configuration:

- **PyPI Package**: https://pypi.org/project/arshai/
- **GitHub Repository**: https://github.com/nimunzn/arshai
- **Documentation**: https://nimunzn.github.io/arshai/
- **CI/CD**: GitHub Actions
- **Release Frequency**: As needed (semantic versioning)

Prerequisites
=============

To deploy new versions, maintainers need:

1. **Repository Access**: Write permissions to GitHub repository
2. **PyPI Permissions**: Access to manage the arshai package on PyPI
3. **GitHub Secrets**: ``PYPI_TOKEN`` configured in repository settings
4. **Tools**: GitHub CLI (``gh``) for release management

For detailed instructions, see the :doc:`pypi-deployment` guide.