"""Setup script for backward compatibility with pip."""

from setuptools import setup

if __name__ == "__main__":
    try:
        setup(name="arshai")
    except:  # noqa
        print(
            "\n\nAn error occurred during setup! Please use Poetry for installation instead:\n"
            "    pip install poetry\n"
            "    poetry install\n"
        ) 