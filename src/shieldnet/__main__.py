"""Allow ``python -m shieldnet ...`` as well as the installed ``shieldnet`` script.

Both spellings matter: the console script is nicer to type, but it only exists after
``pip install -e .``, and in Colab a plain ``!python -m shieldnet train`` from a cloned
checkout is the shortest path to a running model.
"""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
