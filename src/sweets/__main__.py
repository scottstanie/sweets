"""Main module to provide command line interface to the workflows."""

from .cli import main

# https://docs.python.org/3/library/__main__.html#packaging-considerations
# allows `python -m sweets` to work
main()
