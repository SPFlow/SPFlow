__version__ = "1.0.0"
__author__ = "TODO"


# We now want to import all dispatched function from all modules defined in spflow.modules
# This is done by recursively walking through all sub-packages and modules starting from spflow.modules
# and importing the specified functions from each module.
#
# With this users can simply import via `from spflow import em, log_likelihood, sample` and get all the dispatched methods.

import importlib
import pkgutil

import spflow.modules as spflow_modules

# List of functions to import from modules that we dispatch in the modules
dispatched_functions_to_be_imported = [
    "log_likelihood",
    "em",
    "sample",
    "maximum_likelihood_estimation",
    "marginalize",
]


def __import_from_module(module):
    """Import specified functions from a module.

    This function attempts to import the specified functions listed in
    `functions_to_import` from the given module. If a function is not found,
    it's simply ignored.

    Args:
        module (module): The module from which to import functions.

    """
    for func_name in dispatched_functions_to_be_imported:
        # Check if the module has the function
        if hasattr(module, func_name):
            globals()[func_name] = getattr(module, func_name)


def __walk_packages(path, prefix):
    """Recursively walk through packages and import specified functions.

    This function recursively walks through all sub-packages and modules
    starting from the given path, importing the specified functions
    from each module. If a module is a package, it calls itself
    recursively to walk through its sub-packages and modules.

    Args:
        path (iterable): The path to start walking packages from.
        prefix (str): The prefix to use for module names.

    """
    for module_info in pkgutil.walk_packages(path, prefix):
        module_name = module_info.name
        module = importlib.import_module(module_name)
        __import_from_module(module)
        if module_info.ispkg:
            __walk_packages(module.__path__, module.__name__ + ".")


# Start the recursive walking from the base module
__walk_packages(spflow_modules.__path__, spflow_modules.__name__ + ".")
