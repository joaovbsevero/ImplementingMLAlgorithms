import contextlib
import importlib
import pkgutil
from dataclasses import dataclass
from typing import Callable

from .algorithm import Algorithm


@dataclass
class AlgorithmMetadata:
    name: str
    description: str
    func: Callable[[], None]


def import_all_modules(package_name: str):
    ignored_modules = (
        "algorithms.__main__",
        "algorithms.__init__",
        "algorithms.algorithm",
        "algorithms.app",
        "algorithms.utils",
    )

    package = importlib.import_module(package_name)
    package_path = package.__path__

    for _, module_name, is_pkg in pkgutil.walk_packages(
        package_path, package_name + "."
    ):
        if module_name in ignored_modules:
            continue

        if module_name.endswith("__init__"):
            continue

        with contextlib.suppress(ImportError):
            importlib.import_module(module_name)

        if is_pkg:
            import_all_modules(module_name)


def get_algorithms() -> dict[str, AlgorithmMetadata]:
    algorithms = {}
    for clazz in Algorithm.__subclasses__():
        parts = clazz.__module__.split(".")
        name = f"{"-".join(parts[1:-1])}-{clazz.name()}"

        metadata = AlgorithmMetadata(
            name=name,
            description=clazz.description(),
            func=clazz.run,
        )
        algorithms[name] = metadata

    return algorithms
