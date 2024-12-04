# from typing import Callable

# from . import ensemble, reinforcement, supervised, unsupervised


# def algorithms() -> dict[str, Callable[[], None]]:
#     algs = {}
#     for name, func in ensemble.algorithms().items():
#         algs[f"ensemble-{name}"] = func

#     for name, func in reinforcement.algorithms().items():
#         algs[f"reinforcement-{name}"] = func

#     for name, func in supervised.algorithms().items():
#         algs[f"supervised-{name}"] = func

#     for name, func in unsupervised.algorithms().items():
#         algs[f"unsupervised-{name}"] = func

#     return algs
