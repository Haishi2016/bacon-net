# mnist-addition-v4/reasoners/__init__.py

from .bacon_reasoner import BaconAdditionReasoner

REASONER_REGISTRY = {
    "bacon": BaconAdditionReasoner,
    # later: "dcr": DCRAdditionReasoner, "deepproblog": ..., etc.
}
