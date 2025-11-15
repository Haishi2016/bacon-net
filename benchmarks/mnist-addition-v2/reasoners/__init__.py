# mnist-addition-v2/reasoners/__init__.py

from .ltn_reasoner import LTNAdditionReasoner
from .mlp_reasonser import MLPAdditionReasoner
from .deepproblog_reasoner import DeepProbLogAdditionReasoner

REASONER_REGISTRY = {
    "ltn": LTNAdditionReasoner,
    "mlp": MLPAdditionReasoner,
    "deepproblog": DeepProbLogAdditionReasoner,
    # later: "dcr": DCRAdditionReasoner, "bacon": BACONAdditionReasoner, ...
}
