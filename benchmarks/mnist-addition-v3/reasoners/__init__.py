# mnist-addition-v3/reasoners/__init__.py

from .deepproblog_reasoner import DeepProbLogAdditionReasoner
from .deepstochlog_reasoner import DeepStochLogAdditionReasoner
from .ltn_reasoner import LTNAdditionReasoner
from .bacon_reasoner import BaconAdditionReasoner

REASONER_REGISTRY = {
    "deepproblog":  DeepProbLogAdditionReasoner,
    "deepstochlog": DeepStochLogAdditionReasoner,
    "ltn":          LTNAdditionReasoner,
    "bacon":        BaconAdditionReasoner,
}
