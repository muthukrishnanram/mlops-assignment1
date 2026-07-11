"""Feature engineering: the ColumnTransformer shared by training and the exported model.

Kept as a small, readable function (rather than only existing inside a pickled
artifact) so the preprocessing logic stays transparent and rebuildable from
source if the serialized pipeline format ever becomes unreadable across
library versions.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import CATEGORICAL_FEATURES, CONTINUOUS_FEATURES  # noqa: E402


def build_preprocessor() -> ColumnTransformer:
    """Continuous features are standardized; categorical features are one-hot encoded
    (binary features collapse to a single 0/1 column via drop='if_binary', multi-valued
    ones like `cp`/`thal` get a column per category so the model doesn't assume a false
    ordinal relationship between e.g. chest-pain types)."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), CONTINUOUS_FEATURES),
            (
                "cat",
                OneHotEncoder(drop="if_binary", handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
