import numpy as np
import pandas as pd

from atomiq.preprocessing import make_preprocessor
from atomiq.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def _tiny_frame():
    return pd.DataFrame(
        {
            **{name: [1, 2, 3, 4] for name in NUMERIC_FEATURES},
            **{name: ["a", "a", "b", "b"] for name in CATEGORICAL_FEATURES},
        }
    )


def test_numeric_imputer_is_fit_only_on_the_supplied_training_fold():
    frame = _tiny_frame()
    transformer = make_preprocessor(scale_numeric=False)
    transformer.fit(frame.iloc[:2])
    numeric_pipeline = transformer.named_transformers_["numeric"]
    np.testing.assert_allclose(
        numeric_pipeline.named_steps["imputer"].statistics_,
        np.full(len(NUMERIC_FEATURES), 1.5),
    )


def test_unknown_validation_categories_are_ignored_and_output_is_finite():
    train = _tiny_frame().iloc[:3].copy()
    validation = _tiny_frame().iloc[3:].copy()
    validation.loc[:, CATEGORICAL_FEATURES] = "unseen-category"
    transformer = make_preprocessor(scale_numeric=True)
    transformed = transformer.fit(train).transform(validation)
    assert transformed.shape[0] == 1
    assert np.isfinite(transformed).all()
