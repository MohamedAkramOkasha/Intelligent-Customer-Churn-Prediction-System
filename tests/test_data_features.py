import numpy as np
import pandas as pd
import pytest

from churn import config
from churn.data import clean, encode_target, load_dataset, split, validate
from churn.features import FeatureEngineer, build_preprocessor, transformed_to_source_map


def test_clean_fixes_blank_total_charges():
    df = pd.DataFrame({"TotalCharges": [" ", "10.5"], "SeniorCitizen": [0, 1],
                       "tenure": [0, 1], "MonthlyCharges": [20.0, 10.5]})
    out = clean(df)
    assert out["TotalCharges"].tolist() == [0.0, 10.5]
    assert out["SeniorCitizen"].tolist() == ["No", "Yes"]


def test_full_dataset_loads_and_validates():
    X, y, ids = load_dataset()
    assert len(X) == len(y) == 7043
    assert list(X.columns) == config.RAW_FEATURES
    assert set(y.unique()) == {0, 1}
    assert 0.25 < y.mean() < 0.28
    assert ids.is_unique
    validate(X)


def test_split_is_stratified():
    X, y, ids = load_dataset()
    _, _, y_tr, y_te, _, _ = split(X, y, ids)
    assert abs(y_tr.mean() - y_te.mean()) < 0.01
    assert len(y_te) == pytest.approx(0.2 * len(y), abs=2)


def test_validate_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing required columns"):
        validate(pd.DataFrame({"tenure": [1]}))


def test_encode_target():
    assert encode_target(pd.Series(["Yes", "No"])).tolist() == [1, 0]


def test_feature_engineer(high_risk, low_risk):
    out = FeatureEngineer().transform(pd.DataFrame([high_risk, low_risk]))
    assert list(out.columns) == config.MODEL_CATEGORICAL + config.MODEL_NUMERIC
    hr, lr = out.iloc[0], out.iloc[1]
    assert hr["tenure_group"] == "0-12m" and lr["tenure_group"] == "49m+"
    assert hr["is_auto_payment"] == "No" and lr["is_auto_payment"] == "Yes"
    assert hr["has_family"] == "No" and lr["has_family"] == "Yes"
    assert lr["num_protection_services"] == 4
    assert hr["avg_monthly_spend"] == pytest.approx(99.5)
    assert not out[config.MODEL_NUMERIC].isna().any().any()


def test_feature_engineer_handles_zero_tenure(high_risk):
    high_risk.update(tenure=0, TotalCharges=0.0)
    out = FeatureEngineer().transform(pd.DataFrame([high_risk]))
    assert np.isfinite(out[config.MODEL_NUMERIC].to_numpy(dtype=float)).all()
    assert out.iloc[0]["avg_monthly_spend"] == pytest.approx(high_risk["MonthlyCharges"])


def test_source_map_covers_every_transformed_column(high_risk, low_risk):
    X = FeatureEngineer().transform(pd.DataFrame([high_risk, low_risk]))
    pre = build_preprocessor().fit(X)
    assert len(transformed_to_source_map(pre)) == pre.transform(X).shape[1]
