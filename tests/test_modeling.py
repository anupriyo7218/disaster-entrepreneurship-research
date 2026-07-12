"""
Tests for src/modeling.py.

Covers panel preparation (validation, balance checks, dedup), the TWFE
model fit itself using a small synthetic panel with a known engineered
relationship, diagnostics extraction, and the interpretation generator's
logic branches (significant vs. not significant).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config_loader import load_config
from src.modeling import (
    ModelingError,
    extract_diagnostics,
    fit_twfe_model,
    generate_interpretation,
    prepare_panel_data,
)


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture
def synthetic_flat_df():
    """
    A small synthetic flat dataset (pre-panel-indexing) with a known,
    engineered negative relationship between disaster_count and
    business_applications, plus state-level and month-level offsets, so we
    can check the fitted coefficient recovers something close to the true
    effect once fixed effects are controlled for.
    """
    rng = np.random.default_rng(42)
    states = ["CA", "TX", "NY", "FL"]
    months = pd.period_range("2020-01", "2020-12", freq="M")

    state_effect = {"CA": 5000, "TX": 4000, "NY": 3000, "FL": 3500}
    month_effect = {m: 100 * i for i, m in enumerate(months)}
    true_beta = -50.0

    rows = []
    for state in states:
        for month in months:
            disaster_count = rng.integers(0, 4)
            noise = rng.normal(0, 50)
            y = state_effect[state] + month_effect[month] + true_beta * disaster_count + noise
            rows.append({
                "state": state,
                "year_month": str(month),
                "business_applications": y,
                "disaster_count": disaster_count,
                "disclosure_flag": "valid",
            })

    df = pd.DataFrame(rows)
    # Introduce a couple of missing dependent-variable rows to exercise the drop logic.
    df.loc[df.sample(3, random_state=1).index, "business_applications"] = np.nan
    return df


# --- Panel preparation -----------------------------------------------------

def test_prepare_panel_data_drops_missing_dependent(synthetic_flat_df, config):
    panel = prepare_panel_data(synthetic_flat_df, config)
    assert panel["business_applications"].isnull().sum() == 0
    assert len(panel) == len(synthetic_flat_df) - 3


def test_prepare_panel_data_sets_multiindex(synthetic_flat_df, config):
    panel = prepare_panel_data(synthetic_flat_df, config)
    assert panel.index.names == ["state", "year_month"]
    assert isinstance(panel.index.get_level_values(1)[0], pd.Timestamp)


def test_prepare_panel_data_rejects_duplicate_keys(config):
    bad_df = pd.DataFrame({
        "state": ["CA", "CA"],
        "year_month": ["2020-01", "2020-01"],
        "business_applications": [100.0, 200.0],
        "disaster_count": [0, 1],
    })
    with pytest.raises(ModelingError):
        prepare_panel_data(bad_df, config)


def test_prepare_panel_data_rejects_missing_columns(config):
    bad_df = pd.DataFrame({
        "wrong_col": ["CA"],
        "year_month": ["2020-01"],
        "business_applications": [100.0],
    })
    with pytest.raises(ModelingError):
        prepare_panel_data(bad_df, config)


# --- Model fitting ----------------------------------------------------------

def test_fit_twfe_model_recovers_approximate_true_coefficient(synthetic_flat_df, config):
    """
    With a known engineered beta of -50 and state/month offsets fully
    absorbed by fixed effects, the fitted coefficient should land in the
    right ballpark (not exact, due to noise and a small synthetic sample).
    """
    panel = prepare_panel_data(synthetic_flat_df, config)
    results = fit_twfe_model(panel, config)
    fitted_beta = results.params["disaster_count"]
    # True beta is -50; allow a generous tolerance given the small synthetic panel.
    assert -80 < fitted_beta < -20


def test_fit_twfe_model_uses_clustered_se(synthetic_flat_df, config):
    panel = prepare_panel_data(synthetic_flat_df, config)
    results = fit_twfe_model(panel, config)
    assert results._cov_type.lower().startswith("clustered")


# --- Diagnostics extraction -------------------------------------------------

def test_extract_diagnostics_contains_all_required_fields(synthetic_flat_df, config):
    panel = prepare_panel_data(synthetic_flat_df, config)
    results = fit_twfe_model(panel, config)
    diagnostics = extract_diagnostics(results, config)

    required_fields = {
        "n_observations", "n_entities", "n_time_periods",
        "rsquared_within", "rsquared_between", "rsquared_overall",
        "f_statistic", "f_statistic_pvalue",
        "coefficient", "std_error", "t_statistic", "p_value",
        "ci_lower_95", "ci_upper_95",
    }
    assert required_fields.issubset(diagnostics.keys())
    assert diagnostics["n_entities"] == 4
    assert diagnostics["n_time_periods"] == 12
    assert diagnostics["ci_lower_95"] < diagnostics["coefficient"] < diagnostics["ci_upper_95"]


# --- Interpretation generator -----------------------------------------------

def test_generate_interpretation_flags_significant_result(config, tmp_path, monkeypatch):
    monkeypatch.setattr("src.modeling.PROJECT_ROOT", tmp_path)
    diagnostics = {
        "coefficient": -100.0, "p_value": 0.001,
        "ci_lower_95": -150.0, "ci_upper_95": -50.0,
    }
    text = generate_interpretation(diagnostics, config)
    assert "statistically significant" in text
    assert "not statistically significant" not in text
    assert "negative" in text
    assert "not establish that disasters cause" in text or "NOT establish" in text


def test_generate_interpretation_flags_nonsignificant_result(config, tmp_path, monkeypatch):
    monkeypatch.setattr("src.modeling.PROJECT_ROOT", tmp_path)
    diagnostics = {
        "coefficient": -110.97, "p_value": 0.1136,
        "ci_lower_95": -248.47, "ci_upper_95": 26.52,
    }
    text = generate_interpretation(diagnostics, config)
    assert "not statistically significant" in text
    assert "cannot rule out the possibility of no association" in text


def test_generate_interpretation_never_claims_causality(config, tmp_path, monkeypatch):
    """
    Guards the project's core requirement: no causal language anywhere in
    the auto-generated interpretation, regardless of the result.
    """
    monkeypatch.setattr("src.modeling.PROJECT_ROOT", tmp_path)
    for coef, p in [(-100.0, 0.001), (50.0, 0.3), (0.0, 0.99)]:
        diagnostics = {"coefficient": coef, "p_value": p, "ci_lower_95": coef - 50, "ci_upper_95": coef + 50}
        text = generate_interpretation(diagnostics, config).lower()
        assert "causes" not in text
        assert "proves" not in text
        assert "causal evidence" not in text or "not present this coefficient as causal evidence" in text


# --- Integration check against real Module 2 output ------------------------

def test_modeling_runs_against_real_processed_data(config):
    """
    Confirms the real final_analysis_dataset.csv has what this module needs,
    end to end, without mocking anything.
    """
    from src.modeling import load_analysis_dataset
    raw_final = load_analysis_dataset(config)
    panel = prepare_panel_data(raw_final, config)
    results = fit_twfe_model(panel, config)
    diagnostics = extract_diagnostics(results, config)

    assert diagnostics["n_entities"] == 51
    assert diagnostics["n_observations"] > 0
    assert -1 <= diagnostics["rsquared_within"] <= 1
