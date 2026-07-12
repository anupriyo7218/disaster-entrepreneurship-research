"""
modeling.py

Module 4 deliverable — Two-Way Fixed Effects (TWFE) panel regression.

Estimates whether disaster occurrence is associated with monthly business
application counts, controlling for state fixed effects (absorbs any
time-invariant differences between states — size, industry mix, baseline
entrepreneurial activity) and month-year fixed effects (absorbs any national
shock common to all states in a given month — recessions, seasonality,
policy changes). What's left over is the within-state, within-month
association between disaster_count and business_applications.

This is explicitly an OBSERVATIONAL / CORRELATIONAL analysis. Fixed effects
rule out a specific, important class of confounds (anything constant within
a state or common across all states in a month) but do not establish
causality — states could still experience disasters and business-formation
shocks from a shared third cause within the same month (e.g. a hurricane
depresses both regional economic activity and, mechanically, the disaster
count). No causal language is used anywhere in this module's output.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from linearmodels.panel import PanelOLS
from linearmodels.panel.results import PanelEffectsResults
from scipy import stats

from src.config_loader import load_config
from src.logging_setup import get_logger

logger = get_logger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"


class ModelingError(Exception):
    """Raised when panel preparation or model fitting fails validation."""


# =============================================================================
# Data preparation
# =============================================================================

def load_analysis_dataset(config: Dict) -> pd.DataFrame:
    """
    Load the Module 2 final analysis dataset.

    Parameters
    ----------
    config : dict

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ModelingError
        If the file doesn't exist — Module 2 must run first.
    """
    path = (
        PROJECT_ROOT / config["paths"]["data_processed"]
        / config["processed_outputs"]["final_analysis_dataset"]
    )
    if not path.exists():
        raise ModelingError(f"Final analysis dataset not found at {path}. Run Module 2 first.")

    df = pd.read_csv(path)
    logger.info("Loaded final analysis dataset: %s rows", len(df))
    return df


def prepare_panel_data(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Validate and reshape the flat analysis dataset into a properly indexed
    panel ready for PanelOLS: MultiIndex of (state, year_month), no missing
    dependent-variable rows, no duplicate panel keys.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_analysis_dataset() — flat, one row per state-month.
    config : dict
        Parsed project configuration.

    Returns
    -------
    pd.DataFrame
        Indexed by (state, year_month), with year_month converted to a
        Timestamp (PanelOLS requires a numeric or date-like time index —
        a raw Period index is rejected).

    Raises
    ------
    ModelingError
        If duplicate panel keys exist, or if the entity/time columns are
        missing entirely.

    Assumptions
    ----------
    - Dropping rows with missing business_applications is the right call
      here (rather than imputing) — those rows are the still-unpublished
      2026 months (confirmed in Module 3), not a data quality problem, so
      there's nothing to legitimately impute.
    - The resulting panel is unbalanced only insofar as the most recent
      months are missing for all states equally (not state-specific
      dropout), so this doesn't bias the fixed-effects estimates toward
      any particular state.
    """
    entity_col = config["modeling"]["entity_column"]
    time_col = config["modeling"]["time_column"]
    dep_var = config["modeling"]["dependent_variable"]

    logger.info("Preparing panel data: entity=%s, time=%s, dependent=%s", entity_col, time_col, dep_var)

    if entity_col not in df.columns or time_col not in df.columns:
        raise ModelingError(f"Expected columns '{entity_col}' and '{time_col}' not found in input data.")

    panel = df.copy()

    # PanelOLS needs a numeric or date-like time index — a pandas Period
    # (used elsewhere in the project) is rejected outright, so convert to
    # a Timestamp here rather than carrying Period through the whole pipeline.
    panel[time_col] = pd.PeriodIndex(panel[time_col], freq="M").to_timestamp()

    n_before = len(panel)
    panel = panel.dropna(subset=[dep_var]).copy()
    n_dropped = n_before - len(panel)
    logger.info(
        "Dropped %s rows with missing %s (unpublished recent months); %s rows remain",
        n_dropped, dep_var, len(panel),
    )

    duplicate_keys = panel.duplicated(subset=[entity_col, time_col]).sum()
    if duplicate_keys > 0:
        raise ModelingError(
            f"{duplicate_keys} duplicate ({entity_col}, {time_col}) keys found — "
            "cannot build a valid panel index."
        )
    logger.info("Confirmed no duplicate (%s, %s) panel keys", entity_col, time_col)

    panel = panel.set_index([entity_col, time_col]).sort_index()

    # Balance check: is every entity observed the same number of times?
    # An unbalanced panel isn't automatically wrong (see docstring), but we
    # want it logged explicitly rather than silently fit.
    obs_per_entity = panel.groupby(level=0).size()
    is_balanced = obs_per_entity.nunique() == 1
    logger.info(
        "Panel balance check: %s (%s distinct observation counts across %s entities, range %s-%s)",
        "balanced" if is_balanced else "unbalanced",
        obs_per_entity.nunique(), len(obs_per_entity), obs_per_entity.min(), obs_per_entity.max(),
    )

    logger.info(
        "Panel ready: %s observations, %s entities (states), %s time periods",
        len(panel), panel.index.get_level_values(0).nunique(), panel.index.get_level_values(1).nunique(),
    )
    return panel


# =============================================================================
# Model fitting
# =============================================================================

def fit_twfe_model(panel: pd.DataFrame, config: Dict) -> PanelEffectsResults:
    """
    Fit the two-way fixed effects panel regression:

        business_applications_it = beta * disaster_count_it
                                    + alpha_i (state FE)
                                    + gamma_t (month-year FE)
                                    + epsilon_it

    with standard errors clustered by state to account for serial
    correlation within a state's error terms over time.

    Parameters
    ----------
    panel : pd.DataFrame
        Output of prepare_panel_data() — MultiIndex (state, year_month).
    config : dict

    Returns
    -------
    PanelEffectsResults
        The fitted linearmodels result object.
    """
    dep_var = config["modeling"]["dependent_variable"]
    indep_var = config["modeling"]["independent_variable"]

    formula = f"{dep_var} ~ {indep_var} + EntityEffects + TimeEffects"
    logger.info("Fitting PanelOLS: %s", formula)

    model = PanelOLS.from_formula(formula, data=panel)
    results = model.fit(cov_type=config["modeling"]["cov_type"], cluster_entity=True)

    logger.info(
        "Model fit complete: nobs=%s, rsquared_within=%.4f, coef(%s)=%.3f, p=%.4f",
        results.nobs, results.rsquared_within, indep_var,
        results.params[indep_var], results.pvalues[indep_var],
    )
    return results


# =============================================================================
# Diagnostics extraction
# =============================================================================

def extract_diagnostics(results: PanelEffectsResults, config: Dict) -> Dict:
    """
    Pull every requested diagnostic out of the fitted results object into a
    plain dict, ready to save as a table and feed into the interpretation
    generator.

    Parameters
    ----------
    results : PanelEffectsResults
    config : dict

    Returns
    -------
    dict
        Model-level diagnostics (nobs, entity/time counts, R-squareds,
        F-statistic) and coefficient-level diagnostics (estimate, SE,
        t-stat, p-value, 95% CI) for the independent variable.
    """
    indep_var = config["modeling"]["independent_variable"]
    conf_int = results.conf_int()

    diagnostics = {
        "n_observations": int(results.nobs),
        "n_entities": int(results.entity_info["total"]),
        "n_time_periods": int(results.time_info["total"]),
        "rsquared_within": float(results.rsquared_within),
        "rsquared_between": float(results.rsquared_between),
        "rsquared_overall": float(results.rsquared_overall),
        "f_statistic": float(results.f_statistic.stat),
        "f_statistic_pvalue": float(results.f_statistic.pval),
        "coefficient": float(results.params[indep_var]),
        "std_error": float(results.std_errors[indep_var]),
        "t_statistic": float(results.tstats[indep_var]),
        "p_value": float(results.pvalues[indep_var]),
        "ci_lower_95": float(conf_int.loc[indep_var, "lower"]),
        "ci_upper_95": float(conf_int.loc[indep_var, "upper"]),
    }
    logger.info("Extracted diagnostics: %s", diagnostics)
    return diagnostics


# =============================================================================
# Saving tables and reports
# =============================================================================

def save_regression_table(results: PanelEffectsResults, diagnostics: Dict, config: Dict) -> Path:
    """
    Save a tidy one-row-per-coefficient regression table to outputs/tables/.

    Parameters
    ----------
    results : PanelEffectsResults
    diagnostics : dict
        Output of extract_diagnostics().
    config : dict

    Returns
    -------
    Path
    """
    indep_var = config["modeling"]["independent_variable"]
    table = pd.DataFrame([{
        "variable": indep_var,
        "coefficient": diagnostics["coefficient"],
        "std_error": diagnostics["std_error"],
        "t_statistic": diagnostics["t_statistic"],
        "p_value": diagnostics["p_value"],
        "ci_lower_95": diagnostics["ci_lower_95"],
        "ci_upper_95": diagnostics["ci_upper_95"],
    }])
    out_path = PROJECT_ROOT / config["paths"]["outputs_tables"] / config["modeling"]["outputs"]["regression_table"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    logger.info("Saved regression table to %s", out_path)
    return out_path


def save_diagnostics_table(diagnostics: Dict, config: Dict) -> Path:
    """Save the full diagnostics dict as a long-format CSV to outputs/tables/."""
    table = pd.DataFrame(list(diagnostics.items()), columns=["metric", "value"])
    out_path = PROJECT_ROOT / config["paths"]["outputs_tables"] / config["modeling"]["outputs"]["diagnostics_table"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    logger.info("Saved diagnostics table to %s", out_path)
    return out_path


def save_full_summary(results: PanelEffectsResults, config: Dict) -> Path:
    """Save linearmodels' full text summary (the standard academic regression printout)."""
    out_path = PROJECT_ROOT / config["paths"]["outputs_reports"] / config["modeling"]["outputs"]["full_summary_txt"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(str(results))
    logger.info("Saved full regression summary to %s", out_path)
    return out_path


# =============================================================================
# Interpretation generator
# =============================================================================

def generate_interpretation(diagnostics: Dict, config: Dict) -> str:
    """
    Auto-generate a plain-language interpretation of the regression output:
    sign, statistical significance, economic magnitude, and practical
    takeaway — explicitly framed as association, not causation.

    Parameters
    ----------
    diagnostics : dict
        Output of extract_diagnostics().
    config : dict

    Returns
    -------
    str
        The interpretation text (also saved to outputs/reports/).
    """
    coef = diagnostics["coefficient"]
    p_value = diagnostics["p_value"]
    ci_lower = diagnostics["ci_lower_95"]
    ci_upper = diagnostics["ci_upper_95"]
    indep_var = config["modeling"]["independent_variable"]
    dep_var = config["modeling"]["dependent_variable"]

    sign_word = "positive" if coef > 0 else "negative"
    is_significant = p_value < 0.05
    significance_word = "statistically significant" if is_significant else "not statistically significant"

    lines = [
        "TWFE Regression Interpretation",
        "=" * 40,
        "",
        f"Sign of the coefficient: {sign_word} ({coef:.3f})",
        f"Statistical significance at the 5% level: {significance_word} (p = {p_value:.4f})",
        f"95% confidence interval: [{ci_lower:.3f}, {ci_upper:.3f}]",
        "",
        "Economic interpretation:",
        (
            f"  Holding state-specific factors and national month-to-month conditions fixed, "
            f"one additional disaster declaration in a state-month is associated with a "
            f"{sign_word} change of approximately {abs(coef):.1f} {dep_var} in that same state-month."
        ),
        "",
        "Practical interpretation:",
    ]

    if is_significant:
        lines.append(
            f"  This association is unlikely to be due to chance alone at conventional "
            f"significance thresholds. However, the coefficient's practical size should be "
            f"weighed against typical monthly {dep_var} counts (see Module 3 descriptive "
            f"statistics) before treating it as economically meaningful."
        )
    else:
        lines.append(
            f"  The confidence interval for this coefficient includes zero, meaning we cannot "
            f"rule out the possibility of no association between {indep_var} and {dep_var} once "
            f"state and month-year fixed effects are accounted for. This is consistent with the "
            f"weak raw correlation observed in Module 3's exploratory analysis."
        )

    lines += [
        "",
        "IMPORTANT — this is an observational, correlational analysis.",
        (
            "  State and month-year fixed effects rule out confounding from anything constant "
            "within a state (size, industry mix, baseline entrepreneurial activity) or common "
            "across all states in a given month (national recessions, seasonality, policy "
            "shocks). They do NOT establish that disasters cause changes in business "
            "applications. A state could experience both a disaster and an economic shift in "
            "the same month due to a shared underlying cause, and reverse causality or omitted "
            "state-specific time-varying factors cannot be ruled out by this model. Do not "
            "present this coefficient as causal evidence."
        ),
    ]

    interpretation_text = "\n".join(lines)

    out_path = (
        PROJECT_ROOT / config["paths"]["outputs_reports"]
        / config["modeling"]["outputs"]["interpretation_txt"]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(interpretation_text)
    logger.info("Saved interpretation to %s", out_path)

    return interpretation_text


# =============================================================================
# Visualizations
# =============================================================================

def _save_figure(fig: plt.Figure, filename: str, config: Dict) -> Path:
    out_path = PROJECT_ROOT / config["paths"]["outputs_figures"] / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved figure: %s", out_path)
    return out_path


def plot_coefficient(diagnostics: Dict, config: Dict) -> Path:
    """Point estimate with 95% CI error bar for the single regressor."""
    indep_var = config["modeling"]["independent_variable"]
    coef, lower, upper = diagnostics["coefficient"], diagnostics["ci_lower_95"], diagnostics["ci_upper_95"]
    err = [[coef - lower], [upper - coef]]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.errorbar([0], [coef], yerr=err, fmt="o", markersize=12, capsize=8, color="#1f77b4", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks([0])
    ax.set_xticklabels([indep_var])
    ax.set_ylabel("Coefficient Estimate")
    ax.set_title("TWFE Coefficient Estimate with 95% CI\n(state + month-year fixed effects, clustered SE)")
    fig.tight_layout()
    return _save_figure(fig, "twfe_coefficient_plot.png", config)


def plot_confidence_interval(diagnostics: Dict, config: Dict) -> Path:
    """
    A horizontal confidence-interval range plot — visually distinct from
    the coefficient point-estimate plot, emphasizing the interval itself
    and whether it crosses zero.
    """
    indep_var = config["modeling"]["independent_variable"]
    coef, lower, upper = diagnostics["coefficient"], diagnostics["ci_lower_95"], diagnostics["ci_upper_95"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hlines(y=0, xmin=lower, xmax=upper, color="#ff7f0e", linewidth=8, alpha=0.7)
    ax.axvline(coef, color="black", linewidth=2, label="Point estimate")
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, label="Zero (no association)")
    ax.set_yticks([])
    ax.set_xlabel(f"Coefficient on {indep_var}")
    ax.set_title("95% Confidence Interval for the Disaster Count Coefficient")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return _save_figure(fig, "twfe_confidence_interval_plot.png", config)


def plot_residual_histogram(results: PanelEffectsResults, config: Dict) -> Path:
    """Histogram of model residuals — a first check for normality/skew."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(results.resids, bins=60, color="#9467bd", ax=ax)
    ax.set_title("TWFE Model Residuals — Distribution")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return _save_figure(fig, "twfe_residual_histogram.png", config)


def plot_residual_qq(results: PanelEffectsResults, config: Dict) -> Path:
    """Q-Q plot of residuals against a normal distribution."""
    fig, ax = plt.subplots(figsize=(7, 7))
    stats.probplot(results.resids, dist="norm", plot=ax)
    ax.set_title("TWFE Model Residuals — Normal Q-Q Plot")
    ax.get_lines()[0].set_markerfacecolor("#1f77b4")
    ax.get_lines()[0].set_markeredgecolor("#1f77b4")
    ax.get_lines()[0].set_markersize(3)
    ax.get_lines()[1].set_color("#d62728")
    fig.tight_layout()
    return _save_figure(fig, "twfe_residual_qq_plot.png", config)


def plot_fitted_vs_observed(results: PanelEffectsResults, panel: pd.DataFrame, config: Dict) -> Path:
    """
    Scatter of fitted values (including estimated state/time effects, not
    just the exogenous part) against observed business_applications.

    Note: PanelOLS.fitted_values returns only the exogenous-regressor
    contribution (disaster_count * beta); the absorbed entity/time effects
    live separately in results.estimated_effects. We add them back together
    here so the fitted values are on the same scale as the actual outcome
    — otherwise this plot would be uninformative (fitted values would only
    take two possible values, since there's a single regressor).
    """
    dep_var = config["modeling"]["dependent_variable"]
    fitted_full = results.fitted_values["fitted_values"] + results.estimated_effects["estimated_effects"]
    observed = panel[dep_var]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(observed, fitted_full, alpha=0.15, s=12, color="#2ca02c")
    lims = [min(observed.min(), fitted_full.min()), max(observed.max(), fitted_full.max())]
    ax.plot(lims, lims, color="black", linestyle="--", linewidth=1, label="Perfect fit (45° line)")
    ax.set_xlabel("Observed Business Applications")
    ax.set_ylabel("Fitted Values (incl. state + month-year effects)")
    ax.set_title("Fitted vs. Observed Values")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return _save_figure(fig, "twfe_fitted_vs_observed.png", config)


# =============================================================================
# Orchestration
# =============================================================================

def run_modeling_pipeline() -> PanelEffectsResults:
    """
    Execute the full Module 4 pipeline: load data, prepare the panel, fit
    the TWFE model, extract diagnostics, save all tables/reports/figures,
    and generate the plain-language interpretation.

    Returns
    -------
    PanelEffectsResults
        The fitted model, for interactive use (e.g. in a notebook).
    """
    logger.info("=" * 70)
    logger.info("Module 4: Two-Way Fixed Effects Regression — starting")
    logger.info("=" * 70)

    config = load_config()

    raw_final = load_analysis_dataset(config)
    panel = prepare_panel_data(raw_final, config)

    results = fit_twfe_model(panel, config)
    diagnostics = extract_diagnostics(results, config)

    save_regression_table(results, diagnostics, config)
    save_diagnostics_table(diagnostics, config)
    save_full_summary(results, config)
    generate_interpretation(diagnostics, config)

    plot_coefficient(diagnostics, config)
    plot_confidence_interval(diagnostics, config)
    plot_residual_histogram(results, config)
    plot_residual_qq(results, config)
    plot_fitted_vs_observed(results, panel, config)

    logger.info("=" * 70)
    logger.info("Module 4: Two-Way Fixed Effects Regression — completed successfully")
    logger.info("=" * 70)

    return results


if __name__ == "__main__":
    run_modeling_pipeline()
