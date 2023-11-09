from src.vmx_ppo.unconstrained_demand.recommendation import (
    prep_rules_for_aggregation,
    calculate_weighted_ir,
    aggregate_rules,
    setup_config,
    handle_dealer_data,
    get_recommendations_dataframe,
    recommend_quartile_above_no_sales,
    log_sg_q_no_sales,
    log_sg_q_no_sales_no_rules,
    apply_rules,
)
import pandas as pd
import numpy as np


def test_prep_rules_for_aggregation():
    df_rules_w_sales = pd.DataFrame(
        {
            "dealer_number_latest": [1, 2, 3, 4, 5, 6],
            "correlation_category": [
                "None",
                "Positive",
                "Negative",
                "None",
                "Negative",
                "Positive",
            ],
            "rec_retail_installation_rate": [0.8, 0.4, 0.2, 0.8, 0.2, 0.5],
            "quartile": ["low", "high", "lowest", "high", "medium", "medium"],
            "quantity_retail_grain": [1, 2, 3, 4, 5, 6],
        }
    )

    df_region_dealer = pd.DataFrame(
        {
            "dealer_number_latest": [1, 2, 3, 4, 5, 6],
            "region_code": [100, 100, 100, 100, 100, 100],
            "district_code": [1, 1, 1, 2, 2, 2],
        }
    )

    result_df1, result_df2 = prep_rules_for_aggregation(
        df_rules_w_sales, df_region_dealer, aggregation_threshold=0.5
    )
    expected_result_df1 = pd.DataFrame.from_dict(
        {
            "3": {
                "dealer_number_latest": 4,
                "correlation_category": "None",
                "rec_retail_installation_rate": 0.8,
                "quartile": "high",
                "quantity_retail_grain": 4,
                "correlation_category_agg": "None/Positive",
                "region_code": 100,
                "district_code": 2,
                "rule_agg_level": "dealer-grain-segment-corr-quartile",
                "quantity_retail_group": 4,
            }
        },
        orient="index",
    )
    expected_result_df2 = pd.DataFrame(
        {
            "dealer_number_latest": [1, 2, 3, 5, 6],
            "correlation_category": [
                "None",
                "Positive",
                "Negative",
                "Negative",
                "Positive",
            ],
            "rec_retail_installation_rate": [0.8, 0.4, 0.2, 0.2, 0.5],
            "quartile": ["low", "high", "lowest", "medium", "medium"],
            "quantity_retail_grain": [1, 2, 3, 5, 6],
            "correlation_category_agg": [
                "None/Positive",
                "None/Positive",
                "Negative",
                "Negative",
                "None/Positive",
            ],
            "region_code": [100, 100, 100, 100, 100],
            "district_code": [1, 1, 1, 2, 2],
        }
    )
    pd.testing.assert_frame_equal(
        expected_result_df1.reset_index(drop=True), result_df1.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        expected_result_df2.reset_index(drop=True), result_df2.reset_index(drop=True)
    )


def test_calculate_weighted_ir():
    """Test calculate_weighted_ir function."""
    rec_df = pd.DataFrame(
        {
            "rec_retail_installation_rate": [0.8, 0.5, 0.2],
            "quantity": [1, 2, 3],
        }
    )
    result_series = calculate_weighted_ir(rec_df, "quantity", "district-grain")
    expected_result_series = pd.Series(
        {
            "rec_retail_installation_rate": 0.4,
            "quantity": 6,
            "rule_agg_level": "district-grain",
        }
    )
    assert result_series.shape == (3,)
    pd.testing.assert_series_equal(
        expected_result_series, result_series, check_index=False
    )

    # Test total quantity = 0
    rec_df["quantity"] = 0
    result_series = calculate_weighted_ir(rec_df, "quantity", "district-grain")
    expected_result_series = pd.Series(
        {
            "rec_retail_installation_rate": 0.5,
            "quantity": 0,
            "rule_agg_level": "district-grain",
        }
    )
    assert result_series.shape == (3,)
    pd.testing.assert_series_equal(
        expected_result_series, result_series, check_index=False
    )


def test_recommend_quartile_above_no_sales():
    """Test recommend_quartile_above_no_sales function."""
    df_recommend_raw = pd.DataFrame(
        {
            "series_name": ["A", "A", "A", "A"],
            "label": [0, 0, 0, 0],
            "quartile_quantity_retail_grain": [4, 4, 4, 4],
            "accessory_code": ["AA", "BB", "CC", "DD"],
            "quartile_quantity_retail_ppo": [0, 1, 2, 3],
        }
    )
    agg = "A"
    segment = 2
    aggregation_level = "series"
    param_dict = {"cluster_col": "label"}
    result_df = recommend_quartile_above_no_sales(
        df_recommend_raw, agg, segment, aggregation_level, param_dict
    )
    expected_result_df = pd.DataFrame(
        {
            "series_name": ["A", "A", "A", "A"],
            "accessory_code": ["AA", "BB", "CC", "DD"],
            "quartile_quantity_retail_ppo": [0, 1, 2, 3],
            "quartile_quantity_retail_grain": [4, 4, 4, 4],
            "rec_retail_installation_rate": [0, 0.25, 0.5, 0.75],
            "label": [2, 2, 2, 2],
        }
    )
    pd.testing.assert_frame_equal(
        result_df.reset_index(drop=True), expected_result_df.reset_index(drop=True)
    )


def test_log_sg_q_no_sales():
    # new model
    bpi_log_grain_no_rules_df_updated = pd.DataFrame(
        {
            "series_name": ["A", "A"],
            "model_number": [1000, 2000],
            "category": [
                "new model in Procon not in history",
                "model in history not allocated",
            ],
            "rules_created": [None, None],
        }
    )
    df_model_current = pd.DataFrame(
        {
            "series_name": ["A", "A"],
            "business_month": [202201, 202201],
            "model_number": [1000, 1010],
            "dealer_number_latest": [1, 1],
        }
    )
    result_df = log_sg_q_no_sales(
        bpi_log_grain_no_rules_df_updated, df_model_current, "model"
    )
    expected_result_df = pd.DataFrame(
        {
            "series_name": ["A", "A", "A", "A"],
            "model_number": [1000, 2000, 1000, 1010],
            "category": [
                "new model in Procon not in history",
                "model in history not allocated",
                "Segment-Quartile above has no sales",
                "Segment-Quartile above has no sales",
            ],
            "rules_created": [None, None, "Yes", "Yes"],
            "business_month": [None, None, 202201, 202201],
            "dealer_number_latest": [None, None, 1, 1],
        }
    )
    pd.testing.assert_frame_equal(
        result_df.reset_index(drop=True), expected_result_df.reset_index(drop=True)
    )


def test_log_sg_q_no_sales_no_rules():
    df_model_current = pd.DataFrame.from_dict(
        {
            "1": {
                "series_name": "A",
                "model_number": 1000,
                "dealer_number_latest": 1,
                "business_month": 202201,
            }
        },
        orient="index",
    )
    bpi_log_grain_no_rules_df_updated = pd.DataFrame(
        {
            "series_name": ["A", "A", "A"],
            "dealer_number_latest": [None, None, 1],
            "business_month": [None, None, 202201],
            "model_number": [1000, 2000, 1000],
            "category": [
                "new model in Procon not in history",
                "model in history not allocated",
                "Segment-Quartile above has no sales",
            ],
            "rules_created": [None, None, None],
        }
    )
    result_df = log_sg_q_no_sales_no_rules(
        bpi_log_grain_no_rules_df_updated, df_model_current, "model"
    )
    expected_result_df = pd.DataFrame(
        {
            "series_name": ["A", "A", "A"],
            "dealer_number_latest": [None, None, 1],
            "business_month": [None, None, 202201],
            "model_number": [1000, 2000, 1000],
            "category": [
                "new model in Procon not in history",
                "model in history not allocated",
                "Segment-Quartile above has no sales",
            ],
            "rules_created": [None, None, "No"],
        }
    )
    pd.testing.assert_frame_equal(
        result_df.reset_index(drop=True), expected_result_df.reset_index(drop=True)
    )


def test_aggregate_rules():
    df_rules = pd.DataFrame(
        {
            "series_name": ["A", "A", "A", "A", "A", "A"],
            "label": [0, 0, 0, 0, 0, 0],
            "dealer_number_latest": [1, 2, 3, 4, 5, 6],
            "quartile": [
                "high",
                "low",
                "low",
                "medium",
                "high",
                "Grain not enough dealers",
            ],
            "correlation_category": ["None", "None", "None", "None", "None", "None"],
            "correl": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            "rec_retail_installation_rate": [0.5, 0.6, 0.5, 0.1, 0.2, 0.3],
            "accessory_code": ["AA", "AA", "AA", "AA", "AA", "AA"],
        }
    )
    df_sale_grain_historical = pd.DataFrame(
        {
            "series_name": ["A", "A", "A", "A", "A", "A"],
            "dealer_number_latest": [1, 2, 3, 4, 5, 6],
            "quantity_retail_grain": [1, 10, 5, 2, 3, 4],
        }
    )
    df_region_dealer = pd.DataFrame(
        {
            "dealer_number_latest": [1, 2, 3, 4, 5, 6],
            "region_code": [100, 100, 100, 100, 100, 100],
            "district_code": [2, 2, 2, 2, 2, 3],
        }
    )
    result_df = aggregate_rules(
        df_rules,
        df_sale_grain_historical,
        df_region_dealer,
        series_aggregation_level="series",
        param_dict={
            "rule_aggregation": {
                "series": "district-grain",
                "aggregation_threshold": 0.5,
            }
        },
        series="A",
        target_month_min=202307,
    )
    expected_result_df = pd.DataFrame(
        {
            "series_name": ["A", "A", "A", "A", "A", "A"],
            "label": [0, 0, 0, 0, 0, 0],
            "dealer_number_latest": [4, 5, 6, 2, 3, 1],
            "quartile": [
                "medium",
                "high",
                "Grain not enough dealers",
                "low",
                "low",
                "high",
            ],
            "correlation_category": ["None", "None", "None", "None", "None", "None"],
            "correl": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            "accessory_code": ["AA", "AA", "AA", "AA", "AA", "AA"],
            "quantity_retail_grain": [2, 3, 4, 10, 5, 1],
            "correlation_category_agg": [
                "None/Positive",
                "None/Positive",
                "None/Positive",
                "None/Positive",
                "None/Positive",
                "None/Positive",
            ],
            "region_code": [100, 100, 100, 100, 100, 100],
            "district_code": [2, 2, 3, 2, 2, 2],
            "rec_retail_installation_rate": [0.16, 0.16, 0.3, 0.56667, 0.56667, 0.5],
            "quantity_retail_group": [5, 5, 4, 15, 15, 1],
            "rule_agg_level": [
                "district-grain",
                "district-grain",
                "district-grain",
                "district-grain-segment-corr-quartile",
                "district-grain-segment-corr-quartile",
                "dealer-grain-segment-corr-quartile",
            ],
        }
    )
    pd.testing.assert_frame_equal(
        result_df.reset_index(drop=True), expected_result_df.reset_index(drop=True)
    )


def test_setup_config():
    param_dict = {"mode": "lite", "q4_booster": 1.2, "q1_reducer": 0.8}

    mode, q4_booster, q1_reducer, agg_var = setup_config("series", param_dict)

    assert mode == "lite"
    assert q4_booster == 1.2
    assert q1_reducer == 0.8
    assert agg_var == "series_name"


def test_get_recommendations_dataframe():
    df_getsudo_data = pd.DataFrame(
        {
            "series_name": ["A", "A", "A", "A", "A", "A"],
            "dealer_number_latest": [1, 2, 3, 4, 5, 6],
            "label": [0, 0, 0, 0, 0, 0],
            "correlation_category": ["None", "None", "None", "None", "None", "None"],
            "correl": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            "business_month": [202201, 202201, 202201, 202201, 202201, 202201],
            "quartile": [
                "lowest",
                "low",
                "medium",
                "high",
                "Grain not enough dealers",
                "medium",
            ],
            "no_quartile_new_grain": ["No", "No", "No", "No", "No", "Yes"],
        }
    )

    df_rules_lowest_quartile = pd.DataFrame(
        {
            "label": [0],
            "series_name": ["A"],
            "accessory_code": ["AA"],
            "rec_retail_installation_rate": [0.1],
        }
    )

    df_rules_low_quartile = df_rules_lowest_quartile.copy()
    df_rules_low_quartile["accessory_code"] = "BB"
    df_rules_low_quartile["rec_retail_installation_rate"] = [0.1]

    df_rules_medium_quartile = df_rules_lowest_quartile.copy()
    df_rules_medium_quartile["accessory_code"] = "CC"
    df_rules_medium_quartile["rec_retail_installation_rate"] = [0.1]

    df_rules_high_quartile = df_rules_lowest_quartile.copy()
    df_rules_high_quartile["accessory_code"] = "DD"
    df_rules_high_quartile["rec_retail_installation_rate"] = [0.1]

    df_rules_high_quartile_dealer_lvl = pd.DataFrame(
        {
            "series_name": ["A"],
            "label": [0],
            "quartile": ["high"],
            "correlation_category": ["None"],
            "correl": [0.1],
            "dealer_number_latest": [4],
            "accessory_code": ["EE"],
            "rec_retail_installation_rate": [0.1],
        }
    )

    df_rules_lowest_quartile_dealer_lvl = pd.DataFrame(
        {
            "series_name": ["A"],
            "label": [0],
            "quartile": ["lowest"],
            "correlation_category": ["None"],
            "correl": [0.1],
            "dealer_number_latest": [1],
            "accessory_code": ["FF"],
            "rec_retail_installation_rate": [0.1],
        }
    )

    q4_booster = 0.0
    q1_reducer = 0.0

    df_rules_no_quartile_dealer_lvl = pd.DataFrame(
        {
            "series_name": ["A"],
            "label": [0],
            "quartile": ["Grain not enough dealers"],
            "correlation_category": ["None"],
            "correl": [0.1],
            "accessory_code": ["GG"],
            "rec_retail_installation_rate": [0.1],
            "dealer_number_latest": [5],
        }
    )

    expected_result_df = pd.DataFrame(
        {
            "series_name": ["A"],
            "label": [0],
            "quartile": ["high"],
            "correlation_category": ["None"],
            "correl": [0.1],
            "dealer_number_latest": [4],
            "accessory_code": ["EE"],
            "rec_retail_installation_rate": [0.1],
        }
    )

    result_df = get_recommendations_dataframe(
        "lite",
        "Positive",
        "high",
        df_rules_high_quartile_dealer_lvl,
        df_rules_lowest_quartile_dealer_lvl,
        df_rules_no_quartile_dealer_lvl,
        df_rules_high_quartile,
        df_rules_lowest_quartile,
        df_rules_low_quartile,
        df_rules_medium_quartile,
        q4_booster,
        q1_reducer,
    )

    assert result_df.shape == (1, 8)
    pd.testing.assert_frame_equal(
        result_df.reset_index(drop=True),
        expected_result_df.reset_index(drop=True),
        check_like=True,
    )


def test_handle_dealer_data():
    agg_cols = ["label", "series_name"]
    agg_var = "series_name"
    agg = "A"
    segment = 0
    df_getsudo_data = pd.DataFrame(
        {
            "series_name": ["A", "A", "A", "A", "A", "A"],
            "dealer_number_latest": [1, 2, 3, 4, 5, 6],
            "label": [0, 0, 0, 0, 0, 0],
            "correlation_category": ["None", "None", "None", "None", "None", "None"],
            "correl": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            "business_month": [202201, 202201, 202201, 202201, 202201, 202201],
            "quartile": [
                "lowest",
                "low",
                "medium",
                "high",
                "Grain not enough dealers",
                "medium",
            ],
            "no_quartile_new_grain": ["No", "No", "No", "No", "No", "Yes"],
        }
    )

    df_rules_lowest_quartile = pd.DataFrame(
        {
            "label": [0],
            "series_name": ["A"],
            "accessory_code": ["AA"],
            "rec_retail_installation_rate": [0.1],
        }
    )
    result_df = handle_dealer_data(
        pd.DataFrame(),  # Assuming bpi_log_grain_no_rules_df_updated is an empty DataFrame for simplicity
        "series",
        {
            "mode": "lite",
            "cluster_col": "label",
            "q4_booster": 0.0,
            "q1_reducer": 0.0,
        },
        df_rules_lowest_quartile,
        df_getsudo_data,
        agg_cols,
        agg_var,
        agg,
        segment,
    )

    # Constructing the expected output
    expected_df = df_rules_lowest_quartile.copy()
    expected_df = expected_df[expected_df["series_name"] == agg]
    expected_df = pd.merge(
        expected_df,
        df_getsudo_data[df_getsudo_data["label"] == segment],
        on=agg_cols,
        how="left",
    )

    expected_df["no_model_history_quartile_segment"] = "No"

    pd.testing.assert_frame_equal(result_df, expected_df, check_like=True)


def test_apply_rules():
    """Test apply_rules function."""
    df_getsudo_data = pd.DataFrame(
        {
            "series_name": ["A", "A", "A", "A", "A", "A"],
            "dealer_number_latest": [1, 2, 3, 4, 5, 6],
            "label": [0, 0, 0, 0, 0, 0],
            "correlation_category": ["None", "None", "None", "None", "None", "None"],
            "correl": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            "business_month": [202201, 202201, 202201, 202201, 202201, 202201],
            "quartile": [
                "lowest",
                "low",
                "medium",
                "high",
                "Grain not enough dealers",
                "medium",
            ],
            "no_quartile_new_grain": ["No", "No", "No", "No", "No", "Yes"],
        }
    )

    df_rules_lowest_quartile = pd.DataFrame(
        {
            "label": [0],
            "series_name": ["A"],
            "accessory_code": ["AA"],
            "rec_retail_installation_rate": [0.1],
        }
    )

    df_rules_low_quartile = df_rules_lowest_quartile.copy()
    df_rules_low_quartile["accessory_code"] = "BB"
    df_rules_low_quartile["rec_retail_installation_rate"] = [0.1]

    df_rules_medium_quartile = df_rules_lowest_quartile.copy()
    df_rules_medium_quartile["accessory_code"] = "CC"
    df_rules_medium_quartile["rec_retail_installation_rate"] = [0.1]

    df_rules_high_quartile = df_rules_lowest_quartile.copy()
    df_rules_high_quartile["accessory_code"] = "DD"
    df_rules_high_quartile["rec_retail_installation_rate"] = [0.1]

    df_rules_high_quartile_dealer_lvl = pd.DataFrame(
        {
            "series_name": ["A"],
            "label": [0],
            "quartile": ["high"],
            "correlation_category": ["None"],
            "correl": [0.1],
            "dealer_number_latest": [4],
            "accessory_code": ["EE"],
            "rec_retail_installation_rate": [0.1],
        }
    )

    df_rules_lowest_quartile_dealer_lvl = pd.DataFrame(
        {
            "series_name": ["A"],
            "label": [0],
            "quartile": ["lowest"],
            "correlation_category": ["None"],
            "correl": [0.1],
            "dealer_number_latest": [1],
            "accessory_code": ["FF"],
            "rec_retail_installation_rate": [0.1],
        }
    )

    df_rules_grain_not_enough_dealers = pd.DataFrame(
        {
            "series_name": ["A"],
            "label": [0],
            "quartile": ["Grain not enough dealers"],
            "correlation_category": ["None"],
            "correl": [0.1],
            "accessory_code": ["GG"],
            "rec_retail_installation_rate": [0.1],
            "dealer_number_latest": [5],
        }
    )

    result_df, bpi_log_updated = apply_rules(
        df_getsudo_data,
        df_rules_lowest_quartile,
        df_rules_low_quartile,
        df_rules_medium_quartile,
        df_rules_high_quartile,
        df_rules_high_quartile_dealer_lvl,
        df_rules_lowest_quartile_dealer_lvl,
        df_rules_grain_not_enough_dealers,
        pd.DataFrame(),
        aggregation_level="series",
        param_dict={
            "mode": "lite",
            "cluster_col": "label",
            "q4_booster": 0.0,
            "q1_reducer": 0.0,
        },
        brand="brand",
    )

    expected_result_df = pd.DataFrame(
        {
            "label": [0, 0, 0, 0, 0, 0],
            "series_name": ["A", "A", "A", "A", "A", "A"],
            "accessory_code": ["BB", "CC", "DD", "DD", "EE", "GG"],
            "dealer_number_latest": [1, 2, 3, 6, 4, 5],
            "quartile": [
                "lowest",
                "low",
                "medium",
                "medium",
                "high",
                "Grain not enough dealers",
            ],
            "correlation_category": ["None", "None", "None", "None", "None", "None"],
            "correl": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            "business_month": [202201, 202201, 202201, 202201, 202201, 202201],
            "rec_retail_installation_rate": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            "no_model_history_quartile_segment": ["No", "No", "No", "No", "No", "No"],
            "no_quartile_new_grain": ["No", "No", "No", "Yes", np.nan, np.nan],
        }
    )
    assert result_df.shape == (6, 11)
    pd.testing.assert_frame_equal(
        result_df.reset_index(drop=True),
        expected_result_df.reset_index(drop=True),
        check_like=True,
    )

    # Test negative correlations
    df_getsudo_data["correlation_category"] = "Negative"
    df_rules_high_quartile_dealer_lvl["correlation_category"] = "Negative"
    df_rules_lowest_quartile_dealer_lvl["correlation_category"] = "Negative"
    df_rules_grain_not_enough_dealers["correlation_category"] = "Negative"

    result_df, bpi_log_updated = apply_rules(
        df_getsudo_data,
        df_rules_lowest_quartile,
        df_rules_low_quartile,
        df_rules_medium_quartile,
        df_rules_high_quartile,
        df_rules_high_quartile_dealer_lvl,
        df_rules_lowest_quartile_dealer_lvl,
        df_rules_grain_not_enough_dealers,
        pd.DataFrame(),
        aggregation_level="series",
        param_dict={
            "mode": "lite",
            "cluster_col": "label",
            "q4_booster": 0.0,
            "q1_reducer": 0.0,
        },
        brand="brand",
    )

    expected_result_df = pd.DataFrame(
        {
            "label": [0, 0, 0, 0, 0, 0],
            "series_name": ["A", "A", "A", "A", "A", "A"],
            "accessory_code": ["FF", "AA", "BB", "BB", "CC", "GG"],
            "rec_retail_installation_rate": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            "no_model_history_quartile_segment": ["No", "No", "No", "No", "No", "No"],
            "dealer_number_latest": [1, 2, 3, 6, 4, 5],
            "quartile": [
                "lowest",
                "low",
                "medium",
                "medium",
                "high",
                "Grain not enough dealers",
            ],
            "correlation_category": [
                "Negative",
                "Negative",
                "Negative",
                "Negative",
                "Negative",
                "Negative",
            ],
            "correl": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            "business_month": [202201, 202201, 202201, 202201, 202201, 202201],
            "no_quartile_new_grain": [np.nan, "No", "No", "Yes", "No", np.nan],
            "no_model_history_quartile_segment": ["No", "No", "No", "No", "No", "No"],
        }
    )

    assert result_df.shape == (6, 11)
    pd.testing.assert_frame_equal(
        result_df.reset_index(drop=True),
        expected_result_df.reset_index(drop=True),
        check_like=True,
    )


def test_apply_rules_with_q4_booster_and_q1_reducer():
    """Test apply_rules function with q4_booster parameter."""
    df_getsudo_data = pd.DataFrame(
        {
            "series_name": ["A", "A", "A", "A", "A", "A"],
            "dealer_number_latest": [1, 2, 3, 4, 5, 6],
            "label": [0, 0, 0, 0, 0, 0],
            "correlation_category": [
                "Positive",
                "Positive",
                "Positive",
                "Positive",
                "Positive",
                "Positive",
            ],
            "correl": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            "business_month": [202201, 202201, 202201, 202201, 202201, 202201],
            "quartile": [
                "lowest",
                "low",
                "medium",
                "high",
                "Grain not enough dealers",
                "medium",
            ],
            "no_quartile_new_grain": ["No", "No", "No", "No", "No", "Yes"],
        }
    )

    df_rules_lowest_quartile = pd.DataFrame(
        {
            "label": [0],
            "series_name": ["A"],
            "accessory_code": ["AA"],
            "rec_retail_installation_rate": [0.1],
        }
    )

    df_rules_low_quartile = df_rules_lowest_quartile.copy()
    df_rules_low_quartile["accessory_code"] = "BB"
    df_rules_low_quartile["rec_retail_installation_rate"] = [0.1]

    df_rules_medium_quartile = df_rules_lowest_quartile.copy()
    df_rules_medium_quartile["accessory_code"] = "CC"
    df_rules_medium_quartile["rec_retail_installation_rate"] = [0.1]

    df_rules_high_quartile = df_rules_lowest_quartile.copy()
    df_rules_high_quartile["accessory_code"] = "DD"
    df_rules_high_quartile["rec_retail_installation_rate"] = [0.1]

    df_rules_high_quartile_dealer_lvl = pd.DataFrame(
        {
            "series_name": ["A"],
            "label": [0],
            "quartile": ["high"],
            "correlation_category": ["Positive"],
            "correl": [0.1],
            "dealer_number_latest": [4],
            "accessory_code": ["EE"],
            "rec_retail_installation_rate": [0.1],
        }
    )

    df_rules_lowest_quartile_dealer_lvl = pd.DataFrame(
        {
            "series_name": ["A"],
            "label": [0],
            "quartile": ["lowest"],
            "correlation_category": ["Positive"],
            "correl": [0.1],
            "dealer_number_latest": [1],
            "accessory_code": ["FF"],
            "rec_retail_installation_rate": [0.1],
        }
    )

    df_rules_grain_not_enough_dealers = pd.DataFrame(
        {
            "series_name": ["A"],
            "label": [0],
            "quartile": ["Grain not enough dealers"],
            "correlation_category": ["Positive"],
            "correl": [0.1],
            "accessory_code": ["GG"],
            "rec_retail_installation_rate": [0.1],
            "dealer_number_latest": [5],
        }
    )

    result_df, bpi_log_updated = apply_rules(
        df_getsudo_data,
        df_rules_lowest_quartile,
        df_rules_low_quartile,
        df_rules_medium_quartile,
        df_rules_high_quartile,
        df_rules_high_quartile_dealer_lvl,
        df_rules_lowest_quartile_dealer_lvl,
        df_rules_grain_not_enough_dealers,
        pd.DataFrame(),
        aggregation_level="series",
        param_dict={
            "mode": "lite",
            "cluster_col": "label",
            "q4_booster": 0.05,
            "q1_reducer": 0.05,
        },
        brand="brand",
    )

    expected_result_df = pd.DataFrame(
        {
            "label": [0, 0, 0, 0, 0, 0],
            "series_name": ["A", "A", "A", "A", "A", "A"],
            "accessory_code": ["BB", "CC", "DD", "DD", "EE", "GG"],
            "dealer_number_latest": [1, 2, 3, 6, 4, 5],
            "quartile": [
                "lowest",
                "low",
                "medium",
                "medium",
                "high",
                "Grain not enough dealers",
            ],
            "correlation_category": [
                "Positive",
                "Positive",
                "Positive",
                "Positive",
                "Positive",
                "Positive",
            ],
            "correl": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            "business_month": [202201, 202201, 202201, 202201, 202201, 202201],
            "rec_retail_installation_rate": [0.1, 0.1, 0.1, 0.1, 0.1 * 1.05, 0.1],
            "no_model_history_quartile_segment": ["No", "No", "No", "No", "No", "No"],
            "no_quartile_new_grain": ["No", "No", "No", "Yes", np.nan, np.nan],
        }
    )

    assert result_df.shape == (6, 11)
    pd.testing.assert_frame_equal(
        result_df.reset_index(drop=True),
        expected_result_df.reset_index(drop=True),
        check_like=True,
    )

    # Test negative correlations
    df_getsudo_data["correlation_category"] = "Negative"
    df_rules_high_quartile_dealer_lvl["correlation_category"] = "Negative"
    df_rules_lowest_quartile_dealer_lvl["correlation_category"] = "Negative"
    df_rules_grain_not_enough_dealers["correlation_category"] = "Negative"

    result_df, bpi_log_updated = apply_rules(
        df_getsudo_data,
        df_rules_lowest_quartile,
        df_rules_low_quartile,
        df_rules_medium_quartile,
        df_rules_high_quartile,
        df_rules_high_quartile_dealer_lvl,
        df_rules_lowest_quartile_dealer_lvl,
        df_rules_grain_not_enough_dealers,
        pd.DataFrame(),
        aggregation_level="series",
        param_dict={
            "mode": "lite",
            "cluster_col": "label",
            "q4_booster": 0.05,
            "q1_reducer": 0.05,
        },
        brand="brand",
    )

    expected_result_df = pd.DataFrame(
        {
            "label": [0, 0, 0, 0, 0, 0],
            "series_name": ["A", "A", "A", "A", "A", "A"],
            "accessory_code": ["FF", "AA", "BB", "BB", "CC", "GG"],
            "rec_retail_installation_rate": [0.095, 0.1, 0.1, 0.1, 0.1, 0.1],
            "no_model_history_quartile_segment": ["No", "No", "No", "No", "No", "No"],
            "dealer_number_latest": [1, 2, 3, 6, 4, 5],
            "quartile": [
                "lowest",
                "low",
                "medium",
                "medium",
                "high",
                "Grain not enough dealers",
            ],
            "correlation_category": [
                "Negative",
                "Negative",
                "Negative",
                "Negative",
                "Negative",
                "Negative",
            ],
            "correl": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            "business_month": [202201, 202201, 202201, 202201, 202201, 202201],
            "no_quartile_new_grain": [np.nan, "No", "No", "Yes", "No", np.nan],
            "no_model_history_quartile_segment": ["No", "No", "No", "No", "No", "No"],
        }
    )

    assert result_df.shape == (6, 11)
    pd.testing.assert_frame_equal(
        result_df.reset_index(drop=True),
        expected_result_df.reset_index(drop=True),
        check_like=True,
    )
