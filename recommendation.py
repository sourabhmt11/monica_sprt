import pandas as pd
import logging
from typing import Dict, Tuple, List
from src.vmx_ppo.utils.io import load_data, save_data
from src.vmx_ppo.utils.timer import timer_func
from src.vmx_ppo.utils.enums import AggregationLevel
import concurrent.futures
from concurrent.futures import wait

logger = logging.getLogger("src.vmx_ppo.unconstrained_demand.recommendation")
INSUFFICIENT_GRAIN_DEALERS = "Grain not enough dealers"


def map_correlation_category(df: pd.DataFrame) -> pd.DataFrame:
    """Map treatment for 'None' & 'Positive' correlation to 'None/Positive'."""
    df.loc[:, "correlation_category_agg"] = df["correlation_category"].apply(
        lambda x: "None/Positive" if x in ["None", "Positive"] else "Negative"
    )
    return df


def merge_with_region_and_district(
    df: pd.DataFrame, df_region: pd.DataFrame
) -> pd.DataFrame:
    """Merge sales data with their respective region and district codes."""
    df_region = df_region[
        ["dealer_number_latest", "region_code", "district_code"]
    ].copy()
    merged_df = pd.merge(df, df_region, how="left")

    assert (
        merged_df["region_code"].notnull().all()
        and merged_df["district_code"].notnull().all()
    ), "Some rules don't have associated region or district."

    return merged_df


def filter_non_aggregated_rules(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Extract rules that don't require aggregation based on specific criteria."""
    return df.query(
        f"rec_retail_installation_rate >= {threshold} and "
        "((correlation_category_agg == 'Negative' and quartile == 'lowest') or "
        "(correlation_category_agg == 'None/Positive' and quartile == 'high'))"
    ).copy()


def filter_aggregated_rules(
    df: pd.DataFrame, dealer_lvl_df: pd.DataFrame
) -> pd.DataFrame:
    """Determine rules for aggregation."""
    return (
        df.merge(dealer_lvl_df.drop_duplicates(), how="left", indicator=True)
        .query('_merge == "left_only"')
        .drop("_merge", axis=1)
    ).copy()


def add_additional_columns_to_non_aggregated(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns specifying the aggregation level and quantity retail group."""
    df.loc[
        :, "rule_agg_level"
    ] = AggregationLevel.DEALER_GRAIN_SEGMENT_CORR_QUARTILE.value
    df.loc[:, "quantity_retail_group"] = df["quantity_retail_grain"]
    return df


def prep_rules_for_aggregation(
    df_rules_w_sales: pd.DataFrame,
    df_region_dealer: pd.DataFrame,
    aggregation_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare rules for aggregation by splitting out rules to be kept at dealer-level

    Args:
        df_rules_w_sales (pd.DataFrame): rule recommendations joined with dealer sales data
        df_region_dealer (pd.DataFrame): region-dealer mapping
        aggregation_threshold (float): threshold for aggregation

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: dataframe of rules to be aggregated, rules to be kept at dealer-level
    """

    # 1. Correlation Category Aggregation:
    # Given the nature of the aggregation treatment for 'None' & 'Positive' correlation,
    # we're mapping these into a single category named 'None/Positive'.
    # This step simplifies the subsequent aggregation logic.
    df_rules_w_sales = map_correlation_category(df_rules_w_sales)

    # 2. Augment Sales Data with Geographical Information:
    # To have a more comprehensive dataframe, we're merging sales data with their
    # respective region and district codes. This will facilitate region-based aggregation.
    df_rules_w_sales_region = merge_with_region_and_district(
        df_rules_w_sales, df_region_dealer
    )

    # Ensure every rule's associated dealer is mapped to a region and district.
    # This is crucial for ensuring consistency in later steps.
    assert (df_rules_w_sales_region["region_code"].isnull().sum() == 0) and (
        df_rules_w_sales_region["district_code"].isnull().sum() == 0
    )

    # 3. Non-Aggregation Rules Selection:
    # Some rules don't require aggregation based on specific criteria. Here, we're extracting those rules
    # based on the provided threshold, correlation category, and quartile.
    rules_dealer_lvl = filter_non_aggregated_rules(
        df_rules_w_sales_region, aggregation_threshold
    )

    # 4. Rules for Aggregation:
    # Once non-aggregated rules are identified, the remainder is determined for aggregation.
    # We achieve this by performing an anti-join to get records only present in the original dataset
    # and not in the non-aggregated rules dataset.
    rules_to_agg = filter_aggregated_rules(df_rules_w_sales_region, rules_dealer_lvl)
    # Confirm the combined count of aggregated and non-aggregated rules matches the original set.
    # This is a sanity check to ensure no data has been inadvertently dropped or duplicated.
    assert len(rules_to_agg) + len(rules_dealer_lvl) == len(
        df_rules_w_sales
    ), "Mismatch in number of rules after processing."

    # 5. Enhance Non-Aggregated Rules Data:
    # For the non-aggregated rules, we add additional data columns specifying the aggregation level and
    # mirroring the quantity retail grain into a new column called 'quantity_retail_group'.
    if not rules_dealer_lvl.empty:
        rules_dealer_lvl = add_additional_columns_to_non_aggregated(rules_dealer_lvl)

    return rules_dealer_lvl, rules_to_agg


def calculate_weighted_ir(x: pd.DataFrame, qty_col: str, agg_level: str) -> pd.Series:
    """Calculate weighted installation rate for aggregation

    Args:
        x (pd.DataFrame): dataframe of rules to be aggregated
        qty_col (str): column name for quantity
        agg_level (str): rule aggregation level (e.g. district-grain-segment-quartile-corr)

    Returns:
        pd.Series: Series of weighted installation rate
    """
    rec_retail_installation_rate = (
        x["rec_retail_installation_rate"] * x[qty_col]
    ).sum() / (x[qty_col].sum())
    quantity_retail_group = x[qty_col].sum()
    if quantity_retail_group == 0:
        rec_retail_installation_rate = x["rec_retail_installation_rate"].median()
    return pd.Series(
        {
            "rec_retail_installation_rate": rec_retail_installation_rate,
            "quantity_retail_group": quantity_retail_group,
            "rule_agg_level": agg_level,
        }
    )


def validate_aggregation_level(level: str):
    """Validate the provided aggregation level."""
    if not AggregationLevel.is_valid(level):
        logger.exception(f"No such rule_aggregation_level: {level}")
        raise Exception(f"No such rule_aggregation_level: {level}")


def add_sales_to_dataframe(
    df_rules: pd.DataFrame, df_sales: pd.DataFrame, grain_cols: List[str]
) -> pd.DataFrame:
    """Add sales data to the rules dataframe."""
    df_sales_dealer = df_sales[grain_cols].copy()
    df_rules_w_sales = pd.merge(df_rules, df_sales_dealer, how="left")
    df_rules_w_sales["quantity_retail_grain"].fillna(0, inplace=True)
    return df_rules_w_sales


def is_no_aggregation_needed(level: str) -> bool:
    """Check if the rules require no further aggregation."""
    return level == AggregationLevel.DEALER_GRAIN_SEGMENT_CORR_QUARTILE.value


def process_no_aggregation(df_rules_w_sales: pd.DataFrame) -> pd.DataFrame:
    """Process rules that do not require any further aggregation."""
    df_rules_w_sales.loc[
        :, "rule_agg_level"
    ] = AggregationLevel.DEALER_GRAIN_SEGMENT_CORR_QUARTILE.value
    df_rules_w_sales.loc[:, "quantity_retail_group"] = df_rules_w_sales[
        "quantity_retail_grain"
    ]
    return df_rules_w_sales


def log_rules_info(rules_dealer_lvl: pd.DataFrame, rules_to_agg: pd.DataFrame):
    """Log information about number of rules at dealer level and those to be aggregated."""
    logger.info(
        f"Keeping {len(rules_dealer_lvl):,} rules at dealer-level, aggregating {len(rules_to_agg):,} rules further"
    )


def is_district_aggregation_needed(level: str) -> bool:
    """Check if the rules require district-level aggregation."""
    return level == AggregationLevel.DISTRICT_GRAIN_SEGMENT_CORR_QUARTILE.value


def aggregate_rules(
    df_rules: pd.DataFrame,
    df_sale_grain_historical: pd.DataFrame,
    df_region_dealer: pd.DataFrame,
    series_aggregation_level: str,
    param_dict: Dict,
    series: str,
    target_month_min: int,
) -> pd.DataFrame:
    """
    Aggregate rules to specified aggregation level using historical sales data.

    This function takes in dealer-level rules and aggregates them based on
    specified rules and conditions. The aggregation is based on sales data,
    region-dealer mappings, and predefined aggregation levels.

    Args:
        df_rules (pd.DataFrame): Dataframe containing rules at dealer level.
        df_sale_grain_historical (pd.DataFrame): Historical sales data to weigh rules.
        df_region_dealer (pd.DataFrame): Mapping between regions and dealers.
        series_aggregation_level (str): Aggregation level, either "series" or "model".
        param_dict (Dict): Dictionary containing parameters and aggregation thresholds.
        series (str): Vehicle series name
        target_month_min (int): Represents the value of a specific month

    Returns:
        pd.DataFrame: Dataframe of aggregated rules.
    """
    # # Extract rule aggregation threshold and level from the parameter dictionary1
    # rule_aggregation_threshold = param_dict["rule_aggregation"]["aggregation_threshold"]
    # rule_aggregation_level = param_dict["rule_aggregation"][series_aggregation_level]
    #
    # # Ensure that the provided aggregation level is one of the accepted values
    # validate_aggregation_level(rule_aggregation_level)

    # Extract rule aggregation threshold and level from the parameter dictionary and validating those values
    def rule_agg_params_and_validation(param_dict, series_aggregation_level):

        rule_aggregation_threshold = param_dict["rule_aggregation"]["aggregation_threshold"]
        rule_aggregation_level = param_dict["rule_aggregation"][series_aggregation_level]

        # Ensure that the provided aggregation level is one of the accepted values
        validate_aggregation_level(rule_aggregation_level)

        return rule_aggregation_threshold, rule_aggregation_level

    rule_aggregation_threshold, rule_aggregation_level = rule_agg_params_and_validation(param_dict,series_aggregation_level)

    # In the case where there are no cars/rules to aggregate, simply log and return the empty dataframe
    if df_rules.empty:
        logger.info(f"No {series} allocated in the {target_month_min}.")
        return df_rules

    # Define grain columns for merging and computations based on the series aggregation level
    grain_cols = ["series_name", "dealer_number_latest", "quantity_retail_grain"]
    if series_aggregation_level in ("model", "grade"):
        grain_cols += ["model_number"]

    # Merge rules dataframe with sales data for weighted aggregation in subsequent steps
    df_rules_w_sales = add_sales_to_dataframe(
        df_rules, df_sale_grain_historical, grain_cols
    )

    # If the rules don't need any further aggregation (as specified), handle them separately
    if is_no_aggregation_needed(rule_aggregation_level):
        return process_no_aggregation(df_rules_w_sales)

    # Split the rules into two categories: ones that remain at dealer level and ones that need further aggregation
    rules_dealer_lvl, rules_to_agg = prep_rules_for_aggregation(
        df_rules_w_sales,
        df_region_dealer,
        rule_aggregation_threshold,
    )

    # Log information about number of rules at dealer level and those that will be aggregated
    # Log the number of rules that fall into each category
    log_rules_info(rules_dealer_lvl, rules_to_agg)

    # Define the columns to be used for aggregation
    grain_agg_cols = ["series_name"]
    if series_aggregation_level in ("model", "grade"):
        grain_agg_cols += ["model_number"]

    # TODO: Modularize the remaining pieces of the function

    # Aggregating rules that fall below the threshold to district level
    rules_district_agg = rules_to_agg.groupby(
        [
            "region_code",
            "district_code",
            "label",
            "correlation_category_agg",
            "quartile",
            "accessory_code",
        ]
        + grain_agg_cols,
        as_index=False,
    ).apply(
        calculate_weighted_ir,
        qty_col="quantity_retail_grain",
        agg_level=AggregationLevel.DISTRICT_GRAIN_SEGMENT_CORR_QUARTILE.value,
    )

    # Log the number of rules aggregated to the district level
    logger.info(
        f"Aggregated {len(rules_to_agg):,} rules to {len(rules_district_agg):,} at the district-level"
    )

    # If only district level aggregation is needed, then merge results and return
    if is_district_aggregation_needed(rule_aggregation_level):
        rules_agg_expanded = pd.merge(
            rules_to_agg.drop("rec_retail_installation_rate", axis=1),
            rules_district_agg,
            how="inner",
        )
        # number of rules showing should be the same
        assert len(rules_to_agg) == len(rules_agg_expanded)
        # add dealer-level rules back to aggregated rules
        df_rules_final = pd.concat(
            [rules_agg_expanded, rules_dealer_lvl], axis=0, ignore_index=True
        )
        return df_rules_final

    # For rules requiring further aggregation, split them based on the threshold
    # This step segregates rules that meet the threshold and ones that need more aggregation

    # Filter rules that are above the threshold
    # Using the query function, we filter rows where the "rec_retail_installation_rate" is
    # greater than or equal to the defined threshold
    rules_district_lvl_above_threshold = rules_district_agg.query(
        f"rec_retail_installation_rate >= {rule_aggregation_threshold}"
    )

    # Merge (or join) the original rules dataframe with these filtered rules, essentially creating a combined dataset.
    # This is done to retain all columns while focusing only on rules that are above the threshold
    rules_district_lvl_above_expanded = pd.merge(
        rules_to_agg.drop("rec_retail_installation_rate", axis=1),
        rules_district_lvl_above_threshold,
        how="inner",
    )

    # Filter rules that are below the threshold
    rules_district_lvl_below_threshold = rules_district_agg.query(
        f"rec_retail_installation_rate < {rule_aggregation_threshold}"
    )

    # Similarly, merge the original rules dataframe with these filtered rules for the below threshold category
    rules_district_lvl_below_expanded = pd.merge(
        rules_to_agg.drop("rec_retail_installation_rate", axis=1),
        rules_district_lvl_below_threshold,
        how="inner",
    )

    # Assert to ensure we have not lost or gained any rules during the split process
    assert len(rules_district_lvl_below_expanded) + len(
        rules_district_lvl_above_expanded
    ) == len(rules_to_agg)

    # Logging the number of rules retained at the district level and the number of rules
    # that will undergo further aggregation
    logger.info(
        f"Keeping {len(rules_district_lvl_above_expanded):,} rules at district-level, "
        f"aggregating {len(rules_district_lvl_below_expanded):,} rules further"
    )

    # Depending on the rule aggregation level, decide the columns on which aggregation should be performed.
    # These columns act as grouping parameters when performing further aggregations

    # For DISTRICT_GRAIN aggregation level
    if rule_aggregation_level == AggregationLevel.DISTRICT_GRAIN.value:
        agg_cols = ["region_code", "district_code", "accessory_code"] + grain_agg_cols

    # For DISTRICT_GRAIN_SEGMENT aggregation level
    elif rule_aggregation_level == AggregationLevel.DISTRICT_GRAIN_SEGMENT.value:
        agg_cols = [
            "region_code",
            "district_code",
            "label",
            "accessory_code",
        ] + grain_agg_cols

    # For DISTRICT_GRAIN_SEGMENT_CORR aggregation level
    elif rule_aggregation_level == AggregationLevel.DISTRICT_GRAIN_SEGMENT_CORR.value:
        agg_cols = [
            "region_code",
            "district_code",
            "correlation_category_agg",
            "label",
            "accessory_code",
        ] + grain_agg_cols
    else:
        logger.exception(
            f"rule_aggregation_level unaccounted for: {rule_aggregation_level}"
        )
        raise Exception(
            f"rule_aggregation_level unaccounted for: {rule_aggregation_level}"
        )

    # If there are rules that are below the threshold and need further aggregation
    if not rules_district_lvl_below_threshold.empty:
        # Aggregate the rules using the columns defined above and apply the calculate_weighted_ir
        # function to perform the aggregation
        rules_agg = rules_district_lvl_below_threshold.groupby(
            agg_cols, as_index=False
        ).apply(
            calculate_weighted_ir,
            qty_col="quantity_retail_group",
            agg_level=rule_aggregation_level,
        )

        # Expand the aggregated rules by merging them with the rules_district_lvl_below_expanded
        # dataframe to ensure all columns are present
        rules_agg_expanded = pd.merge(
            rules_district_lvl_below_expanded.drop(
                [
                    "rec_retail_installation_rate",
                    "quantity_retail_group",
                    "rule_agg_level",
                ],
                axis=1,
            ),
            rules_agg,
            how="inner",
        )
    else:
        # If there are no rules below the threshold, initialize an empty dataframe
        rules_agg_expanded = pd.DataFrame({})

    # Combining all datasets - the aggregated rules, the rules retained at district level and the rules at dealer level
    df_rules_final = pd.concat(
        [rules_agg_expanded, rules_district_lvl_above_expanded, rules_dealer_lvl],
        axis=0,
        ignore_index=True,
    )

    # Final assertion to ensure data integrity.
    # The total number of rules in the final dataset should be the same as the original rules dataset.
    assert len(df_rules_final) == len(df_rules)
    return df_rules_final


def recommend_quartile_above_no_sales(
    df_recommend_raw: pd.DataFrame,
    agg: int,
    cluster: int,
    aggregation_level: str,
    param_dict: Dict,
) -> pd.DataFrame:
    """
    Generates recommendation if segment & quartile above had no sales of the series/model. Recommendation is based on
    weighted IR for other models within same series and segment, other models within same series in other
    segments, or other segments (if run at series-level)
    Args:
        df_recommend_raw(pd.DataFrame): recommended rules dataframe on segmentation, model, ppo level
        agg(int): grain (model number or series name)
        cluster(int): segmentation number
        param_dict: input parameters
    Returns:
        pd.DataFrame: dataframe with recommended PPO for the grain
    """
    # select out the segmentation
    df_recommend = df_recommend_raw[
        df_recommend_raw[param_dict["cluster_col"]] == cluster
    ].copy(deep=True)

    # if entire segmentation, quartile above did not get allocated any other models, or aggregation level is series
    # then look across segmentation
    if df_recommend.empty:
        df_recommend = df_recommend_raw.copy(deep=True)
        # if entire quartile above did not get allocated any other models, then return empty dataframe
        if df_recommend_raw.empty:
            return pd.DataFrame()
        # drop segmentation and later assign segment to be the segmentation id that is passed in
        groupby_col = ["series_name"]
    else:
        groupby_col = [param_dict["cluster_col"], "series_name"]

    # get quantity retail grain for each grain, segmentation
    if aggregation_level in ("model", "grade"):
        df_q_quantity_retail_grain = df_recommend[
            [
                param_dict["cluster_col"],
                "series_name",
                "model_number",
                "quartile_quantity_retail_grain",
            ]
        ].drop_duplicates()
    elif aggregation_level == "series":
        df_q_quantity_retail_grain = df_recommend[
            [
                param_dict["cluster_col"],
                "series_name",
                "quartile_quantity_retail_grain",
            ]
        ].drop_duplicates()
    else:
        logger.exception(f"aggregation_level unaccounted for: {aggregation_level}")
        raise Exception(f"aggregation_level unaccounted for: {aggregation_level}")

    # get quantity retail grain across model (across segmentation if needed)
    df_q_quantity_retail_grain = df_q_quantity_retail_grain.groupby(
        groupby_col, as_index=False
    ).agg({"quartile_quantity_retail_grain": "sum"})

    # get quantity retail ppo for each ppo across models (across segmentation if needed)
    df_recommend = df_recommend.groupby(
        groupby_col + ["accessory_code"], as_index=False
    ).agg({"quartile_quantity_retail_ppo": "sum"})

    # merge quantity retail grain on to quantity retail ppo
    df_recommend = pd.merge(
        df_recommend, df_q_quantity_retail_grain, on=groupby_col, how="left"
    )

    # calculate rec retail install rate
    df_recommend["rec_retail_installation_rate"] = (
        df_recommend["quartile_quantity_retail_ppo"]
        / df_recommend["quartile_quantity_retail_grain"]
    )
    # fill IR NAs here, otherwise rule aggregation will error out
    df_recommend["rec_retail_installation_rate"] = df_recommend[
        "rec_retail_installation_rate"
    ].fillna(0)

    if aggregation_level in ("model", "grade"):
        # put back model_number col as the new model, later used to apply rules to new model
        df_recommend["model_number"] = agg

    # if aggregated acorss segmentation, put back label col as the passed in segmentation, later used to apply rules
    if param_dict["cluster_col"] not in df_recommend.columns:
        df_recommend[param_dict["cluster_col"]] = cluster
    return df_recommend


def log_sg_q_no_sales(
    bpi_log_grain_no_rules_df_updated: pd.DataFrame,
    df_model_current: pd.DataFrame,
    aggregation_level: str,
) -> pd.DataFrame:
    """
    Log the grain -dealer combos where the segment - quartile above have no sales (no rec rules)
    Args:
        bpi_log_grain_no_rules_df_updated (pd.DataFrame): bpi log with new models / models not allocated in ProCon
        df_model_current (pd.DataFrame): ProCon data
        aggregation_level (str): grain level, model or series
    Returns:
        pd.DataFrame: BPI log with grain - dealer combo where the segment - quartile above have no sales (no rec rules)
        appended
    """
    if aggregation_level in ("model", "grade"):
        # get new models
        new_model_set = set(
            bpi_log_grain_no_rules_df_updated[
                bpi_log_grain_no_rules_df_updated["category"]
                == "new model in ProCon not in history"
            ]["model_number"].values
        )
        # get all set of grain where quartile above has no sales
        sg_q_no_sales_set = set(df_model_current["model_number"].values)

        # get models-dealer combos where the quartile above has no sales (no rules) within the same segment
        # and the models are not new models
        sg_q_no_sales_list = list(sg_q_no_sales_set - new_model_set)

        # if not all models are new models
        if len(sg_q_no_sales_list) != 0:
            cols_to_include_sg_q_no_sales = [
                "business_month",
                "series_name",
                "model_number",
                "dealer_number_latest",
            ]
            sg_q_no_sales_df = df_model_current[
                df_model_current["model_number"].isin(sg_q_no_sales_list)
            ][cols_to_include_sg_q_no_sales].drop_duplicates()
            sg_q_no_sales_df["category"] = "Segment-Quartile above has no sales"
            sg_q_no_sales_df["rules_created"] = "Yes"

            bpi_log_grain_no_rules_df_append = bpi_log_grain_no_rules_df_updated.append(
                sg_q_no_sales_df
            )
        else:
            bpi_log_grain_no_rules_df_append = bpi_log_grain_no_rules_df_updated.copy(
                deep=True
            )
    elif aggregation_level == "series":
        cols_to_include_sg_q_no_sales = [
            "business_month",
            "series_name",
            "dealer_number_latest",
        ]
        sg_q_no_sales_df = df_model_current[
            cols_to_include_sg_q_no_sales
        ].drop_duplicates()
        sg_q_no_sales_df["category"] = "Segment-Quartile above has no sales"
        sg_q_no_sales_df["rules_created"] = "Yes"
        bpi_log_grain_no_rules_df_append = bpi_log_grain_no_rules_df_updated.append(
            sg_q_no_sales_df
        )
    else:
        logger.exception(f"aggregation_level unaccounted for: {aggregation_level}")
        raise Exception(f"aggregation_level unaccounted for: {aggregation_level}")

    return bpi_log_grain_no_rules_df_append


def log_sg_q_no_sales_no_rules(
    bpi_log_grain_no_rules_df_updated: pd.DataFrame,
    df_model_current: pd.DataFrame,
    aggregation_level: str,
) -> pd.DataFrame:
    """
    Mark rules_created = 'No' if quartile above have no sales for any grain across all segments
    Args:
        bpi_log_grain_no_rules_df_updated (pd.DataFrame): bpi log with new models / models not allocated in ProCon /
        grain - dealer where segment - quartile above have no sales
        df_model_current (pd.DataFrame): ProCon data
        aggregation_level (str): grain level, model or series
    Returns:
        pd.DataFrame: bpi log with rules_created column updated if quartile above have no sales for any grain across
        all segments
    """
    agg_cols = ["business_month", "series_name", "dealer_number_latest"]
    if aggregation_level in ("model", "grade"):
        agg_cols += ["model_number"]
    df_no_rules = df_model_current[agg_cols]
    df_no_rules["still_no_rules"] = "Yes"

    bpi_log_grain_no_rules_df_updated = pd.merge(
        bpi_log_grain_no_rules_df_updated, df_no_rules, on=agg_cols, how="left"
    )
    bpi_log_grain_no_rules_df_updated.loc[
        bpi_log_grain_no_rules_df_updated["still_no_rules"] == "Yes", "rules_created"
    ] = "No"
    bpi_log_grain_no_rules_df_updated.drop(columns=["still_no_rules"], inplace=True)

    return bpi_log_grain_no_rules_df_updated


def setup_config(
    aggregation_level: str, param_dict: Dict
) -> Tuple[str, float, float, str]:
    """
    Configures settings based on the provided aggregation level and parameter dictionary.
    Returns the rule mode, quartile boosters and reducers, and the appropriate aggregation variable.
    """
    rule_mode = param_dict["mode"]
    q4_booster = float(param_dict["q4_booster"])
    q1_reducer = float(param_dict["q1_reducer"])
    logger.info(f"Q4 Booster set to {q4_booster}")
    logger.info(f"Q1 Reducer set to {q1_reducer}")
    if aggregation_level == "series":
        agg_var = "series_name"
    elif aggregation_level in ("model", "grade"):
        agg_var = "model_number"
    return rule_mode, q4_booster, q1_reducer, agg_var


def get_recommendations_dataframe(
    rule_mode: str,
    correlation: str,
    quartile_class: str,
    df_rules_high_quartile_dealer_lvl: pd.DataFrame,
    df_rules_lowest_quartile_dealer_lvl: pd.DataFrame,
    df_rules_no_quartile_dealer_lvl: pd.DataFrame,
    df_rules_high_quartile: pd.DataFrame,
    df_rules_lowest_quartile: pd.DataFrame,
    df_rules_low_quartile: pd.DataFrame,
    df_rules_medium_quartile: pd.DataFrame,
    q4_booster: float,
    q1_reducer: float,
) -> pd.DataFrame:
    """
    Provides recommendations based on the given correlation and quartile class using the specified rule mode.
    Based on the Correlation Category it will reduce or boost the installation rates

    Args:
        rule_mode (str): The mode for rule application, either 'lite' or 'maxi'.
        correlation (str): The correlation category, can be 'Negative', 'Positive' or 'None'.
        quartile_class (str): The quartile classification. Values include 'lowest', 'low', 'medium', 'high', or 'Grain not enough dealers'.
        df_rules_high_quartile_dealer_lvl (pd.DataFrame): Rules dataframe specific to high quartile at the dealer level.
        df_rules_lowest_quartile_dealer_lvl (pd.DataFrame): Rules dataframe specific to lowest quartile at the dealer level.
        df_rules_no_quartile_dealer_lvl (pd.DataFrame): Rules dataframe specific to grains without sufficient dealers.
        df_rules_high_quartile (pd.DataFrame): Rules dataframe for high quartile.
        df_rules_lowest_quartile (pd.DataFrame): Rules dataframe for lowest quartile.
        df_rules_low_quartile (pd.DataFrame): Rules dataframe for low quartile.
        df_rules_medium_quartile (pd.DataFrame): Rules dataframe for medium quartile.
        q4_booster (float): Boost FACTOR for the 4th quartile.
        q1_reducer (float): Reduction FACTOR for the 1st quartile.

    Returns:
        pd.DataFrame: A dataframe containing the recommendations based on the provided criteria.
    """

    recommend_map = {
        ("Negative", "lite"): {
            "lowest": (df_rules_lowest_quartile_dealer_lvl, q1_reducer),
            INSUFFICIENT_GRAIN_DEALERS: df_rules_no_quartile_dealer_lvl,
            "low": df_rules_lowest_quartile,
            "medium": df_rules_low_quartile,
            "high": df_rules_medium_quartile,
        },
        ("Positive", "lite"): {
            "high": (df_rules_high_quartile_dealer_lvl, q4_booster),
            INSUFFICIENT_GRAIN_DEALERS: df_rules_no_quartile_dealer_lvl,
            "medium": df_rules_high_quartile,
            "low": df_rules_medium_quartile,
            "lowest": df_rules_low_quartile,
        },
        ("None", "lite"): {
            "high": df_rules_high_quartile_dealer_lvl,
            INSUFFICIENT_GRAIN_DEALERS: df_rules_no_quartile_dealer_lvl,
            "medium": df_rules_high_quartile,
            "low": df_rules_medium_quartile,
            "lowest": df_rules_low_quartile,
        },
        ("Negative", "maxi"): {
            "lowest": (df_rules_lowest_quartile_dealer_lvl, q1_reducer),
            INSUFFICIENT_GRAIN_DEALERS: df_rules_no_quartile_dealer_lvl,
            None: df_rules_lowest_quartile,
        },
        ("Positive", "maxi"): {
            "high": (df_rules_high_quartile_dealer_lvl, q4_booster),
            INSUFFICIENT_GRAIN_DEALERS: df_rules_no_quartile_dealer_lvl,
            None: df_rules_high_quartile,
        },
        (None, "maxi"): {
            "high": df_rules_high_quartile_dealer_lvl,
            INSUFFICIENT_GRAIN_DEALERS: df_rules_no_quartile_dealer_lvl,
            None: df_rules_high_quartile,
        },
    }
    recommendation = recommend_map.get((correlation, rule_mode)).get(
        quartile_class, df_rules_lowest_quartile
    )

    # if the recommendation is a tuple, which would contain a DataFrame and a FACTOR
    if isinstance(recommendation, tuple):
        # Unpack the tuple into a DataFrame and a FACTOR
        df_recommend, FACTOR = recommendation

        # Define the column name to which the FACTOR will be applied
        INSTALL_RATE_COL = "rec_retail_installation_rate"

        # If the correlation is "Negative", apply a reduction FACTOR to the column values
        # Apply a reduction to each value in the column, ensuring it does not go below 0.0
        if correlation == "Negative":
            df_recommend[INSTALL_RATE_COL] = df_recommend[INSTALL_RATE_COL].apply(
                lambda x: max(0.0, (1.0 - FACTOR) * x)
            )

        elif correlation == "Positive":
            # If the correlation is not "Negative", apply a boost FACTOR to the column values
            # This will be the case for "Positive" correlation
            df_recommend[INSTALL_RATE_COL] = df_recommend[INSTALL_RATE_COL].apply(
                lambda x: min(1.0, (1.0 + FACTOR) * x)
            )

    else:
        # If the recommendation is not a tuple, it's a DataFrame that we copy directly
        df_recommend = recommendation.copy(deep=True)

    return df_recommend


def handle_dealer_data(
    bpi_log_grain_no_rules_df_updated: pd.DataFrame,
    aggregation_level: str,
    param_dict: Dict,
    df_recommend: pd.DataFrame,
    df_model_current: pd.DataFrame,
    agg_cols: List,
    agg_var: str,
    agg: str,
    segment: str,
) -> pd.DataFrame:
    """
    The function takes in a details about sales, models of cars, accessories, other details, and
    provides recommendations to boost the sales of cars/accessories to the respective dealers.

    Args:
    - bpi_log_grain_no_rules_df_updated: A log of any dealers who didn't get new advice.
    - aggregation_level: The detail level of data we're looking at.
    - param_dict: A dictionary of all the different configurations and information.
    - df_recommend: Initial set of suggestions.
    - df_model_current: data on Sales.
    - agg_cols: Filtering/sorting data by.
    - agg_var: The main detail we're using to sort the suggestions.
    - agg: The specific value of the detail we're interested in.
    - segment: The group of dealers we're focusing on.

    Returns:
    A ready-to-use list of suggestions for each dealer.
    """
    # keep a copy of recommendation quartile with all models
    df_recommend_raw = df_recommend.copy(deep=True)

    # filter the recommendation quartile to the model and segment
    df_recommend = df_recommend[
        (df_recommend[agg_var] == agg)
        & (df_recommend[param_dict["cluster_col"]] == segment)
    ].copy()
    df_recommend["no_model_history_quartile_segment"] = "No"

    # check if quartile above has no sales, resulting no rec rules
    if df_recommend.empty:
        # log empty rec df (segment quartile above had no sales) in BPI run summary files
        bpi_log_grain_no_rules_df_updated = log_sg_q_no_sales(
            bpi_log_grain_no_rules_df_updated, df_model_current, aggregation_level
        )

        # recommend based on weighted IR for all other models within same series, segmentation
        df_recommend = recommend_quartile_above_no_sales(
            df_recommend_raw, agg, segment, aggregation_level, param_dict
        )
        df_recommend["no_model_history_quartile_segment"] = "Yes"

        # if still no recommendation after recommend_quartile_above_no_sales,
        # log the dealer-grain combo and mark rules_created = 'No'
        if df_recommend.empty:
            log_sg_q_no_sales_no_rules(
                bpi_log_grain_no_rules_df_updated, df_model_current, aggregation_level
            )

            # if we are at a quartile where install rates don't change
    if "dealer_number_latest" in df_recommend.columns:
        # make sure to remove in recommended df
        # the dealers that is not ordering the model in current month
        df_recommend = df_recommend[
            df_recommend["dealer_number_latest"].isin(
                df_model_current.dealer_number_latest.unique()
            )
        ]
        df_recommend["business_month"] = df_model_current.business_month.unique()[0]
        df_recommend_dealer_added = (
            df_recommend[
                agg_cols
                + [
                    "accessory_code",
                    "dealer_number_latest",
                    "quartile",
                    "correlation_category",
                    "correl",
                    "business_month",
                    "rec_retail_installation_rate",
                    "no_model_history_quartile_segment",
                ]
            ]
            .drop_duplicates()
            .copy()
        )

    # no recommendations at all
    elif df_recommend.empty:
        df_recommend_dealer_added = pd.DataFrame()
    else:
        # add all dealers that have interaction (stock/sale, etc) with this model in this month
        # into recommendation rules
        df_recommend_dealer_added = pd.merge(
            df_recommend[
                agg_cols
                + [
                    "accessory_code",
                    "rec_retail_installation_rate",
                    "no_model_history_quartile_segment",
                ]
            ].drop_duplicates(),
            df_model_current[
                agg_cols
                + [
                    "dealer_number_latest",
                    "quartile",
                    "correlation_category",
                    "correl",
                    "business_month",
                    "no_quartile_new_grain",
                ]
            ].drop_duplicates(),
            on=agg_cols,
            how="left",
            validate="many_to_many",
        )
    return df_recommend_dealer_added


def aggregate_recommendation_data(
    df_getsudo_data: pd.DataFrame,
    df_rules_high_quartile_dealer_lvl: pd.DataFrame,
    df_rules_lowest_quartile_dealer_lvl: pd.DataFrame,
    df_rules_no_quartile_dealer_lvl: pd.DataFrame,
    df_rules_high_quartile: pd.DataFrame,
    df_rules_lowest_quartile: pd.DataFrame,
    df_rules_low_quartile: pd.DataFrame,
    df_rules_medium_quartile: pd.DataFrame,
    q4_booster: float,
    q1_reducer: float,
    bpi_log_grain_no_rules_df_updated: pd.DataFrame,
    agg_cols: list,
    agg_var: str,
    aggregation_level: str,
    rule_mode: str,
    param_dict: dict,
    segments: list,
    correlations: list,
    quartile_classes: list,
) -> pd.DataFrame:
    df_rules_result_final = pd.DataFrame()

    # loop through each cluster in the new month model level sales data
    for segment in segments:
        df_model_current_cluster = df_getsudo_data[
            df_getsudo_data[param_dict["cluster_col"]] == segment
        ].copy()

        # -----------Same as df_get_sudo_data
        # within a segment, for each model, create quartiles based on MSRP
        for agg in df_model_current_cluster[agg_var].unique():
            df_model_current_cluster_model = df_model_current_cluster[
                df_model_current_cluster[agg_var] == agg
            ]

            # -----------Same as df_get_sudo_data
            # loop through the correlation for each model-cluster pair
            for correlation in correlations:
                df_model_current_cluster_model_corr = df_model_current_cluster_model[
                    df_model_current_cluster_model["correlation_category"]
                    == correlation
                ]

                # loop through each quartile's dealers
                for quartile_class in quartile_classes:
                    df_model_current = df_model_current_cluster_model_corr[
                        df_model_current_cluster_model_corr["quartile"]
                        == quartile_class
                    ]

                    if df_model_current.shape[0] == 0:
                        continue

                    df_recommend = get_recommendations_dataframe(
                        rule_mode,
                        correlation,
                        quartile_class,
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

                    df_recommend_dealer_added = handle_dealer_data(
                        bpi_log_grain_no_rules_df_updated,
                        aggregation_level,
                        param_dict,
                        df_recommend,
                        df_model_current,
                        agg_cols,
                        agg_var,
                        agg,
                        segment,
                    )

                    if not df_recommend_dealer_added.empty:
                        df_rules_result_final = pd.concat(
                            [df_rules_result_final, df_recommend_dealer_added],
                            ignore_index=True,
                        )

    return df_rules_result_final


def apply_rules(
    df_getsudo_data: pd.DataFrame,
    df_rules_lowest_quartile: pd.DataFrame,
    df_rules_low_quartile: pd.DataFrame,
    df_rules_medium_quartile: pd.DataFrame,
    df_rules_high_quartile: pd.DataFrame,
    df_rules_high_quartile_dealer_lvl: pd.DataFrame,
    df_rules_lowest_quartile_dealer_lvl: pd.DataFrame,
    df_rules_no_quartile_dealer_lvl: pd.DataFrame,
    bpi_log_grain_no_rules_df: pd.DataFrame,
    aggregation_level: str,
    param_dict: Dict,
    brand: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Recommend install rate for each ppo based on segment, grain, correlation and quartile for each dealer.

    Args:
        df_getsudo_data (pd.DataFrame): new month current allocation
        df_rules_lowest_quartile (pd.DataFrame): lowest quartile rules
        df_rules_low_quartile (pd.DataFrame): low quartile rules
        df_rules_medium_quartile (pd.DataFrame): medium quartile rules
        df_rules_high_quartile (pd.DataFrame): high quartile rules
        df_rules_high_quartile_dealer_lvl (pd.DataFrame): high quartile rules at dealer level
        df_rules_lowest_quartile_dealer_lvl (pd.DataFrame): lowest quartile rules at dealer level
        df_rules_no_quartile_dealer_lvl (pd.DataFrame): no quartile rules at dealer level
        bpi_log_grain_no_rules_df (pd.DataFrame): BPI log for grain no rules
        aggregation_level (str): the level of aggregation
        param_dict (Dict): parameter dictionary
        brand (str): brand of vehicle

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: PPO level sales with rules applied, BPI log for grain no rules, with new edge cases logged
    """

    # Set up configuration settings
    rule_mode, q4_booster, q1_reducer, agg_var = setup_config(
        aggregation_level, param_dict
    )

    # Initialize an empty DataFrame for the final results and a copy of the BPI log DataFrame
    df_rules_result_final = pd.DataFrame()
    bpi_log_grain_no_rules_df_updated = bpi_log_grain_no_rules_df.copy(deep=True)

    # Determine aggregation columns based on aggregation level
    agg_cols = [param_dict["cluster_col"], "series_name"]
    if aggregation_level in ("model", "grade"):
        agg_cols += ["model_number"]

    # Process data for each unique segments, aggregation variable, correlation, and quartile class

    segments = df_getsudo_data[param_dict["cluster_col"]].unique()
    correlations = ["Positive", "Negative", "None"]
    quartile_classes = ["lowest", "low", "medium", "high", INSUFFICIENT_GRAIN_DEALERS]
    df_rules_result_final = aggregate_recommendation_data(
        df_getsudo_data,
        df_rules_high_quartile_dealer_lvl,
        df_rules_lowest_quartile_dealer_lvl,
        df_rules_no_quartile_dealer_lvl,
        df_rules_high_quartile,
        df_rules_lowest_quartile,
        df_rules_low_quartile,
        df_rules_medium_quartile,
        q4_booster,
        q1_reducer,
        bpi_log_grain_no_rules_df_updated,
        agg_cols,
        agg_var,
        aggregation_level,
        rule_mode,
        param_dict,
        segments,
        correlations,
        quartile_classes,
    )

    # Ensure the final DataFrame contains the necessary columns
    if "no_quartile_new_grain" not in df_rules_result_final.columns:
        df_rules_result_final["no_quartile_new_grain"] = None

    bpi_log_grain_no_rules_df_updated["brand"] = brand

    return df_rules_result_final, bpi_log_grain_no_rules_df_updated


def recommendation_series_month(
    series: str,
    target_month_min: int,
    io_dict: dict,
    base_parameters: dict,
    data_catalog: dict,
    run_version: str,
    df_region_dealer: pd.DataFrame,
) -> None:
    aggregation_level = base_parameters["aggregation_level"][series]
    brand = base_parameters["brand"]
    io_dict["series"] = series
    io_dict["target_month_min"] = target_month_min

    logger.info(f"{series} {target_month_min} part 3 started")

    ######################################################################################################
    # read data
    ######################################################################################################
    getsudo_dealer_grain_corr_quartile_df = load_data(
        data_catalog["dealer_grain_corr_quartile_df"], param_dict=io_dict
    )
    df_rules_high_quartile_dealer_lvl = load_data(
        data_catalog["rules_high_q_dealer"], param_dict=io_dict
    )
    df_rules_lowest_quartile_dealer_lvl = load_data(
        data_catalog["rules_lowest_q_dealer"], param_dict=io_dict
    )
    df_rules_no_quartile_dealer_lvl = load_data(
        data_catalog["rules_no_q_dealer"], param_dict=io_dict
    )
    df_rules_lowest_quartile = load_data(
        data_catalog["rules_lowest_q"], param_dict=io_dict
    )
    df_rules_low_quartile = load_data(data_catalog["rules_low_q"], param_dict=io_dict)
    df_rules_medium_quartile = load_data(
        data_catalog["rules_medium_q"], param_dict=io_dict
    )
    df_rules_high_quartile = load_data(data_catalog["rules_high_q"], param_dict=io_dict)
    df_sale_grain_historical = load_data(
        data_catalog["historical_grain_qty"], param_dict=io_dict
    )

    # load current bpi grain no rules log
    bpi_log_grain_no_rules_df = load_data(
        data_catalog["bpi_log_grain_no_rules"],
        param_dict={"run_version": run_version, "brand": brand},
    )

    ######################################################################################################
    # Apply recommended rules to allocation
    ######################################################################################################
    logger.info(f"Applying rules for {series} {target_month_min}")
    df_rules_result_final, bpi_log_grain_no_rules_df_updated = apply_rules(
        getsudo_dealer_grain_corr_quartile_df,
        df_rules_lowest_quartile,
        df_rules_low_quartile,
        df_rules_medium_quartile,
        df_rules_high_quartile,
        df_rules_high_quartile_dealer_lvl,
        df_rules_lowest_quartile_dealer_lvl,
        df_rules_no_quartile_dealer_lvl,
        bpi_log_grain_no_rules_df,
        aggregation_level,
        base_parameters,
        brand,
    )
    logger.info(f"Successfully applied rules for {series} {target_month_min}")
    logger.info(f"Aggregating rules for {series} {target_month_min}")
    df_rules_aggregated = aggregate_rules(
        df_rules_result_final,
        df_sale_grain_historical,
        df_region_dealer,
        aggregation_level,
        base_parameters,
        series,
        target_month_min,
    )
    logger.info(f"Successfully aggregated rules for {series} {target_month_min}")

    ######################################################################################################
    # save data
    ######################################################################################################
    save_data(
        data_catalog["rules_recommendation_dmaa"],
        df_rules_aggregated,
        param_dict=io_dict,
    )
    save_data(
        data_catalog["bpi_log_grain_no_rules"],
        bpi_log_grain_no_rules_df_updated,
        param_dict={
            "run_version": run_version,
            "brand": brand,
        },
    )
    logger.info(f"{series} {target_month_min} part 3 ended")


@timer_func
def run(base_parameters: Dict, data_catalog: Dict, run_version: str) -> None:
    """.
    Generate ppo recommendation for the latest month

    Args:
        base_parameters (Dict): parameters for run
        data_catalog (Dict): data catalog
        run_version (str): run version with date and random hash
    """
    logger.info(f"Run version: {run_version}")

    mode = base_parameters["mode"]
    brand = base_parameters["brand"]
    iteration_list = load_data(
        data_catalog["run_dict"],
        param_dict={"run_version": run_version, "brand": brand},
    ).values

    logger.info(f"Running mode: {mode}, iteration_list: {iteration_list}")
    io_dict = {
        "mode": mode,
        "run_version": run_version,
    }
    df_region_dealer = load_data(data_catalog["region_dealer"], param_dict=io_dict)

    is_parallel_process = base_parameters["parallel_process"]
    if is_parallel_process:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for series, target_month_min in iteration_list:
                futures.append(
                    executor.submit(
                        recommendation_series_month,
                        series,
                        target_month_min,
                        io_dict.copy(),
                        base_parameters,
                        data_catalog,
                        run_version,
                        df_region_dealer,
                    )
                )
            wait(futures)
    else:
        for series, target_month_min in iteration_list:
            recommendation_series_month(
                series,
                target_month_min,
                io_dict.copy(),
                base_parameters,
                data_catalog,
                run_version,
                df_region_dealer,
            )
