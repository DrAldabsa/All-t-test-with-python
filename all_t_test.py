import pandas as pd
import numpy as np
from scipy import stats
import warnings

#C:\Users\Cyber\Desktop\مركز الدبسا للاستشارات البحثية والاحصائية\pythonProject\.venv\wheat_yield_ALL_ttest_test_data.csv
# ==========================================================
# Helper function: IQR outlier detection
# ==========================================================

def detect_outliers_iqr(values):

    Q1 = np.percentile(values, 25)
    Q3 = np.percentile(values, 75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_mask = (
        (values < lower_bound) |
        (values > upper_bound)
    )

    outliers = values[outlier_mask]

    return (
        outlier_mask,
        outliers,
        lower_bound,
        upper_bound
    )

######
def check_column(test_type:str, column, df, column1, column2):
    if test_type == "one-sample":
        if column not in df.columns:
            raise ValueError(
                f"Column '{column}' was not found.\n"
                f"Available columns:\n{df.columns.tolist()}"
            )
    elif test_type == "independent" or test_type == "paired":
        for col in [column1, column2]:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' was not found.\n"
                    f"Available columns:\n"
                    f"{df.columns.tolist()}"
                )
########
def convert_to_numeric(test_type, df, column, column1, column2):

    if test_type == "one-sample":

        original_values = df[column].copy()

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        conversion_errors = (
            original_values.notna() &
            df[column].isna()
        ).sum()

        if conversion_errors > 0:
            warnings.warn(
                f"{conversion_errors} non-numeric values "
                f"in '{column}' were converted to NaN."
            )

    elif test_type == "independent" or test_type == "paired":

        for column_name in (column1, column2):

            original_values = df[column_name].copy()

            df[column_name] = pd.to_numeric(
                df[column_name],
                errors="coerce"
            )

            conversion_errors = (
                original_values.notna() &
                df[column_name].isna()
            ).sum()

            if conversion_errors > 0:
                warnings.warn(
                    f"{conversion_errors} non-numeric values "
                    f"in '{column_name}' were converted to NaN."
                )

#####
def missing_value(test_type ,df, column, min_n, column1, column2):
    if test_type == "one-sample":
        missing_count = df[column].isna().sum()
        total_rows = len(df)
        missing_percentage = (
            missing_count / total_rows * 100
            if total_rows > 0 else 0
        )

        if missing_count > 0:
            warnings.warn(
                f"{missing_count} missing values detected "
                f"({missing_percentage:.2f}%). "
                "They will be excluded from the analysis."
            )

        # Remove missing values
        values = df[column].dropna().to_numpy()
        n_before_outliers = len(values)
        if n_before_outliers < min_n:
            raise ValueError(
                f"Only {n_before_outliers} valid observations "
                f"remain. Minimum required is {min_n}."
            )
        return (values, n_before_outliers, missing_count, missing_percentage, total_rows)

    elif test_type == "independent" or test_type == "paired":
        group1 = df[column1].dropna().to_numpy()
        group2 = df[column2].dropna().to_numpy()
        missing_count1 = df[column1].isna().sum()
        missing_count2 = df[column2].isna().sum()

        # ------------------------------------------------------
        # Initial sample sizes
        # ------------------------------------------------------
        n1_before = len(group1)
        n2_before = len(group2)

        if n1_before < min_n:
            raise ValueError(
                f"'{column1}' has only "
                f"{n1_before} valid observations. "
                f"Minimum required is {min_n}."
            )

        if n2_before < min_n:
            raise ValueError(
                f"'{column2}' has only "
                f"{n2_before} valid observations. "
                f"Minimum required is {min_n}."
            )
    ####
        paired_df = df[[column1, column2]].dropna()
        missing_pairs = (len(df) - len(paired_df))
        if missing_pairs > 0:
            warnings.warn(
                f"{missing_pairs} incomplete pair(s) "
                "were excluded from the analysis."
            )

        if len(paired_df) < min_n:
            raise ValueError(
                f"Only {len(paired_df)} complete pairs "
                f"remain. Minimum required is {min_n}."
            )
        group1_p = paired_df[column1].to_numpy()
        group2_p = paired_df[column2].to_numpy()
        return (group1, group2, missing_count1, missing_count2, n1_before, n2_before,
                paired_df, missing_pairs, group1_p, group2_p )


# ==========================================================
# Main t-test function
# ==========================================================

def ttest_analysis(
    file_path: str,
    test_type: str,
    column: str = None,
    column1: str = None,
    column2: str = None,
    population_mean: float = None,
    alpha: float = 0.05,
    confidence_level: float = 0.95,
    alternative: str = "two-sided",
    outlier_method: str = "IQR",
    remove_outliers: bool = True,
    variance_test: str = "welch",
    min_n: int = 3
):

    """
    Perform t-test analysis.

    Supported tests:
        - one-sample
        - independent
        - paired

    Parameters
    ----------
    file_path : str
        Path to CSV file.

    test_type : str
        Type of t-test:
        "one-sample",
        "independent",
        "paired"

    column : str
        Column for One-Sample t-test.

    column1 : str
        First column for Independent or Paired t-test.

    column2 : str
        Second column for Independent or Paired t-test.

    population_mean : float
        Reference mean for One-Sample t-test.

    alpha : float
        Significance level.

    confidence_level : float
        Confidence level.

    alternative : str
        "two-sided", "greater", or "less".

    outlier_method : str
        Currently supports "IQR".

    remove_outliers : bool
        Whether to remove detected outliers.

    variance_test : str
        For Independent t-test:
        "student" = equal variances
        "welch"   = unequal variances

    min_n : int
        Minimum sample size.

    Returns
    -------
    dict
        Statistical analysis results.
    """

    # ==========================================================
    # 1. Validate general inputs
    # ==========================================================

    if not 0 < alpha < 1:
        raise ValueError(
            "alpha must be between 0 and 1."
        )

    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level must be between 0 and 1."
        )

    if alternative not in [
        "two-sided",
        "greater",
        "less"
    ]:
        raise ValueError(
            "alternative must be "
            "'two-sided', 'greater', or 'less'."
        )

    if outlier_method not in ["IQR"]:
        raise ValueError(
            "Currently only 'IQR' is supported."
        )

    if test_type not in [
        "one-sample",
        "independent",
        "paired"
    ]:
        raise ValueError(
            "test_type must be "
            "'one-sample', 'independent', or 'paired'."
        )

    if test_type == "independent":

        if variance_test not in [
            "student",
            "welch"
        ]:
            raise ValueError(
                "variance_test must be "
                "'student' or 'welch'."
            )

    # ==========================================================
    # 2. Read CSV
    # ==========================================================

    try:

        df = pd.read_csv(
            file_path,
            na_values=[
                "",
                "NA",
                "N/A",
                "NaN",
                "NULL",
                "null",
                "-",
                "missing"
            ]
        )

    except FileNotFoundError:

        raise FileNotFoundError(
            f"File not found:\n{file_path}"
        )

    except Exception as e:

        raise RuntimeError(
            f"Error reading CSV file: {e}"
        )

    # ==========================================================
    # 3. ONE-SAMPLE T-TEST
    # ==========================================================

    if test_type == "one-sample":

        # ------------------------------------------------------
        # Check column
        # ------------------------------------------------------
        check_column(test_type, column,df, column1, column2)

        # ------------------------------------------------------
        # Convert column to numeric
        # ------------------------------------------------------

        convert_to_numeric(test_type,df, column, column1, column2)

        # ------------------------------------------------------
        # Missing values
        # ------------------------------------------------------

        (values, n_before_outliers, missing_count, missing_percentage, total_rows) = missing_value(df, column, min_n)

        # ------------------------------------------------------
        # Detect outliers
        # ------------------------------------------------------

        outliers = np.array([])

        lower_bound = np.nan
        upper_bound = np.nan

        if outlier_method == "IQR":

            (
                outlier_mask,
                outliers,
                lower_bound,
                upper_bound
            ) = detect_outliers_iqr(values)

        # ------------------------------------------------------
        # Warn about outliers
        # ------------------------------------------------------

        outlier_count = len(outliers)

        outlier_percentage = (
            outlier_count / len(values) * 100
        )

        if outlier_count > 0:

            warnings.warn(
                f"{outlier_count} outlier(s) detected "
                f"({outlier_percentage:.2f}%)."
            )

        # ------------------------------------------------------
        # Remove or retain outliers
        # ------------------------------------------------------

        if remove_outliers and outlier_count > 0:

            values_clean = values[
                ~outlier_mask
            ]

            warnings.warn(
                "Outliers were removed before "
                "performing the t-test."
            )

        else:

            values_clean = values

            if outlier_count > 0:

                warnings.warn(
                    "Outliers were detected but retained "
                    "because remove_outliers=False."
                )

        # ------------------------------------------------------
        # Final sample size
        # ------------------------------------------------------

        n = len(values_clean)

        if n < min_n:

            raise ValueError(
                f"Only {n} observations remain after cleaning. "
                f"Minimum required is {min_n}."
            )

        # ------------------------------------------------------
        # Descriptive statistics
        # ------------------------------------------------------

        mean = np.mean(values_clean)

        sd = np.std(
            values_clean,
            ddof=1
        )

        median = np.median(values_clean)

        minimum = np.min(values_clean)

        maximum = np.max(values_clean)

        standard_error = sd / np.sqrt(n)

        # ------------------------------------------------------
        # Manual t statistic
        # ------------------------------------------------------

        t_manual = (
            (mean - population_mean)
            / standard_error
        )

        df_degrees = n - 1

        # ------------------------------------------------------
        # SciPy One-Sample t-test
        # ------------------------------------------------------

        result = stats.ttest_1samp(
            values_clean,
            popmean=population_mean,
            alternative=alternative
        )

        t_statistic = float(
            result.statistic
        )

        p_value = float(
            result.pvalue
        )

        # ------------------------------------------------------
        # Confidence interval
        # ------------------------------------------------------

        ci = result.confidence_interval(
            confidence_level=confidence_level
        )

        ci_lower = float(ci.low)

        ci_upper = float(ci.high)

        # ------------------------------------------------------
        # Cohen's d
        # ------------------------------------------------------

        cohens_d = (
            (mean - population_mean)
            / sd
        )

        # ------------------------------------------------------
        # Mean difference
        # ------------------------------------------------------

        mean_difference = (
            mean - population_mean
        )

        # ------------------------------------------------------
        # Decision
        # ------------------------------------------------------

        if p_value < alpha:

            decision = "Reject H0"

            significance = (
                "Statistically significant"
            )

        else:

            decision = "Fail to reject H0"

            significance = (
                "Not statistically significant"
            )

        # ------------------------------------------------------
        # Effect size interpretation
        # ------------------------------------------------------

        abs_d = abs(cohens_d)

        if abs_d < 0.20:

            effect_interpretation = "Negligible"

        elif abs_d < 0.50:

            effect_interpretation = "Small"

        elif abs_d < 0.80:

            effect_interpretation = "Medium"

        else:

            effect_interpretation = "Large"

        # ------------------------------------------------------
        # Return One-Sample results
        # ------------------------------------------------------

        results = {

            "test_type": "One-Sample t-test",

            # Data information
            "file": file_path,
            "variable": column,

            # Sample information
            "n_original": total_rows,
            "missing_count": int(missing_count),
            "missing_percentage": missing_percentage,

            "n_before_outliers":
                n_before_outliers,

            "outlier_count":
                outlier_count,

            "outlier_percentage":
                outlier_percentage,

            "outliers":
                outliers.tolist(),

            "n_final": n,

            # Descriptive statistics
            "mean": mean,
            "median": median,
            "sd": sd,
            "se": standard_error,
            "minimum": minimum,
            "maximum": maximum,

            # Hypothesis
            "population_mean":
                population_mean,

            # Test
            "t_manual":
                t_manual,

            "t_statistic":
                t_statistic,

            "df":
                df_degrees,

            "p_value":
                p_value,

            # CI
            "confidence_level":
                confidence_level,

            "ci_lower":
                ci_lower,

            "ci_upper":
                ci_upper,

            # Effect
            "mean_difference":
                mean_difference,

            "cohens_d":
                cohens_d,

            "effect_interpretation":
                effect_interpretation,

            # Decision
            "alpha":
                alpha,

            "decision":
                decision,

            "significance":
                significance
        }

        return results

    # ==========================================================
    # 4. INDEPENDENT T-TEST
    # ==========================================================

    elif test_type == "independent":


        # ------------------------------------------------------
        # Check columns
        # ------------------------------------------------------
        check_column(test_type, column, df, column1, column2)


        # ------------------------------------------------------
        # Convert columns to numeric
        # ------------------------------------------------------
        convert_to_numeric(test_type,df, column, column1, column2)


        # ------------------------------------------------------
        # Missing values & Initial sample sizes
        # ------------------------------------------------------
        (group1, group2, missing_count1, missing_count2, n1_before, n2_before,_,_,_,_) = missing_value(test_type ,df, column, min_n, column1, column2)


        # ------------------------------------------------------
        # Detect outliers - Group 1
        # ------------------------------------------------------

        (
            outlier_mask1,
            outliers1,
            lower_bound1,
            upper_bound1
        ) = detect_outliers_iqr(group1)

        # ------------------------------------------------------
        # Detect outliers - Group 2
        # ------------------------------------------------------

        (
            outlier_mask2,
            outliers2,
            lower_bound2,
            upper_bound2
        ) = detect_outliers_iqr(group2)

        # ------------------------------------------------------
        # Warn about outliers
        # ------------------------------------------------------

        if len(outliers1) > 0:

            warnings.warn(
                f"{len(outliers1)} outlier(s) detected "
                f"in '{column1}'."
            )

        if len(outliers2) > 0:

            warnings.warn(
                f"{len(outliers2)} outlier(s) detected "
                f"in '{column2}'."
            )

        # ------------------------------------------------------
        # Remove or retain outliers
        # ------------------------------------------------------

        if remove_outliers:

            group1_clean = group1[
                ~outlier_mask1
            ]

            group2_clean = group2[
                ~outlier_mask2
            ]

            if len(outliers1) > 0 or len(outliers2) > 0:

                warnings.warn(
                    "Outliers were removed before "
                    "performing the Independent t-test."
                )

        else:

            group1_clean = group1
            group2_clean = group2

        # ------------------------------------------------------
        # Final sample sizes
        # ------------------------------------------------------

        n1 = len(group1_clean)
        n2 = len(group2_clean)

        if n1 < min_n:

            raise ValueError(
                f"Only {n1} observations remain in "
                f"'{column1}' after cleaning."
            )

        if n2 < min_n:

            raise ValueError(
                f"Only {n2} observations remain in "
                f"'{column2}' after cleaning."
            )

        # ------------------------------------------------------
        # Descriptive statistics
        # ------------------------------------------------------

        mean1 = np.mean(group1_clean)
        mean2 = np.mean(group2_clean)

        sd1 = np.std(
            group1_clean,
            ddof=1
        )

        sd2 = np.std(
            group2_clean,
            ddof=1
        )

        median1 = np.median(group1_clean)
        median2 = np.median(group2_clean)

        minimum1 = np.min(group1_clean)
        minimum2 = np.min(group2_clean)

        maximum1 = np.max(group1_clean)
        maximum2 = np.max(group2_clean)

        # ------------------------------------------------------
        # Mean difference
        # ------------------------------------------------------

        mean_difference = (
            mean1 - mean2
        )

        # ------------------------------------------------------
        # Independent t-test
        # ------------------------------------------------------

        if variance_test == "student":

            result = stats.ttest_ind(
                group1_clean,
                group2_clean,
                equal_var=True,
                alternative=alternative
            )

            test_name = (
                "Independent Samples "
                "Student's t-test"
            )

        else:

            result = stats.ttest_ind(
                group1_clean,
                group2_clean,
                equal_var=False,
                alternative=alternative
            )

            test_name = (
                "Independent Samples "
                "Welch's t-test"
            )

        t_statistic = float(
            result.statistic
        )

        p_value = float(
            result.pvalue
        )

        df_degrees = float(
            result.df
        )

        # ------------------------------------------------------
        # Confidence interval
        # ------------------------------------------------------

        ci = result.confidence_interval(
            confidence_level=confidence_level
        )

        ci_lower = float(ci.low)

        ci_upper = float(ci.high)

        # ------------------------------------------------------
        # Cohen's d
        # ------------------------------------------------------

        if variance_test == "student":

            pooled_sd = np.sqrt(
                (
                    (n1 - 1) * sd1**2 +
                    (n2 - 1) * sd2**2
                )
                /
                (n1 + n2 - 2)
            )

            cohens_d = (
                (mean1 - mean2)
                / pooled_sd
            )

        else:

            # For Welch's t-test,
            # use the average within-group variance
            average_sd = np.sqrt(
                (sd1**2 + sd2**2) / 2
            )

            cohens_d = (
                (mean1 - mean2)
                / average_sd
            )

        # ------------------------------------------------------
        # Effect size interpretation
        # ------------------------------------------------------

        abs_d = abs(cohens_d)

        if abs_d < 0.20:

            effect_interpretation = "Negligible"

        elif abs_d < 0.50:

            effect_interpretation = "Small"

        elif abs_d < 0.80:

            effect_interpretation = "Medium"

        else:

            effect_interpretation = "Large"

        # ------------------------------------------------------
        # Decision
        # ------------------------------------------------------

        if p_value < alpha:

            decision = "Reject H0"

            significance = (
                "Statistically significant"
            )

        else:

            decision = "Fail to reject H0"

            significance = (
                "Not statistically significant"
            )

        # ------------------------------------------------------
        # Return Independent results
        # ------------------------------------------------------

        results = {

            "test_type":
                test_name,

            "file":
                file_path,

            "group1":
                column1,

            "group2":
                column2,

            "variance_test":
                variance_test,

            # Sample information
            "n_original_group1":
                n1_before,

            "n_original_group2":
                n2_before,

            "missing_count_group1":
                int(missing_count1),

            "missing_count_group2":
                int(missing_count2),

            "n_before_outliers_group1":
                n1_before,

            "n_before_outliers_group2":
                n2_before,

            "outlier_count_group1":
                len(outliers1),

            "outlier_count_group2":
                len(outliers2),

            "outliers_group1":
                outliers1.tolist(),

            "outliers_group2":
                outliers2.tolist(),

            "n_final_group1":
                n1,

            "n_final_group2":
                n2,

            # Descriptive statistics
            "mean_group1":
                mean1,

            "mean_group2":
                mean2,

            "median_group1":
                median1,

            "median_group2":
                median2,

            "sd_group1":
                sd1,

            "sd_group2":
                sd2,

            "minimum_group1":
                minimum1,

            "minimum_group2":
                minimum2,

            "maximum_group1":
                maximum1,

            "maximum_group2":
                maximum2,

            # Test
            "t_statistic":
                t_statistic,

            "df":
                df_degrees,

            "p_value":
                p_value,

            # CI
            "confidence_level":
                confidence_level,

            "ci_lower":
                ci_lower,

            "ci_upper":
                ci_upper,

            # Effect
            "mean_difference":
                mean_difference,

            "cohens_d":
                cohens_d,

            "effect_interpretation":
                effect_interpretation,

            # Decision
            "alpha":
                alpha,

            "decision":
                decision,

            "significance":
                significance
        }

        return results

    # ==========================================================
    # 5. PAIRED T-TEST
    # ==========================================================

    elif test_type == "paired":

        # ------------------------------------------------------
        # Check columns
        # ------------------------------------------------------
        check_column(test_type, column, df, coulmn1, column2)


        # ------------------------------------------------------
        # Convert to numeric
        # ------------------------------------------------------

        convert_to_numeric(test_type, df, column, column1, column2)

        # ------------------------------------------------------
        # Keep only complete pairs
        # ------------------------------------------------------
        (_, _, _, _, _, _, paired_df, missing_pairs, group1_p, group2_p) = missing_value(test_type ,df, column, min_n, column1, column2)


        # ------------------------------------------------------
        # Calculate paired differences
        # ------------------------------------------------------

        differences = (
            group1_p - group2_p
        )

        # ------------------------------------------------------
        # Detect outliers on differences
        # ------------------------------------------------------

        (
            outlier_mask,
            outliers,
            lower_bound,
            upper_bound
        ) = detect_outliers_iqr(
            differences
        )

        if len(outliers) > 0:

            warnings.warn(
                f"{len(outliers)} outlier(s) detected "
                "in the paired differences."
            )

        # ------------------------------------------------------
        # Remove or retain outliers
        # ------------------------------------------------------

        if remove_outliers:

            group1_clean = group1_p[
                ~outlier_mask
            ]

            group2_clean = group2_p[
                ~outlier_mask
            ]

            differences_clean = differences[
                ~outlier_mask
            ]

            if len(outliers) > 0:

                warnings.warn(
                    "Outlier pairs were removed before "
                    "performing the Paired t-test."
                )

        else:

            group1_clean = group1_p
            group2_clean = group2_p
            differences_clean = differences

        # ------------------------------------------------------
        # Final sample size
        # ------------------------------------------------------

        n = len(differences_clean)

        if n < min_n:

            raise ValueError(
                f"Only {n} paired observations remain "
                f"after cleaning."
            )

        # ------------------------------------------------------
        # Descriptive statistics
        # ------------------------------------------------------

        mean1 = np.mean(group1_clean)
        mean2 = np.mean(group2_clean)

        sd1 = np.std(
            group1_clean,
            ddof=1
        )

        sd2 = np.std(
            group2_clean,
            ddof=1
        )

        median1 = np.median(group1_clean)
        median2 = np.median(group2_clean)

        minimum1 = np.min(group1_clean)
        minimum2 = np.min(group2_clean)

        maximum1 = np.max(group1_clean)
        maximum2 = np.max(group2_clean)

        # ------------------------------------------------------
        # Difference statistics
        # ------------------------------------------------------

        mean_difference = np.mean(
            differences_clean
        )

        sd_difference = np.std(
            differences_clean,
            ddof=1
        )

        se_difference = (
            sd_difference /
            np.sqrt(n)
        )

        # ------------------------------------------------------
        # Manual t statistic
        # ------------------------------------------------------

        t_manual = (
            mean_difference /
            se_difference
        )

        df_degrees = n - 1

        # ------------------------------------------------------
        # SciPy Paired t-test
        # ------------------------------------------------------

        result = stats.ttest_rel(
            group1_clean,
            group2_clean,
            alternative=alternative
        )

        t_statistic = float(
            result.statistic
        )

        p_value = float(
            result.pvalue
        )

        # ------------------------------------------------------
        # Confidence interval
        # ------------------------------------------------------

        ci = result.confidence_interval(
            confidence_level=confidence_level
        )

        ci_lower = float(ci.low)

        ci_upper = float(ci.high)

        # ------------------------------------------------------
        # Cohen's d for paired data
        # ------------------------------------------------------

        cohens_d = (
            mean_difference /
            sd_difference
        )

        # ------------------------------------------------------
        # Effect size interpretation
        # ------------------------------------------------------

        abs_d = abs(cohens_d)

        if abs_d < 0.20:

            effect_interpretation = "Negligible"

        elif abs_d < 0.50:

            effect_interpretation = "Small"

        elif abs_d < 0.80:

            effect_interpretation = "Medium"

        else:

            effect_interpretation = "Large"

        # ------------------------------------------------------
        # Decision
        # ------------------------------------------------------

        if p_value < alpha:

            decision = "Reject H0"

            significance = (
                "Statistically significant"
            )

        else:

            decision = "Fail to reject H0"

            significance = (
                "Not statistically significant"
            )

        # ------------------------------------------------------
        # Return Paired results
        # ------------------------------------------------------

        results = {

            "test_type":
                "Paired Samples t-test",

            "file":
                file_path,

            "variable1":
                column1,

            "variable2":
                column2,

            # Sample information
            "n_original_pairs":
                len(df),

            "incomplete_pairs":
                int(missing_pairs),

            "n_before_outliers":
                len(paired_df),

            "outlier_count":
                len(outliers),

            "outlier_percentage":
                (
                    len(outliers) /
                    len(differences) * 100
                ),

            "outliers":
                outliers.tolist(),

            "n_final":
                n,

            # Descriptive statistics
            "mean_variable1":
                mean1,

            "mean_variable2":
                mean2,

            "median_variable1":
                median1,

            "median_variable2":
                median2,

            "sd_variable1":
                sd1,

            "sd_variable2":
                sd2,

            "minimum_variable1":
                minimum1,

            "minimum_variable2":
                minimum2,

            "maximum_variable1":
                maximum1,

            "maximum_variable2":
                maximum2,

            # Difference
            "mean_difference":
                mean_difference,

            "sd_difference":
                sd_difference,

            "se_difference":
                se_difference,

            # Test
            "t_manual":
                t_manual,

            "t_statistic":
                t_statistic,

            "df":
                df_degrees,

            "p_value":
                p_value,

            # CI
            "confidence_level":
                confidence_level,

            "ci_lower":
                ci_lower,

            "ci_upper":
                ci_upper,

            # Effect
            "cohens_d":
                cohens_d,

            "effect_interpretation":
                effect_interpretation,

            # Decision
            "alpha":
                alpha,

            "decision":
                decision,

            "significance":
                significance
        }

        return results


# ==========================================================
# USER INPUT
# ==========================================================

print("=" * 60)
print("T-TEST ANALYSIS")
print("=" * 60)

print("\nSelect t-test type:")

print("1. One-Sample t-test")
print("2. Independent Samples t-test")
print("3. Paired Samples t-test")

choice = input(
    "\nEnter your choice (1-3): "
).strip()


# ==========================================================
# COMMON INPUT
# ==========================================================

file_path = input(
    "\nEnter CSV file path: "
).strip()

alpha = float(
    input(
        "Enter alpha (default 0.05): "
    ) or 0.05
)

confidence_level = float(
    input(
        "Enter confidence level "
        "(default 0.95): "
    ) or 0.95
)

print("\nAlternative hypothesis:")
print("1. two-sided")
print("2. greater")
print("3. less")

alternative_choice = input(
    "Enter choice (1-3): "
).strip()

alternative_map = {
    "1": "two-sided",
    "2": "greater",
    "3": "less"
}

alternative = alternative_map.get(
    alternative_choice,
    "two-sided"
)

remove_outliers_input = input(
    "\nRemove outliers? (y/n, default y): "
).strip().lower()

remove_outliers = (
    False
    if remove_outliers_input == "n"
    else True
)


# ==========================================================
# ONE-SAMPLE
# ==========================================================

if choice == "1":

    column = input(
        "\nEnter column name: "
    ).strip()

    population_mean = float(
        input(
            "Enter population/reference mean: "
        )
    )

    results = ttest_analysis(
        file_path=file_path,
        test_type="one-sample",
        column=column,
        population_mean=population_mean,
        alpha=alpha,
        confidence_level=confidence_level,
        alternative=alternative,
        outlier_method="IQR",
        remove_outliers=remove_outliers
    )


# ==========================================================
# INDEPENDENT
# ==========================================================

elif choice == "2":

    column1 = input(
        "\nEnter first group column: "
    ).strip()

    column2 = input(
        "Enter second group column: "
    ).strip()

    print(
        "\nSelect variance assumption:"
    )

    print(
        "1. Student's t-test "
        "(Equal variances)"
    )

    print(
        "2. Welch's t-test "
        "(Unequal variances)"
    )

    variance_choice = input(
        "Enter choice (1-2): "
    ).strip()

    variance_test = (
        "student"
        if variance_choice == "1"
        else "welch"
    )

    results = ttest_analysis(
        file_path=file_path,
        test_type="independent",
        column1=column1,
        column2=column2,
        alpha=alpha,
        confidence_level=confidence_level,
        alternative=alternative,
        outlier_method="IQR",
        remove_outliers=remove_outliers,
        variance_test=variance_test
    )


# ==========================================================
# PAIRED
# ==========================================================

elif choice == "3":

    column1 = input(
        "\nEnter first paired column: "
    ).strip()

    column2 = input(
        "Enter second paired column: "
    ).strip()

    results = ttest_analysis(
        file_path=file_path,
        test_type="paired",
        column1=column1,
        column2=column2,
        alpha=alpha,
        confidence_level=confidence_level,
        alternative=alternative,
        outlier_method="IQR",
        remove_outliers=remove_outliers
    )


# ==========================================================
# INVALID CHOICE
# ==========================================================

else:

    raise ValueError(
        "Invalid choice. Please select 1, 2, or 3."
    )


# ==========================================================
# OUTPUT
# ==========================================================

print("\n")
print("=" * 60)
print("STATISTICAL RESULTS")
print("=" * 60)

for key, value in results.items():

    print(
        f"{key}: {value}"
    )