import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# from skopt import BayesSearchCV
import numpy as np
import joblib


def process_csv_files(directory):
    """
    Read data files and combine into single dataset.
    Clean and transform data.

    Parameters
    ----------
    directory : STR
        PATH OF FOLDER WITH EXTRACTED CSV DATA FILES.

    Returns
    -------
    combined_df : DATAFRAME
        CONSOLIDATED HOUSING DATA.

    """
    # Get all CSV files in the directory
    csv_files = [
        file for file in os.listdir(directory) if file.endswith(".csv")
    ]

    # Read and concatenate all CSV files
    dataframes = []
    for file in csv_files:
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)
        dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Display basic information about the dataset
    print(combined_df.info())

    # Check for missing values
    print(combined_df.isnull().sum())

    # Convert "flat_model" column to uppercase
    combined_df["flat_model"] = combined_df["flat_model"].str.upper()

    # Create "year" column from the "month" column
    combined_df["year"] = combined_df["month"].str[:4]

    # Overwrite "remaining_lease" column with the calculated value
    combined_df["remaining_lease"] = 99 - (
        combined_df["year"].astype(int) - combined_df["lease_commence_date"]
    )

    # Convert "month" column to date format
    combined_df["month"] = pd.to_datetime(
        combined_df["month"], format="%Y-%m", errors="coerce"
    )
    # New consolidated bins for "storey_range"
    consolidated_bins = [(1, 15), (16, 30), (31, 51)]

    # Apply rebinning
    combined_df["re_binned_range"] = combined_df["storey_range"].apply(
        lambda r: next(
            f"{bin_start} TO {bin_end}"
            for bin_start, bin_end in consolidated_bins
            if int(r.split(" TO ")[0]) >= bin_start
            and int(r.split(" TO ")[1]) <= bin_end
        )
    )
    # Drop unnecessary columns
    combined_df = combined_df.drop(
        ["storey_range", "lease_commence_date"], axis=1
    )

    # Save the cleaned dataframe to a new CSV file
    output_file = os.path.join(directory, "dataset.csv")
    combined_df.to_csv(output_file, index=False)
    return combined_df


def perform_eda(df):
    """
    Generate plots for exploratory data analysis.

    Parameters
    ----------
    df : DataFrame
        CONSOLIDATED HOUSING DATA.

    Returns
    -------
    None.

    """
    # Convert month and year to datetime
    df["month"] = pd.to_datetime(df["month"])
    df["year"] = df["month"].dt.year
    df["month"] = df["month"].dt.month

    # Plot average resale price per year
    average_price_per_year = (
        df.groupby("year")["resale_price"].mean().reset_index()
    )
    average_price_per_year_ftype = (
        df.groupby(["year", "flat_type"])["resale_price"].mean().reset_index()
    )
    average_price_per_year_town = (
        df.groupby(["year", "town"])["resale_price"].mean().reset_index()
    )

    plt.figure(figsize=(15, 6))
    sns.lineplot(
        x="year", y="resale_price", data=average_price_per_year, marker="o"
    )
    plt.title("Average Resale Price Per Year")
    plt.xlabel("Year")
    plt.ylabel("Average Resale Price")
    plt.savefig("average_resale_price_per_year.png")
    plt.show()

    plt.figure(figsize=(15, 6))
    sns.lineplot(
        x="year",
        y="resale_price",
        hue="flat_type",
        data=average_price_per_year_ftype,
        marker="o",
    )
    plt.title("Average Resale Price Per Year by Flat Type")
    plt.xlabel("Year")
    plt.ylabel("Average Resale Price")
    plt.savefig("average_resale_price_per_year_ftype.png")
    plt.show()

    plt.figure(figsize=(15, 6))
    sns.lineplot(
        x="year",
        y="resale_price",
        hue="town",
        data=average_price_per_year_town,
        marker="o",
    )
    plt.title("Average Resale Price Per Year by Town")
    plt.xlabel("Year")
    plt.ylabel("Average Resale Price")
    plt.savefig("average_resale_price_per_year_town.png")
    plt.show()

    # Plot price change percentage per year
    df["price_change"] = df.groupby("year")["resale_price"].pct_change() * 100

    plt.figure(figsize=(15, 6))
    sns.barplot(
        x="year",
        y="price_change",
        data=df.dropna(),
        errorbar=None,
    )
    plt.title("Yearly Price Change Percentage")
    plt.xlabel("Year")
    plt.ylabel("Price Change (%)")
    plt.savefig("yearly_price_change_percentage.png")
    plt.show()


def load_and_preprocess_data(df):
    """
    Pre-process data and split into training and test sets.

    Parameters
    ----------
    df : DATAFRAME
        CONSOLIDATED HOUSING DATA.

    Returns
    -------
    X_train_preprocessed : CSR MATRIX
        PREPROCESSED X TRAINING SET.
    X_test_preprocessed : CSR MATRIX
        PREPROCESSED X TEST SET.
    y_train : SERIES
        Y TRAINING SET.
    y_test : SERIES
        Y TEST SET.
    preprocessor : OBJECT
        PREPROCESSOR OBJECT.
    feature_names : LIST
        LIST OF FEATURES IN THE TRAINING/TEST SET.

    """
    # Convert re_binned_range to numeric
    df["re_binned_range"] = (
        df["re_binned_range"].str.split(" TO ").str[0].astype(float)
    )
    # Filter most recent data (after year 2018)
    df = df[df["year"] > 2018]

    # Drop features and split features and target
    X = df.drop(["resale_price", "block", "street_name", "flat_model"], axis=1)
    y = df["resale_price"]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Identify numeric and categorical columns
    numeric_features = [
        "floor_area_sqm",
        "remaining_lease",
        "year",
        "month",
        "re_binned_range",
    ]
    categorical_features = [
        "town",
        "flat_type",
    ]

    # Create preprocessing pipelines
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="missing"),
            ),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Fit the preprocessor to the training data
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    # Get feature names after preprocessing
    onehot_encoder = preprocessor.named_transformers_["cat"].named_steps[
        "onehot"
    ]
    cat_feature_names = onehot_encoder.get_feature_names_out(
        categorical_features
    ).tolist()
    feature_names = numeric_features + cat_feature_names

    return (
        X_train_preprocessed,
        X_test_preprocessed,
        y_train,
        y_test,
        preprocessor,
        feature_names,
    )


def build_and_tune_model(X_train, X_test, y_train, y_test, feature_names):
    """
    Perform tuning using Bayesian search optimisation.
    Build random forest regressor model.

    Parameters
    ----------
    X_train : CSR MATRIX
        PREPROCESSED X TRAINING SET.
    X_test : CSR MATRIX
        PREPROCESSED X TEST SET.
    y_train : SERIES
        Y TRAINING SET.
    y_test : SERIES
        Y TEST SET.
    feature_names : LIST
        LIST OF FEATURES IN THE TRAINING/TEST SET.

    Returns
    -------
    rf : OBJECT
        RANDOM FOREST REGRESSOR MODEL TO PREDICT HOUSING PRICES.
    imp_df : DataFrame
        ORDERED TABLE OF FEATURE IMPORTANCES.

    """

    # =============================================================================
    #     # PERFORM BAYESIAN SEARCH TO TUNE HYPERPARAMETERS
    #     # Define the search space
    #     search_spaces = {
    #         "n_estimators": (100, 1000),
    #         "max_depth": (10, 100),
    #         "min_samples_split": (2, 10),
    #         "min_samples_leaf": (1, 5),
    #     }
    #
    #     # Create a RandomForestRegressor
    #     rf = RandomForestRegressor(random_state=42)
    #
    #     # Set up BayesSearchCV
    #     bayes_search = BayesSearchCV(
    #         estimator=rf,
    #         search_spaces=search_spaces,
    #         n_iter=50,  # number of parameter settings that are sampled
    #         cv=5,
    #         verbose=3,
    #         random_state=42,
    #         n_jobs=-1,  # use all available cores
    #     )
    #
    #     # Fit BayesSearchCV
    #     bayes_search.fit(X_train, y_train)
    #
    #     # Print the best parameters and best score
    #     print("Best parameters:", bayes_search.best_params_)
    #     print("Best score:", bayes_search.best_score_)
    # =============================================================================

    # Initiate random forest regressor using best parameters
    rf = RandomForestRegressor(
        n_estimators=1000,
        min_samples_split=10,
        min_samples_leaf=1,
        max_depth=31,
        random_state=42,
    ).fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Test set RMSE: {rmse:.2f}")
    print(f"Test set R-squared: {r2:.2f}")

    # Get feature importances
    importances = rf.feature_importances_
    imp_df = pd.DataFrame(
        {"Feature": feature_names, "Gini Importance": importances}
    ).sort_values("Gini Importance", ascending=False)
    print(imp_df.head(10))

    # Save the model
    joblib.dump(rf, "rf_model.joblib")

    return rf, imp_df


if __name__ == "__main__":
    # Ensure data files are extracted to Kaggle_HDB in project root folder
    FILE_DIR = os.path.join(
        os.path.dirname(os.path.realpath("__file__")), "Kaggle_HDB"
    )
    dataset = process_csv_files(FILE_DIR)
    perform_eda(dataset)
    X_train, X_test, y_train, y_test, preprocessor, features = (
        load_and_preprocess_data(dataset)
    )
    rf_model, feature_importance = build_and_tune_model(
        X_train, X_test, y_train, y_test, features
    )
