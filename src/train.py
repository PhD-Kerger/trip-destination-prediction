import json
import os  # File system operations
from .logger import Logger  # Custom logger
from .db_engine import DBEngine  # Database engine
import joblib  # Model/scaler loading
import pandas as pd  # Data manipulation
import numpy as np  # Numerical operations
from geopy.distance import geodesic  # Geospatial distance
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from optuna.samplers import TPESampler
import optuna


class Trainer:
    def __init__(
        self,
        NAME="test",
        TARGETCITY="test",
        SPLIT_RATIO=0.8,
        HYP_OPTIMIZATION_ENABLED=False,
        HYP_TRIALS=50,
        HYP_SEED=42,
        HYPERPARAMETERS_N_ESTIMATORS=100,
        HYPERPARAMETERS_MAX_DEPTH=6,
        HYPERPARAMETERS_LEARNING_RATE=0.1,
        HYPERPARAMETERS_SUBSAMPLE=1.0,
        HYPERPARAMETERS_COLSAMPLE_BYTREE=1.0,
    ):
        self.NAME = NAME
        self.TARGETCITY = TARGETCITY
        self.SPLIT_RATIO = SPLIT_RATIO
        self.HYP_OPTIMIZATION_ENABLED = HYP_OPTIMIZATION_ENABLED
        self.HYP_TRIALS = HYP_TRIALS
        self.HYP_SEED = HYP_SEED
        self.HYPERPARAMETERS_N_ESTIMATORS = HYPERPARAMETERS_N_ESTIMATORS
        self.HYPERPARAMETERS_MAX_DEPTH = HYPERPARAMETERS_MAX_DEPTH
        self.HYPERPARAMETERS_LEARNING_RATE = HYPERPARAMETERS_LEARNING_RATE
        self.HYPERPARAMETERS_SUBSAMPLE = HYPERPARAMETERS_SUBSAMPLE
        self.HYPERPARAMETERS_COLSAMPLE_BYTREE = HYPERPARAMETERS_COLSAMPLE_BYTREE
        self.logger = Logger.get_logger(
            name=self.__class__.__name__,
            log_file_path=f"logs/{self.NAME}_training.log",
        )

    def load_trips_from_db(self):
        db_engine = DBEngine()
        try:
            with db_engine.connect() as conn:
                result = conn.exec_driver_sql(
                    f"SELECT t.timestamp_lend, t.timestamp_return, t.pedelec_battery_lend, "
                    f"t.pedelec_battery_return AS pedelec_battery_returned, "
                    f"t.current_range_meters_lend, "
                    f"t.current_range_meters_return AS current_range_meters_returned, "
                    f"ST_Y(gl.location::geometry) AS lat_lend, "
                    f"ST_X(gl.location::geometry) AS lng_lend, "
                    f"ST_Y(gr.location::geometry) AS lat_returned, "
                    f"ST_X(gr.location::geometry) AS lng_returned "
                    f"FROM trips t "
                    f"JOIN geo_information gl ON t.location_id_lend = gl.location_id "
                    f"JOIN geo_information gr ON t.location_id_return = gr.location_id "
                    f"WHERE t.network_name_lend = '{self.TARGETCITY}' AND t.network_name_return = '{self.TARGETCITY}'"
                ).fetchall()
                self.logger.info(
                    f"Loaded {len(result)} trips for city '{self.TARGETCITY}'."
                )
                self.df = (
                    pd.DataFrame(result, columns=result[0].keys())
                    if result
                    else pd.DataFrame()
                )
        except Exception as e:
            self.logger.error(f"Failed to load trips for city '{self.TARGETCITY}': {e}")
            self.df = pd.DataFrame()  # Set empty DataFrame on failure

    def load_trips_from_csv_string(self, csv_string):
        self.df = csv_string

    def train(self):
        df = self.df
        # check if the data contains the necessary columns
        required_columns = [
            "lat_lend",
            "lat_returned",
            "lng_lend",
            "lng_returned",
            "timestamp_lend",
            "timestamp_return",
        ]
        if not all(column in df.columns for column in required_columns):
            self.logger.error(
                f"Training data must contain the following columns: {required_columns}"
            )
            return

        # Parse timestamps to Unix seconds
        df["timestamp_lend"] = (
            pd.to_datetime(df["timestamp_lend"], utc=True).astype("int64") // 10**9
        )
        df["timestamp_return"] = (
            pd.to_datetime(df["timestamp_return"], utc=True).astype("int64") // 10**9
        )

        # Calculate training features and labels
        df["time_diff"] = df["timestamp_return"] - df["timestamp_lend"]

        # check if battery columns are present, if not create them with default values
        if "pedelec_battery_lend" not in df.columns:
            df["pedelec_battery_lend"] = 0
        if "pedelec_battery_returned" not in df.columns:
            df["pedelec_battery_returned"] = 0

        df["battery_diff"] = df["pedelec_battery_lend"] - df["pedelec_battery_returned"]
        df["range_diff"] = (
            df["current_range_meters_lend"] - df["current_range_meters_returned"]
        )
        df["distance"] = df.apply(
            lambda row: geodesic(
                (row["lat_lend"], row["lng_lend"]),
                (row["lat_returned"], row["lng_returned"]),
            ).meters,
            axis=1,
        )
        df["mean_speed_distance_based"] = df.apply(
            lambda row: (
                round((row["distance"] / 1000) / (row["time_diff"] / 3600), 2)
                if row["time_diff"] > 0
                else 0
            ),
            axis=1,
        )
        df["mean_speed_range_based"] = df.apply(
            lambda row: (
                round((row["range_diff"] / 1000) / (row["time_diff"] / 3600), 2)
                if row["time_diff"] > 0
                else 0
            ),
            axis=1,
        )

        # only scale time_diff, battery_diff, range_diff, distance, mean_speed_distance_based, mean_speed_range_based
        scaler = StandardScaler()
        df[
            [
                "time_diff",
                "battery_diff",
                "range_diff",
                "distance",
                "mean_speed_distance_based",
                "mean_speed_range_based",
            ]
        ] = scaler.fit_transform(
            df[
                [
                    "time_diff",
                    "battery_diff",
                    "range_diff",
                    "distance",
                    "mean_speed_distance_based",
                    "mean_speed_range_based",
                ]
            ]
        )
        df = df[
            [
                "lat_lend",
                "lng_lend",
                "time_diff",
                "battery_diff",
                "range_diff",
                "distance",
                "mean_speed_distance_based",
                "mean_speed_range_based",
                "lat_returned",
                "lng_returned",
            ]
        ]

        # export scaler
        os.makedirs(f"data/{self.NAME}", exist_ok=True)
        joblib.dump(scaler, f"data/{self.NAME}/scaler.pkl")

        # split data into training and test set with SPLIT_RATIO
        X = df[
            [
                "lat_lend",
                "lng_lend",
                "time_diff",
                "battery_diff",
                "range_diff",
                "distance",
                "mean_speed_distance_based",
                "mean_speed_range_based",
            ]
        ]
        y = df[["lat_returned", "lng_returned"]]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1 - self.SPLIT_RATIO, random_state=self.HYP_SEED
        )

        # Train XGBoost model
        if self.HYP_OPTIMIZATION_ENABLED:
            optuna.logging.set_verbosity(optuna.logging.INFO)
            sampler = TPESampler(seed=self.HYP_SEED)
            self.logger.info(
                f"Starting hyperparameter optimization with Optuna with {self.HYP_TRIALS} trials..."
            )
            study = optuna.create_study(direction="minimize", sampler=sampler)
            study.optimize(
                lambda trial: self.objective(
                    trial, X_train, y_train, X_test, y_test, self.HYP_SEED
                ),
                n_trials=self.HYP_TRIALS,
            )

            trial = study.best_trial
            self.logger.info("MSE: {}".format(trial.value))
            self.logger.info("Best hyperparameters: {}".format(trial.params))

            model = XGBRegressor(**trial.params, n_jobs=-1, random_state=self.HYP_SEED)
        else:
            self.logger.info("Training XGBoost model with default hyperparameters...")
            model = XGBRegressor(
                n_estimators=self.HYPERPARAMETERS_N_ESTIMATORS,
                max_depth=self.HYPERPARAMETERS_MAX_DEPTH,
                learning_rate=self.HYPERPARAMETERS_LEARNING_RATE,
                subsample=self.HYPERPARAMETERS_SUBSAMPLE,
                colsample_bytree=self.HYPERPARAMETERS_COLSAMPLE_BYTREE,
                n_jobs=-1,
                random_state=self.HYP_SEED,
            )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        self.logger.info("Mean Squared Error: %s", mean_squared_error(y_test, y_pred))
        self.logger.info("Mean Absolute Error: %s", mean_absolute_error(y_test, y_pred))
        self.logger.info("R2 Score: %s", r2_score(y_test, y_pred))
        self.logger.info("RMSE: %s", np.sqrt(mean_squared_error(y_test, y_pred)))

        # calculate haversine distance between predicted and actual
        y_test["predicted_lat"] = y_pred[:, 0]
        y_test["predicted_lng"] = y_pred[:, 1]

        y_test["haversine_distance"] = y_test.apply(
            lambda row: geodesic(
                (row["lat_returned"], row["lng_returned"]),
                (row["predicted_lat"], row["predicted_lng"]),
                ellipsoid="WGS-84",
            ).m,
            axis=1,
        )

        self.logger.info(
            "Mean Haversine Distance: %s", y_test["haversine_distance"].mean()
        )
        self.logger.info(
            "Median Haversine Distance: %s", y_test["haversine_distance"].median()
        )
        self.logger.info(
            "Std Haversine Distance: %s", y_test["haversine_distance"].std()
        )
        self.logger.info(
            "Min Haversine Distance: %s", y_test["haversine_distance"].min()
        )
        self.logger.info(
            "Max Haversine Distance: %s", y_test["haversine_distance"].max()
        )

        # export model
        os.makedirs(f"data/{self.NAME}", exist_ok=True)
        joblib.dump(model, f"data/{self.NAME}/model.pkl")

        return {
            "mse": mean_squared_error(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
            "hyperparameters": {
                "n_estimators": model.n_estimators,
                "max_depth": model.max_depth,
                "learning_rate": model.learning_rate,
                "subsample": model.subsample,
                "colsample_bytree": model.colsample_bytree,
            },
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mean_haversine_distance": y_test["haversine_distance"].mean(),
            "median_haversine_distance": y_test["haversine_distance"].median(),
            "std_haversine_distance": y_test["haversine_distance"].std(),
            "min_haversine_distance": y_test["haversine_distance"].min(),
            "max_haversine_distance": y_test["haversine_distance"].max(),
        }

    def objective(self, trial, X_train, y_train, X_test, y_test, HYP_SEED=42):
        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)

        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            n_jobs=-1,
            random_state=HYP_SEED,
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "mse": mean_squared_error(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
        }

        # log metrics into json file
        with open(f"data/{self.NAME}/training_metrics.json", "a") as f:
            json.dump(metrics, f, indent=4)

        self.logger.info(
            f"Trial with n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}, subsample={subsample}, colsample_bytree={colsample_bytree} resulted in MSE: {metrics['mse']}, MAE: {metrics['mae']}, RMSE: {metrics['rmse']}, R2: {metrics['r2']}"
        )
