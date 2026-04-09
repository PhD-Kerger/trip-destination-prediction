import csv  # CSV file reading
import json
from .logger import Logger  # Custom logger
import uuid  # Unique ID generation
from collections import defaultdict  # Default dictionary
from datetime import datetime  # Date and time manipulation
import requests  # HTTP requests
from random import randint  # Random number generation
import tqdm  # Progress bars
import joblib  # Model/scaler loading
from pathlib import Path
import pandas as pd  # Data manipulation
import numpy as np  # Numerical operations
from geopy.distance import geodesic  # Geospatial distance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class Predictor:
    def __init__(
        self,
        trips,
        OSRM_ENABLED=False,
        OSRM_ALTERNATIVE_PERCENTAGE=0,
        OSRM_ENDPOINT="http://localhost:5000/route/v1/bike/",
        TARGETCITY="test",
        NAME="test",
    ):

        self.all_rentals, self.all_returns = self.parse_trips_from_csv_string(trips)
        self.OSRM_ENABLED = OSRM_ENABLED
        self.OSRM_ALTERNATIVE_PERCENTAGE = OSRM_ALTERNATIVE_PERCENTAGE
        self.OSRM_ENDPOINT = OSRM_ENDPOINT
        self.TARGETCITY = TARGETCITY
        self.NAME = NAME
        self.thresholds = [
            0,
            50,
            100,
            150,
            200,
            250,
            300,
            350,
            400,
            450,
            500,
            550,
            600,
            650,
            700,
            750,
            800,
            850,
            900,
            950,
            1000,
        ]

        # Setup logger as class attribute
        self.logger = Logger.get_logger(
            name=self.__class__.__name__,
            log_file_path=Path("logs") / "logs.log",
        )

    def parse_trips_from_csv_string(csv_string):
        """f
        Parse a CSV string of trips into rentals and returns dicts,
        applying the same filters as process_file.
        """
        rentals, returns = {}, {}
        reader = csv.DictReader(csv_string.splitlines())
        for row in reader:
            rental_time = int(datetime.fromisoformat(row["timestamp_lend"]).timestamp())
            return_time = int(
                datetime.fromisoformat(row["timestamp_returned"]).timestamp()
            )

            if return_time - rental_time > 3 * 60 * 60:
                continue

            rental_battery = row["pedelec_battery_lend"]
            return_battery = row["pedelec_battery_returned"]

            if (
                rental_battery is None
                or return_battery is None
                or rental_battery == ""
                or return_battery == ""
                or return_battery == "100"
            ):
                rental_battery = 0
                return_battery = 0

            if return_battery > rental_battery:
                continue

            yyyy_mm_dd = row["timestamp_lend"].split(" ")[0].split("T")[0]
            if yyyy_mm_dd not in rentals:
                rentals[yyyy_mm_dd] = []
            if yyyy_mm_dd not in returns:
                returns[yyyy_mm_dd] = []

            rentals[yyyy_mm_dd].append(
                {
                    "vehicle_id": row["vehicle_id"],
                    "spoofed_id": str(uuid.uuid4()),
                    "timestamp": rental_time,
                    "lat": float(row["lat_lend"]),
                    "lng": float(row["lng_lend"]),
                    "lat_returned_true": float(row["lat_returned"]),
                    "lng_returned_true": float(row["lng_returned"]),
                    "battery": int(rental_battery),
                    "range": int(row["current_range_meters_lend"]),
                }
            )

            returns[yyyy_mm_dd].append(
                {
                    "vehicle_id": row["vehicle_id"],
                    "spoofed_id": str(uuid.uuid4()),
                    "timestamp": return_time,
                    "lat": float(row["lat_returned"]),
                    "lng": float(row["lng_returned"]),
                    "battery": int(return_battery),
                    "range": int(row["current_range_meters_returned"]),
                }
            )
        return rentals, returns

    def compute_accuracy_at_thresholds(
        self, df: pd.DataFrame, thresholds: list[float]
    ) -> pd.DataFrame:
        """
        Compute accuracy at various distance thresholds.
        For each threshold, relax the condition for a correct prediction
        to include predictions within the threshold distance.
        """
        accuracies = []

        for threshold in thresholds:
            relaxed_correct = df["correct"] | (df["distance"] <= threshold)
            accuracy_pct = relaxed_correct.sum() / len(df)
            df["predicted_label"] = df["vehicle_id_rental"].where(
                relaxed_correct, df["vehicle_id_return"]
            )
            accuracies.append(
                {
                    "threshold": threshold,
                    "accuracy": accuracy_pct,
                }
            )

        return pd.DataFrame(accuracies)

    def predict(self):
        # Filter out returns with 100% battery (likely manually charged, not predictable)
        removed = []
        n_100 = 0
        all_returns_filtered = defaultdict(list)
        for date, returns in self.all_returns.items():
            for ret in returns:
                if ret["battery"] is not None and ret["battery"] < 100:
                    all_returns_filtered[date].append(ret)
                else:
                    n_100 += 1
                    # Track removed spoofed_id for reference
                    removed.append(ret["spoofed_id"])

        self.logger.info(f"Removed {n_100} returns with 100% battery")

        # Match each rental to possible returns using multiple filters
        rental_to_possible_returns = defaultdict(list)

        blacklist = set()  # Track returns already matched to a non-moving trip
        blacklist_pairs = set()  # Track matched rental-return pairs

        for date, rentals in tqdm.tqdm(
            self.all_rentals.items(), total=len(self.all_rentals)
        ):
            returns = all_returns_filtered[date]
            for rental in rentals:
                possible_returns = []
                for ret in returns:
                    # Skip if already matched to a non-moving trip
                    if ret["spoofed_id"] in blacklist:
                        continue

                    rental_time = rental["timestamp"]
                    return_time = ret["timestamp"]
                    rental_battery = rental["battery"]
                    return_battery = ret["battery"]

                    time_diff = return_time - rental_time
                    # If battery is None, set to 0
                    if rental_battery is None:
                        rental_battery = 0
                    if return_battery is None:
                        return_battery = 0

                    battery_diff = rental_battery - return_battery

                    # Filter out invalid or unrealistic trips
                    if rental_time > return_time:
                        continue
                    if return_battery > rental_battery:
                        continue
                    if time_diff > 1800:  # More than 30 minutes
                        continue
                    if battery_diff > 10:
                        continue

                    # Calculate geospatial and range-based features
                    if self.OSRM_ENABLED:
                        response = requests.get(
                            f"{self.OSRM_ENDPOINT}{rental['lng']},{rental['lat']};{ret['lng']},{ret['lat']}?overview=false&alternatives=3"
                        )
                        data = response.json()
                        # check if to use alternative with a certain probability
                        if "routes" in data and len(data["routes"]) > 0:
                            rand = randint(1, 100)
                            if rand <= self.OSRM_ALTERNATIVE_PERCENTAGE:
                                distance = int(data["routes"][-1]["distance"])
                            else:
                                distance = int(data["routes"][0]["distance"])
                        else:
                            # Use geopy for geodesic distance as fallback
                            distance = int(
                                geodesic(
                                    (rental["lat"], rental["lng"]),
                                    (ret["lat"], ret["lng"]),
                                ).meters
                            )
                    else:
                        # Use geopy for geodesic distance
                        distance = int(
                            geodesic(
                                (rental["lat"], rental["lng"]), (ret["lat"], ret["lng"])
                            ).meters
                        )
                    mean_speed = (
                        round((distance / 1000) / (time_diff / 3600), 2)
                        if time_diff > 0
                        else 0
                    )
                    range_diff = rental["range"] - ret["range"]
                    mean_speed_range_based = (
                        round((range_diff / 1000) / (time_diff / 3600), 2)
                        if time_diff > 0
                        else 0
                    )

                    # More filters for realistic trips
                    if mean_speed > 20:
                        continue
                    if distance > 3000:
                        continue

                    # Special case: non-moving trip (likely correct match)
                    if (
                        time_diff <= 6
                        and range_diff == 0
                        and battery_diff <= 1
                        and distance < 1000
                    ):
                        possible_returns = []
                        blacklist.add(ret["spoofed_id"])
                        blacklist_pairs.add((rental["spoofed_id"], ret["spoofed_id"]))
                        possible_returns.append(
                            {
                                "vehicle_id_rental": rental["vehicle_id"],
                                "vehicle_id_return": ret["vehicle_id"],
                                "spoofed_id_rental": rental["spoofed_id"],
                                "spoofed_id_return": ret["spoofed_id"],
                                "lat_rental": rental["lat"],
                                "lng_rental": rental["lng"],
                                "lat_return": ret["lat"],
                                "lng_return": ret["lng"],
                                "lat_returned_true": rental["lat_returned_true"],
                                "lng_returned_true": rental["lng_returned_true"],
                                "timestamp_rental": rental_time,
                                "timestamp_return": return_time,
                                "battery_rental": rental_battery,
                                "battery_return": return_battery,
                                "range_rental": rental["range"],
                                "range_return": ret["range"],
                                "distance": distance,
                                "time_diff": time_diff,
                                "battery_diff": battery_diff,
                                "range_diff": range_diff,
                                "speed": mean_speed,
                                "speed_range_based": mean_speed_range_based,
                                "date": date,
                            }
                        )
                        break

                    # Skip if range difference is too high for zero time
                    if time_diff == 0 and range_diff >= 1000:
                        continue

                    # Add possible return
                    possible_returns.append(
                        {
                            "vehicle_id_rental": rental["vehicle_id"],
                            "vehicle_id_return": ret["vehicle_id"],
                            "spoofed_id_rental": rental["spoofed_id"],
                            "spoofed_id_return": ret["spoofed_id"],
                            "lat_rental": rental["lat"],
                            "lng_rental": rental["lng"],
                            "lat_return": ret["lat"],
                            "lng_return": ret["lng"],
                            "lat_returned_true": rental["lat_returned_true"],
                            "lng_returned_true": rental["lng_returned_true"],
                            "timestamp_rental": rental_time,
                            "timestamp_return": return_time,
                            "battery_rental": rental_battery,
                            "battery_return": return_battery,
                            "range_rental": rental["range"],
                            "range_return": ret["range"],
                            "distance": distance,
                            "time_diff": time_diff,
                            "battery_diff": battery_diff,
                            "range_diff": range_diff,
                            "speed": mean_speed,
                            "speed_range_based": mean_speed_range_based,
                            "date": date,
                        }
                    )

                if date not in rental_to_possible_returns:
                    rental_to_possible_returns[date] = {}

                rental_to_possible_returns[date][
                    rental["spoofed_id"]
                ] = possible_returns

        # Load trained model and scaler
        model = joblib.load(f"data/{self.NAME}/model.pkl")
        scaler = joblib.load(f"data/{self.NAME}/scaler.pkl")
        prediced_trips = []
        all_predicted_trips = []

        # For each rental, predict the most likely return using the model
        for date, rentals in rental_to_possible_returns.items():
            self.logger.info(f"Date: {date}")
            for rental in tqdm.tqdm(rentals, total=len(rentals)):
                data = rentals[rental]
                # Skip if there are no possible returns
                if len(data) == 0:
                    continue
                data = data[0]
                date = pd.to_datetime(data["timestamp_rental"], unit="s")
                distances = []
                distances_true = []
                ret_ids = []
                for ret in rentals[rental]:
                    # Prepare features for prediction
                    ret_features = {
                        "lat_lend": data["lat_rental"],
                        "lng_lend": data["lng_rental"],
                        "time_diff": ret["time_diff"],
                        "battery_diff": ret["battery_diff"],
                        "range_diff": ret["range_diff"],
                        "distance": ret["distance"],
                        "mean_speed_distance_based": ret["speed"],
                        "mean_speed_range_based": ret["speed_range_based"],
                    }
                    X = pd.DataFrame([ret_features])
                    # Scale numerical features
                    X[
                        [
                            "time_diff",
                            "battery_diff",
                            "range_diff",
                            "distance",
                            "mean_speed_distance_based",
                            "mean_speed_range_based",
                        ]
                    ] = scaler.transform(
                        X[
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

                    # Predict return coordinates
                    y_pred = model.predict(X)

                    # Calculate geodesic distances for evaluation
                    distance = geodesic(
                        (y_pred[0][0], y_pred[0][1]),
                        (ret["lat_return"], ret["lng_return"]),
                        ellipsoid="WGS-84",
                    ).m
                    distance_true = geodesic(
                        (y_pred[0][0], y_pred[0][1]),
                        (ret["lat_returned_true"], ret["lng_returned_true"]),
                        ellipsoid="WGS-84",
                    ).m
                    distances.append(distance)
                    distances_true.append(distance_true)
                    ret_ids.append(ret["vehicle_id_return"])

                # Select the return with the minimum predicted distance
                min_distance_idx = distances.index(min(distances))
                predicted_trip = rentals[rental][min_distance_idx]
                predicted_trip["lat_returned"] = y_pred[0][0]
                predicted_trip["lng_returned"] = y_pred[0][1]

                # Store prediction results
                prediced_trips.append(
                    {
                        "vehicle_id_rental": predicted_trip["vehicle_id_rental"],
                        "vehicle_id_return": predicted_trip["vehicle_id_return"],
                        "distance": min(distances),
                        "distance_true": min(distances_true),
                        "mean_distance": sum(distances) / len(distances),
                        "median_distance": np.median(distances),
                        "distances_true": distances_true,
                        "distances": distances,
                        "possibilities": len(distances),
                        "correct": predicted_trip["vehicle_id_rental"]
                        == predicted_trip["vehicle_id_return"],
                        "lat_lend": predicted_trip["lat_rental"],
                        "lng_lend": predicted_trip["lng_rental"],
                        "lat_returned": predicted_trip["lat_return"],
                        "lng_returned": predicted_trip["lng_return"],
                        "pedelec_battery_lend": predicted_trip["battery_rental"],
                        "pedelec_battery_returned": predicted_trip["battery_return"],
                        "pedelec_battery_diff": predicted_trip["battery_diff"],
                        "current_range_meters_lend": predicted_trip["range_rental"],
                        "weekday_lend": date.weekday(),
                        "hour_lend": date.hour,
                        "possible": predicted_trip["vehicle_id_rental"] in ret_ids,
                    }
                )
                for ret in rentals[rental]:
                    predicted_label = False
                    # Mark the one with the lowest distance as predicted
                    if ret == predicted_trip:
                        predicted_label = True
                    # Store all predictions for confusion matrix
                    all_predicted_trips.append(
                        {
                            "true_label": ret["vehicle_id_rental"]
                            == ret["vehicle_id_return"],
                            "predicted_label": predicted_label,
                        }
                    )

        # Convert predictions to DataFrame for analysis
        df = pd.DataFrame(prediced_trips)

        # Count possible and not possible predictions
        self.logger.info(f"Possible: {len([trip for trip in df['possible'] if trip])}")
        self.logger.info(
            f"Not Possible: {len([trip for trip in df['possible'] if not trip])}"
        )

        # Compute confusion matrix for all predictions
        true_labels = [trip["true_label"] for trip in all_predicted_trips]
        predicted_labels = [trip["predicted_label"] for trip in all_predicted_trips]
        confusion_matrix_result = confusion_matrix(true_labels, predicted_labels)

        self.logger.info(f"Confusion matrix:\n{confusion_matrix_result}")

        # Calculate and log precision, recall, f1 score, and accuracy
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)
        accuracy = accuracy_score(true_labels, predicted_labels)

        self.logger.info(f"Precision: {precision}")
        self.logger.info(f"Recall: {recall}")
        self.logger.info(f"F1 Score: {f1}")
        self.logger.info(f"Accuracy: {accuracy}")

        accuracies = self.compute_accuracy_at_thresholds(df, self.thresholds)
        for _, row in accuracies.iterrows():
            self.logger.info(
                f"Threshold: {row['threshold']}m, Accuracy: {row['accuracy']:.2%}"
            )

        # return a json with all predicted trips and their details, also include the accuracy at different thresholds and the confusion matrix, precision, recall and f1 score
        results = {
            "predicted_trips": prediced_trips,
            "accuracies": accuracies.to_dict(orient="records"),
            "confusion_matrix": confusion_matrix_result.tolist(),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
        }

        # generate timestamp for when the prediction was made
        timestamp = datetime.now().isoformat()
        results["timestamp"] = timestamp

        # write output to json file
        with open(f"data/{self.NAME}/results_{timestamp}.json", "w") as f:
            json.dump(results, f, indent=4)

        self.logger.info(f"Results saved to results_{self.NAME}_{timestamp}.json")
