import sys
import yaml
from src import Trainer, Predictor, Logger

# Load configuration from YAML file
with open("/app/config.yaml", "r") as f:
    config = yaml.safe_load(f)

OSRM_ENABLED = config["osrm"]["enabled"]
OSRM_ALTERNATIVE_PERCENTAGE = config["osrm"]["alternative_percentage"]
OSRM_ENDPOINT = config["osrm"]["osrm_endpoint"]

TARGETCITY = config["targetcity"]
NAME = config["name"]
SPLIT_RATIO = config["training"]["split_ratio"]
HYP_OPTIMIZATION_ENABLED = config["training"]["hyperparameter_optimization"]["enabled"]
HYP_TRIALS = config["training"]["hyperparameter_optimization"]["trials"]
HYP_SEED = config["training"]["hyperparameter_optimization"]["seed"]

HYPERPARAMETERS_N_ESTIMATORS = config["training"]["hyperparameters"]["n_estimators"]
HYPERPARAMETERS_MAX_DEPTH = config["training"]["hyperparameters"]["max_depth"]
HYPERPARAMETERS_LEARNING_RATE = config["training"]["hyperparameters"]["learning_rate"]
HYPERPARAMETERS_SUBSAMPLE = config["training"]["hyperparameters"]["subsample"]
HYPERPARAMETERS_COLSAMPLE_BYTREE = config["training"]["hyperparameters"][
    "colsample_bytree"
]

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("predict", "train"):
        logger = Logger.get_logger(
            name="TripDestinationPrediction",
            log_file_path="logs/logs.log",
        )
        logger.error("First argument must be 'train' or 'predict'.")
        sys.exit(1)

    command = sys.argv[1]

    if command == "train":
        trainer = Trainer(
            NAME=NAME,
            TARGETCITY=TARGETCITY,
            SPLIT_RATIO=SPLIT_RATIO,
            HYP_OPTIMIZATION_ENABLED=HYP_OPTIMIZATION_ENABLED,
            HYP_TRIALS=HYP_TRIALS,
            HYP_SEED=HYP_SEED,
            HYPERPARAMETERS_N_ESTIMATORS=HYPERPARAMETERS_N_ESTIMATORS,
            HYPERPARAMETERS_MAX_DEPTH=HYPERPARAMETERS_MAX_DEPTH,
            HYPERPARAMETERS_LEARNING_RATE=HYPERPARAMETERS_LEARNING_RATE,
            HYPERPARAMETERS_SUBSAMPLE=HYPERPARAMETERS_SUBSAMPLE,
            HYPERPARAMETERS_COLSAMPLE_BYTREE=HYPERPARAMETERS_COLSAMPLE_BYTREE,
        )
        # check if there is a third argument for data via command line
        if len(sys.argv) >= 3:
            logger.info("Loading trips from CSV string provided as command line argument.")
            trainer.load_trips_from_csv_string(sys.argv[2])
        else:
            logger.info("Loading trips from database.")
            trainer.load_trips_from_db()
        
        
        trainer.train()
    elif command == "predict":
        if len(sys.argv) < 3:
            logger.error("'predict' requires a CSV string as the second argument.")
            sys.exit(1)
        predictor = Predictor(
            trips=sys.argv[2],
            OSRM_ENABLED=OSRM_ENABLED,
            OSRM_ALTERNATIVE_PERCENTAGE=OSRM_ALTERNATIVE_PERCENTAGE,
            OSRM_ENDPOINT=OSRM_ENDPOINT,
            TARGETCITY=TARGETCITY,
            NAME=NAME,
        )
        predictor.predict()
