# Multi-Operator Free-Floating Trip Destination Prediction in Public Mobility Sharing Systems

Daniel Kerger, Heiner Stuckenschmidt

Data and Web Science Group, University of Mannheim, B 6, 26, Mannheim, 68159, Baden-Württemberg, Germany

## Abstract
Public mobility sharing systems are an important component of sustainable transport, particularly for last-mile travel. However, analysing trip patterns using open standards such as GBFS can be challenging due to vehicles frequently being assigned new identifiers and missing GPS trajectories, preventing a detailed tracking. To overcome this limitation, we present a machine learning pipeline that \red{retrospectively} predicts trip destinations within this circumstances—making it possible to partially recover travel patterns for GBFS data.

Our approach involves a three-step prediction pipeline: (1) candidate generation and reduction using spatial-temporal filtering; (2) multi-target regression via XGBoost to estimate destination coordinates; and (3) selection of the best-matching candidate. Our approach achieves an average accuracy of 77% across five German and 74\% across five international cities within a tolerance of 500 metres. Compared to existing approaches, our method improves prediction accuracy by an average of 20% over methods that also do not use user-specific or GPS trajectory features.

These results demonstrate the feasibility of accurately predicting destinations in shared mobility despite rotating vehicle identifiers and missing trajectory data, thereby supporting improved system analysis and planning.

## Usage
This repository contains the needed code to execute the pipeline used in our paper with a subset of the datasets from the city of Heidelberg. To run the notebook, install all dependencies listed in the requirements.txt file with Python 3.10. All results are displayed in the notebook output cells.

## OSRM Setup
To run the notebook, you need to have an OSRM backend running with cartographic data for the city you are using. Data for Heidelberg can be downloaded from [Geofabrik](https://download.geofabrik.de/europe/germany/baden-wuerttemberg.html). After downloading the data, you can set up OSRM by following the instructions on the [OSRM GitHub page](https://github.com/Project-OSRM/osrm-backend). We have added a docker-compose file to facilitate the setup process. You can start the OSRM service by running
```bash
docker-compose up -d
```
in the directory containing the docker-compose.yml file.

## Data Usage Restrictions
The dataset included in this repository is intended solely for use within the scope of reproducing the experiments described in our paper. It may not be used for other scenarios or purposes without explicit permission from the authors.
