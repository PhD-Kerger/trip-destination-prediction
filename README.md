# Multi-Operator Free-Floating Trip Destination Prediction in Public Mobility Sharing Systems

Daniel Kerger, Heiner Stuckenschmidt

Data and Web Science Group, University of Mannheim, B 6, 26, Mannheim, 68159, Baden-Württemberg, Germany

## Abstract

Public mobility sharing systems are an important component of sustainable transport, particularly for last-mile travel. However, analysing trip patterns using open standards such as GBFS can be challenging due to vehicles frequently being assigned new identifiers and missing GPS trajectories, preventing a detailed tracking. To overcome this limitation, we present a machine learning pipeline that \red{retrospectively} predicts trip destinations within this circumstances—making it possible to partially recover travel patterns for GBFS data.

Our approach involves a three-step prediction pipeline: (1) candidate generation and reduction using spatial-temporal filtering; (2) multi-target regression via XGBoost to estimate destination coordinates; and (3) selection of the best-matching candidate. Our approach achieves an average accuracy of 77% across five German and 74\% across five international cities within a tolerance of 500 metres. Compared to existing approaches, our method improves prediction accuracy by an average of 20% over methods that also do not use user-specific or GPS trajectory features.

These results demonstrate the feasibility of accurately predicting destinations in shared mobility despite rotating vehicle identifiers and missing trajectory data, thereby supporting improved system analysis and planning.

## Usage

This repository contains the needed code to execute the pipeline used in our paper with a subset of the datasets from the city of Heidelberg. To run the notebook, install all dependencies listed in the requirements.txt file with Python 3.10. All results are displayed in the notebook output cells.

If you want to use the Docker Compose project with the provided Dockerfile, you can build the image with the following command:

```bash
docker-compose up
```

Then you can run the script via an API call to the running container:

```docker exec trip-destination-prediction python main.py "vehicle_id,timestamp_lend,lng_lend,lat_lend,timestamp_returned,lng_returned,lat_returned,current_range_meters_lend,pedelec_battery_lend,pedelec_battery_returned,current_range_meters_returned
7194,2024-03-01 00:06:00+00:00,8.6419,49.4125,2024-03-01 00:12:00+00:00,8.6419,49.4125,37000,83,83,37000
12078,2024-03-01 00:06:00+00:00,8.6893,49.4137,2024-03-01 00:15:00+00:00,8.6801,49.4233,46000,100,98,45000
2719,2024-03-01 00:06:00+00:00,8.6541,49.4061,2024-03-01 00:15:00+00:00,8.668,49.4013,39000,87,84,37000
10122,2024-03-01 00:09:00+00:00,8.7087,49.4114,2024-03-01 00:18:00+00:00,8.7128,49.4099,14000,40,40,14000"
```

## OSRM Setup

To run the notebook, you need to have an OSRM backend running with cartographic data for the city you are using. Data for Heidelberg can be downloaded from [Geofabrik](https://download.geofabrik.de/europe/germany/baden-wuerttemberg.html). After downloading the data, you can set up OSRM by following the instructions on the [OSRM GitHub page](https://github.com/Project-OSRM/osrm-backend).

## Data Usage Restrictions

The dataset included in this repository is intended solely for use within the scope of reproducing the experiments described in our paper. It may not be used for other scenarios or purposes without explicit permission from the authors.

## Citation

```bibtex
@article{KERGER2026100105,
  title = {Multi-operator free-floating GBFS trip destination prediction in public mobility sharing systems},
  journal = {Journal of Cycling and Micromobility Research},
  volume = {7},
  pages = {100105},
  year = {2026},
  issn = {2950-1059},
  doi = {https://doi.org/10.1016/j.jcmr.2025.100105},
  url = {https://www.sciencedirect.com/science/article/pii/S295010592500049X},
  author = {Daniel Kerger and Heiner Stuckenschmidt},
}
```
