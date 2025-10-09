# Multi-Operator Free-Floating Trip Destination Prediction in Public Mobility Sharing Systems

## Abstract
Public mobility sharing systems are an important component of sustainable transport, particularly for last-mile travel. However, analysing trip patterns using open standards can be challenging due to vehicles frequently being assigned new identifiers, preventing tracking. To overcome this limitation, we present a machine learning pipeline that predicts trip destinations—making it possible to recover travel patterns.

Our approach involves a three-step prediction pipeline: (1) candidate generation and reduction using spatial-temporal filtering; (2) multi-target regression via XGBoost to estimate destination coordinates; and (3) selection of the best-matching candidate. Our approach achieves an average accuracy of 77% within a tolerance of 500 metres across 10 international cities. Compared to existing approaches, our method improves prediction accuracy by an average of 20% over methods that do not use user-specific features.

These results demonstrate the feasibility of accurately predicting destinations in shared mobility despite rotating vehicle identifiers, thereby supporting improved system analysis and planning.

## Usage
This repository contains the needed code to execute the pipeline used in our paper with a subset of the datasets from the city of Heidelberg. To run the notebook, install all dependencies listed in the requirements.txt file with Python 3.10. All results are displayed in the notebook output cells.

## Data Usage Restrictions
The dataset included in this repository is intended solely for use within the scope of reproducing the experiments described in our paper. It may not be used for other scenarios or purposes without explicit permission from the authors.
