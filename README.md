# Streaming Traffic Forecasting with Random Forests in MATLAB

This project forecasts **Netflix network traffic** from time-series traffic data using regression trees and random forests implemented from scratch in MATLAB.

The main goal is to compare a **1-step-ahead baseline** with a **direct multi-step forecasting strategy** that predicts up to 20 future time steps without recursively feeding predictions back into the model.

## Project Overview

The MATLAB Live Script performs the complete forecasting pipeline:

* Loads SONICATEL traffic data from CSV.
* Builds time-based and lag-based features.
* Applies a `log1p` transformation to the target traffic values.
* Splits the data chronologically into 80% training and 20% testing data.
* Applies min-max normalization using training-set statistics only.
* Trains a custom regression tree for 1-step forecasting.
* Trains a custom random forest for 1-step forecasting.
* Trains 20 separate random forests for direct multi-step forecasting.
* Evaluates forecasts using RMSE in the original traffic units.
* Plots actual vs. predicted traffic and RMSE across forecast horizons.

## Forecasting Strategy

### 1-Step Baseline

The baseline predicts the next traffic observation using:

* **Regression Tree (RT)**
* **Random Forest (RF)**

The test features contain the real historical lag values, providing a 1-step forecasting benchmark.

### Direct Multi-Step Forecasting

Instead of recursively using one prediction to generate the next, the project trains a separate random forest for every forecast horizon:

```text
Model 1  -> traffic at t + 1
Model 2  -> traffic at t + 2
...
Model 20 -> traffic at t + 20
```

This avoids the error accumulation that can occur in recursive multi-step forecasting.

With 5-minute observations, the 20-step forecast corresponds to approximately **100 minutes ahead**.

## Feature Engineering

For each observation, the model uses:

* Day of week
* Hour of day
* Previous 12 Netflix traffic observations
* Netflix traffic from the same time one day earlier (`288` samples)
* Optional traffic variables when available:

  * `IN`
  * `OUT`
  * `VOIP`

Because the data is treated as 5-minute intervals:

* 12 lag observations represent approximately **1 hour of recent history**.
* 288 lag observations represent approximately **24 hours**.

## Dataset Requirements

The script expects the following file in the MATLAB working directory:

```text
SONICATEL traffic train.csv
```

### Required Columns

```text
dayweek
hour
Netflix
```

### Optional Columns

```text
IN
OUT
VOIP
```

If the optional columns are missing, the script automatically continues using only the time features and Netflix lag features.

## Model Configuration

| Parameter                |                    Value |
| ------------------------ | -----------------------: |
| Recent Netflix lags      |                       12 |
| Daily lag                |                      288 |
| Maximum forecast horizon |                       20 |
| Training split           |                      80% |
| Testing split            |                      20% |
| Trees per random forest  |                       80 |
| Maximum tree depth       |                       12 |
| Minimum leaf size        |                        2 |
| RF features per split    | `max(2, floor(sqrt(p)))` |

The regression-tree and random-forest algorithms are implemented directly in the Live Script using the custom `buildTree` and `predictTree` functions.

## Results

Saved output from the current experiment produced the following 1-step baseline RMSE values:

| Model           |    1-Step RMSE |
| --------------- | -------------: |
| Regression Tree | 24,424,922.753 |
| Random Forest   | 15,210,672.250 |

The random forest significantly improves on the single regression tree for the 1-step forecasting task.

Selected RMSE values from the direct multi-step random-forest models are:

| Forecast Horizon | Approx. Time Ahead |       RMSE |
| ---------------: | -----------------: | ---------: |
|                1 |              5 min | 12,971,000 |
|                5 |             25 min | 14,314,000 |
|               10 |             50 min | 16,172,000 |
|               15 |             75 min | 16,515,000 |
|               20 |            100 min | 17,993,000 |

Overall, forecast error tends to increase as the prediction horizon becomes longer.

> Results depend on the dataset and random bootstrap sampling used to train each forest.

## Visualizations

The script generates three figures:

1. **1-step forecast comparison** — actual traffic vs. regression-tree and random-forest predictions.
2. **20-step direct forecast** — actual traffic vs. the horizon-20 random-forest prediction.
3. **RMSE vs. forecast horizon** — shows how prediction error changes from horizon 1 through 20.

## How to Run

1. Clone or download this repository.
2. Place `SONICATEL traffic train.csv` in the project directory.
3. Open:

```text
Streaming_Traffic_Data.mlx
```

4. Run the Live Script in MATLAB.
5. Review the RMSE values printed in the output.
6. Inspect the generated prediction and error plots.

## Repository Structure

```text
.
├── Streaming_Traffic_Data.mlx   # Main forecasting pipeline
├── SONICATEL traffic train.csv  # Input dataset
└── README.md
```

## Implementation Details

### Target Transformation

Traffic values are transformed before model training:

```matlab
Y_log = log1p(Y);
```

Predictions are converted back to the original scale using:

```matlab
expm1(...)
```

This reduces the effect of very large traffic values during training while allowing evaluation in the original traffic units.

### Time-Series Train/Test Split

The dataset is split chronologically rather than randomly shuffled:

```text
First 80% -> Training
Last 20%  -> Testing
```

Preserving time order is important for forecasting because it prevents future observations from leaking into the training set.

### Feature Normalization

Min-max normalization parameters are calculated using only the training set:

```matlab
min_vals = min(X_train, [], 1);
max_vals = max(X_train, [], 1);
```

The same training statistics are then applied to the test set, preventing test-data leakage.

### Custom Regression Tree

The `buildTree` function recursively searches for feature thresholds that minimize squared prediction error.

Tree growth stops when:

* The maximum tree depth is reached.
* The node contains too few observations.
* No split improves the prediction error.

Leaf predictions are calculated as the mean target value of the observations reaching that leaf.

### Custom Random Forest

Each random forest contains 80 regression trees.

For every tree:

* A bootstrap sample of the training data is generated.
* A random subset of features is considered at each split.
* A regression tree is trained on the bootstrap sample.

The final forest prediction is the average prediction across all trees.

## Why Direct Multi-Step Forecasting?

A recursive forecasting model predicts:

```text
t+1 -> use prediction to predict t+2
t+2 -> use prediction to predict t+3
...
```

Errors from early predictions can therefore propagate into later forecasts.

This project instead uses **direct forecasting**, where every horizon has an independent model:

```text
X(t) -> Model 1  -> Y(t+1)
X(t) -> Model 2  -> Y(t+2)
...
X(t) -> Model 20 -> Y(t+20)
```

No predicted traffic value is recursively fed back into the model.

## Possible Improvements

Future work could include:

* Comparing the custom implementation with MATLAB's built-in ensemble-learning models.
* Adding MAE, MAPE, R², and normalized RMSE metrics.
* Using walk-forward or rolling-window validation.
* Hyperparameter tuning for tree depth, leaf size, number of trees, and lag lengths.
* Adding cyclical encoding for hour-of-day and day-of-week.
* Calculating feature importance.
* Comparing direct forecasting with recursive forecasting.
* Testing gradient boosting methods.
* Testing neural networks or LSTM-based forecasting.
* Saving trained models and prediction results for reproducible experiments.

## Technologies

* MATLAB
* Time-series forecasting
* Feature engineering
* Regression trees
* Random forests
* Bootstrap aggregation
* Direct multi-step forecasting

