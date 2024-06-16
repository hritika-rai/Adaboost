
# Boosting Algorithm Implementation

## Objective
The aim of this implementation is to develop both serial and parallel versions of the AdaBoost algorithm for binary classification problems. Reseach paper with literature review can be found above in Project-Report

## Methodology

### Logic and Architecture of Serial and Parallel AdaBoost Implementations

#### 1. Inputting Data and Preparation:
- **Both Implementations**: Data is read from a CSV file, resulting in `X_data` (features) and `y_data` (labels).

#### 2. Calculating Error for Weak Classifier:
- **Serial Implementation**: For each feature, error and split value are calculated iteratively within the `weak_classifier` function. The best split is chosen based on minimum error.
- **Parallel Implementation**: The error calculation is parallelized using the `joblib` library by distributing the workload across multiple CPU cores for faster computation.

#### 3. Weak Classifier Training:
- **Serial Implementation**: The `weak_classifier` function sequentially searches for the best split and calculates the beta value, which is used to update sample weights later.
- **Parallel Implementation**: The `weak_classifier` function uses parallel processing to compute errors concurrently, improving efficiency.

#### 4. Sample Weight Update:
- **Serial Implementation**: Sample weights are updated sequentially within the `update_weights` function.
- **Parallel Implementation**: Sample weights are updated in a parallelized manner using `joblib`, dividing the workload into smaller chunks to be processed concurrently.

#### 5. Weak Classifier Prediction:
- **Both Implementations**: Use the `weak_predict` function to make predictions based on the trained weak classifiers.

#### 6. AdaBoost Training Loop:
- **Both Implementations**: The training loop iterates through rounds and updates sample weights based on weak classifiers' performance. These weak classifiers are stored in a list (`hlist`).

#### 7. Model Evaluation:
- **Both Implementations**: The model evaluation function computes predictions using weak classifiers and calculates accuracy. The parallel version uses parallel processing for prediction aggregation.

### Key Differences:
- **Parallelism**: The parallel implementation parallelizes critical sections like error calculation and weight updating, optimizing performance.
- **Efficiency**: The parallel implementation is appreciably faster on large data sets due to its capability to simultaneously operate on different groups of data items.
- **Workload Distribution**: The parallel implementation breaks down error calculation and weight updating tasks into smaller units and executes them concurrently. The serial version performs these tasks sequentially.
- **Library Usage**: The parallel implementation utilizes the `joblib` library for efficient work distribution and control among CPU cores.


---

## Pseudocode

### Serial Implementation

1. **Input:**
   - Training dataset `X_train`, `y_train`
   - Number of iterations `num_iter`

2. **Initialization:**
   - Set `D1 = 1/N` for all `n` samples
   - Initialize an empty list `hlist` to store weak classifiers

3. **Training Loop:**
   - For each iteration `t = 1 to num_iter`:
     - Determine which weak classifier `ht(xi)` minimizes the weighted error:
       - Calculate error for every feature and split value using sample weights `D`
       - Choose the optimal split based on the least weighted error
     - Calculate epsilon (`εt`), the error rate of the weak classifier:
       - εt = (Number of misclassifications) / N, where N is the number of samples
     - Calculate alpha (`αt`), the weight of the weak classifier:
       - αt = 0.5 * ln((1 - εt) / εt)
     - Update sample weights `D`:
       - For each sample `i`:
         - Update as follows: `D_t+1(i) = D_t(i) * exp(-αt * ht(xi) * yi)`
         - Normalize `D_t+1(i)` by the normalization constant `Z`:
           - Z = ∑ (D_t(i) * exp(-αt * ht(xi) * yi))
     - Store the weak classifier `ht(xi)` and its parameters (`αt`, split feature, split value) in `hlist`

4. **Combining Weak Classifiers:**
   - For each sample in the test dataset `X_test`:
     - Calculate the weighted sum of predictions from all weak classifiers:
       - H(xi) = ∑ (αt * ht(xi))
     - Determine the class label based on the sign of `H(xi)`

5. **Evaluation:**
   - Evaluate the model's accuracy using the combined weak classifiers on the test dataset.

### Parallel Implementation

1. **Input:**
   - Training dataset `X_train`, `y_train`
   - Number of iterations `num_iter`

2. **Initialization:**
   - Set `D1 = 1/n` for all `n` samples
   - Initialize an empty list `hlist` to store weak classifiers

3. **Training Loop:**
   - For each iteration `t = 1 to num_iter`:
     - Find weak classifier `ht(xi)` in parallel that minimizes the weighted error:
       - Distribute feature calculation across multiple cores using parallel processing
       - Choose the best split based on the minimum weighted error
     - Calculate epsilon (`εt`), the error rate of the weak classifier:
       - εt = (Number of misclassifications) / N, where N is the number of samples
     - Calculate alpha (`αt`), the weight of the weak classifier:
       - αt = 0.5 * ln((1 - εt) / εt)
     - Update sample weights `D` in parallel:
       - Distribute weight updates across multiple cores using parallel processing
       - Normalize updated sample weights concurrently
     - Store the weak classifier `ht(xi)` and its parameters (`αt`, split feature, split value) in `hlist`

4. **Combining Weak Classifiers:**
   - For each sample in the test dataset `X_test`:
     - Calculate the weighted sum of predictions from all weak classifiers:
       - H(xi) = ∑ (αt * ht(xi))
     - Predict the class label based on the sign of `H(xi)`

5. **Evaluation:**
   - Evaluate the model's accuracy using the combined weak classifiers on the test dataset.
