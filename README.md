# MSCS_634_ProjectDeliverable_3
This Repo is Residency Day 2: Project Deliverable 3: Classification, Clustering, and Pattern Mining MSCS-634-M20

# Deliverable 3: Classification, Clustering, and Pattern Mining
  
**Group Members:**  
Avijit Saha  
Pranoj Thapa  
Sandip KC  
Bharath Singareddy  

**Course:** Advanced Big Data and Data Mining (MSCS-634-M20)   
Dr. Satish Penmatsa

**February 14, 2026**



---

## Overview

Deliverable 3 focuses on applying **unsupervised learning techniques** to discover hidden patterns in the dataset. Unlike Deliverables 1 and 2, which focused on preprocessing and regression modeling, this deliverable emphasizes:

- K-Means Clustering
- Association Rule Mining (Apriori Algorithm)
- Pattern Interpretation
- Real-World Applications

The goal is to extract meaningful insights without using a predefined target variable.

---

# 1. Data Preparation

Before applying unsupervised learning techniques, the dataset was cleaned and transformed.

### Steps Performed:

- Removed missing values
- Handled duplicate and inconsistent columns
- Converted boolean variables to integers
- Applied One-Hot Encoding to categorical variables
- Scaled numerical features using `StandardScaler` (for clustering)
- Created transactional (binary) dataset for association rules

Two datasets were prepared:

| Dataset Type | Purpose |
|-------------|----------|
| Scaled Numerical Dataset | Used for K-Means Clustering |
| One-Hot Encoded Dataset | Used for Apriori Association Rule Mining |

---

# 2. Classification Models:
# 2.1. Decision Tree:
## Objective

- To classify passenger travel data into categories (Low, Medium, High).
- To analyze travel patterns using machine learning.
- To evaluate the performance of the classification model.

## Algorithm Used

- Decision Tree Classifier.
- GridSearchCV for hyperparameter tuning.
- Evaluation using accuracy, F1-score, confusion matrix, and ROC curve.

## Process

- Collected passenger travel data.
- Cleaned and removed missing or invalid values.
- Selected important features (Year, Mode, Statistic).
- Converted categorical data into numerical form.
- Divided data into training and testing sets.
- Trained the Decision Tree model.
- Tuned model parameters.
- Tested and evaluated the model.
- Visualized results using plots and tree diagrams.

## Observations

- The model classified High and Low values accurately.
- Medium values were harder to predict.
- Hyperparameter tuning improved accuracy.
- Most errors occurred between neighboring classes.
- Year and Mode were important features.
- ROC curve showed good classification ability.

## Interpretation

- Decision Tree is effective for analyzing travel data.
- The model helps understand travel behavior patterns.
- Tuning reduced overfitting and improved performance.
- Results are reliable for prediction and analysis.
- The model can support transportation planning decisions.

---

# 2.2. K-Nearest Neighbours

## Objective

- To classify passenger travel data into Low, Medium, and High categories using KNN.
- To compare KNN performance with other classification models.
- To analyze travel patterns based on similarity between data points.

## Algorithm Used

- K-Nearest Neighbors (KNN) Classifier.
- Distance metrics such as Euclidean distance.
- Cross-validation for selecting the best value of K.
- Performance evaluation using accuracy, F1-score, confusion matrix, and ROC curve.

## Process

- Collected and cleaned the dataset.
- Selected important features (Year, Mode, Statistic).
- Converted categorical variables into numeric form.
- Normalized/standardized data for fair distance calculation.
- Split data into training and testing sets.
- Chose optimal K value using validation.
- Trained the KNN model.
- Predicted test data classes.
- Evaluated model performance.

## Observations

- KNN performed well when K value was properly selected.
- Model accuracy decreased for very small or very large K values.
- Sensitive to noise and outliers.
- Performance depends on data scaling.
- Medium category had higher misclassification.
- Requires more computation for large datasets.

## Interpretation

- KNN is simple and easy to understand.
- Works well when similar patterns exist in data.
- Proper scaling and K selection are essential.
- Less suitable for very large datasets.
- Useful as a comparison model for validation.

---
# 2.3. Naive Bayes:

## Objective

- To classify passenger travel data into Low, Medium, and High categories using Naive Bayes.
- To analyze travel patterns based on probability.
- To compare Naive Bayes performance with other classification models.

## Algorithm Used

- Naive Bayes Classifier (Gaussian Naive Bayes).
- Based on Bayes’ Theorem with independence assumption.
- Uses probability distributions for prediction.
- Performance evaluated using accuracy, F1-score, confusion matrix, and ROC curve.

## Process

- Collected and cleaned the dataset.
- Selected important features (Year, Mode, Statistic).
- Converted categorical variables into numeric form.
- Split data into training and testing sets.
- Trained the Naive Bayes model.
- Calculated class probabilities.
- Predicted class labels.
- Evaluated model performance.

## Observations

- Fast training and prediction.
- Works well with large datasets.
- Performs better with independent features.
- Less sensitive to missing values.
- Medium category showed moderate misclassification.
- Accuracy slightly lower compared to Decision Tree and KNN.

## Interpretation

- Naive Bayes is simple and efficient.
- Suitable for quick baseline models.
- Performs well despite strong assumptions.
- Best used when features are mostly independent.
- Useful for comparison and initial analysis.

---

# 3. Hyperparameter Tuning of Decision Tree:

## Objective

- To improve the accuracy of the Decision Tree model.
- To reduce overfitting and underfitting.
- To find the best model settings.
- To increase prediction reliability.

## Algorithm Used

- Decision Tree Classifier.
- GridSearchCV for hyperparameter tuning.
- Cross-validation for performance evaluation.
- Gini Index / Entropy for split quality.

## Process

- Prepared and cleaned the dataset.
- Selected important features.
- Split data into training and testing sets.
- Chose hyperparameters (max_depth, min_samples_split, etc.).
- Applied GridSearchCV.
- Trained multiple models.
- Selected best parameter combination.
- Tested final model on unseen data.

## Observations

- Tuned model showed higher accuracy.
- Overfitting was reduced.
- Tree structure became more balanced.
- Prediction performance improved.
- Training time increased slightly.

## Interpretation

- Proper tuning improves model performance.
- Helps in building stable models.
- Prevents unnecessary tree growth.
- Makes predictions more reliable.
- Essential for real-world applications.

---
# 4. classification model performance using:
# 4.1. Confusion Matrix
## Objective

- To analyze correct and incorrect predictions.
- To understand types of classification errors.
- To evaluate model reliability.

## Algorithm Used

- Confusion Matrix.
- TP, TN, FP, FN metrics.

## Process

- Generated predictions from the model.
- Compared predicted labels with actual labels.
- Created confusion matrix.
- Identified TP, TN, FP, FN values.

## Observations

- Most predictions were correctly classified.
- Some misclassifications were observed.
- False positives and false negatives were minimal.
- Class-wise performance was visible.

## Interpretation

- Shows detailed model performance.
- Helps identify weak classes.
- Useful for improving model accuracy.
- Important for error analysis.
---

---

# 4.2 ROC Curve
## Objective

- To measure the model’s ability to separate classes.
- To evaluate performance at different thresholds.
- To analyze classification quality.

## Algorithm Used

- ROC Curve.
- AUC (Area Under Curve).

## Process

- Calculated probability scores.
- Computed True Positive Rate and False Positive Rate.
- Plotted ROC curve.
- Calculated AUC value.

## Observations

- ROC curve stayed closer to top-left corner.
- High AUC value observed.
- Good class separation achieved.
- Stable performance across thresholds.

## Interpretation

- Higher AUC means better performance.
- Shows model’s discrimination ability.
- Less affected by class imbalance.
- Useful for model comparison.
---

---

## 4.3 Accuracy / F1 Score
## Objective

- To measure overall prediction accuracy.
- To balance precision and recall.
- To evaluate model effectiveness.

## Algorithm Used

- Accuracy Score.
- Precision.
- Recall.
- F1 Score.

## Process

- Compared predicted and actual labels.
- Calculated accuracy value.
- Computed precision and recall.
- Calculated F1 score.

## Observations

- Accuracy was reasonably high.
- F1 score showed balanced performance.
- Precision and recall were stable.
- Suitable for dataset characteristics.

## Interpretation

- Accuracy is good for balanced data.
- F1 score is better for imbalanced data.
- Combined metrics give reliable evaluation.
- Helps in selecting best model.
---
# 5. Clustering Analysis (K-Means)

## Objective

To group similar observations into clusters based on feature similarity.

## Algorithm Used

- **K-Means Clustering**
- Distance Metric: Euclidean Distance
- Optimal K selected using the **Elbow Method**

## Process

1. Scaled numerical features
2. Computed inertia values for k = 1 to 9
3. Plotted Elbow Curve
4. Selected optimal number of clusters
5. Trained final K-Means model
6. Assigned cluster labels to dataset

## Observations

- The Elbow Method identified the optimal number of clusters.
- Clusters grouped similar transportation-related characteristics.
- Data segmentation revealed structural similarities within the dataset.

## Interpretation

Clustering helped identify:

- Transportation behavior groupings
- Similar safety and emission characteristics
- Natural data segmentation patterns

---

---
# 6. Association Rule Mining (Apriori Algorithm)

## Objective

To discover relationships between frequently occurring transportation features.

## Algorithm Used

- **Apriori Algorithm**
- Minimum Support Threshold: Adjusted for meaningful patterns
- Rule Metric: Confidence
- Ranking Metric: Lift

## Process

1. Converted dataset into one-hot encoded transactional format
2. Applied Apriori to find frequent itemsets
3. Generated association rules
4. Filtered rules based on confidence threshold
5. Sorted rules by lift

## Results

- **26 frequent itemsets** were discovered.
- Rules were generated after adjusting support and confidence thresholds.
- Lift was used to identify strong positive associations.

### Example Frequent Itemsets:

- `Mode_Air`
- `Mode_Females`
- `Mode_Males`
- `Mode_CO2 GHG Emissions by Mode`
- `Mode_Injured Persons`

## Rule Interpretation

Association rules revealed:

- Frequently co-occurring transportation attributes
- Links between demographic factors and transportation modes
- Relationships between emissions, fuel efficiency, and safety statistics

Lift values greater than 1 indicated meaningful positive relationships.

---

# 7. Real-World Applications

The discovered patterns can be applied in several practical contexts:

###  Transportation Planning
Clustering helps identify usage patterns for infrastructure optimization.

###  Safety Improvements
Rules linking injury/fatality rates to transportation modes can guide safety policies.

###  Environmental Strategy
Emission-related associations support sustainability initiatives.

###  Demographic Analysis
Patterns involving gender-based transportation usage help create inclusive policies.

###  Operational Optimization
Transportation agencies can use insights to:
- Forecast demand
- Allocate resources efficiently
- Improve service quality

---

# 8. Challenges Encountered

- High dimensionality after one-hot encoding (2400+ features)
- Empty rule outputs due to strict thresholds
- Required tuning of support and confidence parameters
- Computational complexity of Apriori algorithm
- Managing scaling for clustering accuracy

Careful preprocessing and parameter tuning were necessary to extract meaningful patterns.

---

# 9. Key Insights

- Unsupervised learning revealed hidden structural patterns.
- Clustering identified natural groupings in transportation data.
- Association rules uncovered meaningful co-occurrence relationships.
- Feature engineering and preprocessing significantly influenced results.
- Parameter tuning directly impacted pattern discovery.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn (`KMeans`, `StandardScaler`)
- mlxtend (`apriori`, `association_rules`)
- Matplotlib

---

# Conclusion

Deliverable 3 successfully applied:

-  K-Means Clustering  
-  Apriori Association Rule Mining  

The analysis uncovered meaningful hidden patterns within the transportation dataset. These insights demonstrate the effectiveness of unsupervised learning techniques in extracting actionable knowledge from complex real-world data.

This deliverable highlights how clustering and association rule mining transform raw data into structured insights for informed decision-making.
