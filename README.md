# MSCS_634_ProjectDeliverable_3
This Repo is Residency Day 2: Project Deliverable 2: Project Project Deliverable 3: Classification, Clustering, and Pattern Mining MSCS-634-M20

# Deliverable 3: Clustering and Association Rule Mining
  
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

# 3. Clustering Analysis (K-Means)

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

# 4. Association Rule Mining (Apriori Algorithm)

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

# 5. Real-World Applications

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

# 6. Challenges Encountered

- High dimensionality after one-hot encoding (2400+ features)
- Empty rule outputs due to strict thresholds
- Required tuning of support and confidence parameters
- Computational complexity of Apriori algorithm
- Managing scaling for clustering accuracy

Careful preprocessing and parameter tuning were necessary to extract meaningful patterns.

---

# 7. Key Insights

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
