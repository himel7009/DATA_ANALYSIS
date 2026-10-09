# 🎯 Practice Problem Set 02: Data Quality Audit & Feature Engineering Pipeline
**Dataset**: `Dataset_02.csv`  
**Scenario**: You are a Data Analyst / Machine Learning Engineer preparing an anonymized feature dataset for downstream predictive modeling. The raw data comes from multiple logging systems and contains missing records, extreme outliers, non-standard scales, skewed distributions, and unencoded categorical labels. Your goal is to clean, transform, and structure this data into a modeling-ready state.

---

### Part 1: Data Hygiene Audit & Missingness Diagnostics
1. Determine the total record count and the total number of features in the raw dataset.
2. Generate an audit report listing each feature, its current storage data type, the count of missing entries, and the exact percentage of missing data.
3. Rank the features in descending order of missingness. Identify which features have more than 30% of their data missing.
4. Determine how many rows in the dataset are completely intact (i.e. containing zero missing values across all 20 features). What percentage of the overall dataset would be lost if you simply dropped all rows containing nulls?
5. Construct a visual diagnostic map showing the pattern of missing data across all records and features. Do certain features share missingness patterns simultaneously?

---

### Part 2: Missing Data Strategy & Imputation
6. Analyze the distribution and symmetry of continuous numerical features that contain missing values. Identify which features are heavily skewed and which follow a roughly symmetric distribution.
7. Apply an appropriate central tendency measure to impute missing values in continuous numerical columns based on their skewness (mean for symmetric data, median for skewed data).
8. For binary and nominal categorical features that contain missing entries, determine the most frequent category and impute missing records accordingly.
9. For categorical features where missingness itself might represent a distinct state or unrecorded condition, create an explicit category label to represent missing records rather than discarding them.
10. Implement conditional group-based imputation: impute missing values in numerical attributes using the median value of the specific subcategory to which each record belongs.
11. Validate that the dataset no longer contains any unhandled missing values across your targeted attributes.

---

### Part 3: Statistical Outlier Detection & Treatment
12. Inspect the continuous numerical features to identify the presence of anomalous extreme values that could distort analytical averages or model performance.
13. Visually examine the spread, median, and outlier points for the continuous features using quartile-based box plots.
14. Calculate the 25th percentile, 75th percentile, and Interquartile Range for each continuous feature using numerical array calculations.
15. Establish statistical upper and lower outlier fence boundaries for the numerical features and quantify exactly how many data points fall beyond these fences.
16. Implement a winsorization (capping) strategy: cap extreme upper and lower outliers at chosen percentile thresholds (e.g. 1st and 99th percentiles) so that extreme spikes are curbed without discarding legitimate rows.
17. Compare the distributions of the features before and after outlier treatment to verify that the extreme tails have been stabilized while preserving the core shape of the data.
18. Compute standard Z-scores for numerical attributes and identify records with values deviating more than 3 standard deviations from the mean.

---

### Part 4: Distribution Skewness & Mathematical Transformations
19. Quantify the coefficient of skewness across all continuous numerical attributes. Identify which features exhibit strong right-tail skewness.
20. Apply an appropriate mathematical transformation (such as a logarithmic transformation) to compress the right tail and reduce the skewness of heavily skewed positive features.
21. Re-evaluate the skewness score of the transformed features to measure how much closer they moved toward a normal distribution.
22. Implement Min-Max Normalization from scratch to rescale numerical features into a uniform bounding interval between 0.0 and 1.0.
23. Implement Z-Score Standardization from scratch to transform numerical features so they possess a mean of zero and a standard deviation of one.
24. Verify numerically that the standardized features have achieved a zero mean and unit variance.

---

### Part 5: Categorical Encoding & Dimensionality
25. Identify all categorical features in the dataset and determine the cardinality (number of distinct unique values) of each.
26. Convert binary text features (e.g. Yes/No responses) into numeric indicators (1 and 0).
27. For categorical features that possess a natural hierarchical ranking (e.g. Low, Medium, High), apply an ordinal integer encoding that preserves their progression.
28. Convert multi-class nominal categorical features into binary dummy variables, ensuring that collinear reference categories are dropped to prevent the dummy variable trap.
29. Create a derived binary feature that flags whether an integer count feature exceeds an established operational threshold.

---

### Part 6: Correlation Analysis & Feature Diagnostics
30. Calculate the full Pearson correlation matrix across all numerical attributes in the cleaned dataset.
31. Identify pairs of features that exhibit strong positive or strong negative linear relationships.
32. Display the correlation matrix as a clean, formatted heatmap with numerical values omitted or annotated clearly.
33. Mask the redundant upper triangle of the correlation matrix to produce a publication-style diagonal correlation view.
34. Identify any feature pairs showing severe multicollinearity that could cause redundancy in downstream modeling.
35. Create pairwise relationship scatter plots across selected correlated features, using categorical features to color-code the observations.
36. Conduct a final verification check confirming that your processed dataset has zero null entries, fully numeric attributes, and finite values.
