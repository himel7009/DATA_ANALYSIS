# 🎯 Practice Problem Set 01: Customer Behavior & Demographic Analysis
**Dataset**: `Dataset_01.csv`  
**Scenario**: You are a Data Analyst at a regional retail company operating across South Asia. The leadership team wants to understand customer demographics, spending patterns, product preferences, and satisfaction levels to optimize marketing spend and improve customer retention.

---

### Part 1: Initial Data Audit & Health Check
1. Determine the exact dimensions of the dataset (total records and attributes).
2. Inspect the data types of all attributes and verify whether any fields contain missing values.
3. Check whether there are duplicate records representing repeated transactions or customer IDs.
4. Generate the full five-number statistical summary for customer age, purchase amounts, and ratings.
5. Identify all distinct countries represented in the customer base and determine how many customers reside in each country.
6. Calculate the proportion of male versus female customers across the entire platform.

---

### Part 2: Customer Profiling & Demographic Insights
7. Segment the customer base into distinct age brackets (Youth: under 25, Young Adults: 25–35, Middle-Aged: 36–50, Seniors: 51+) and calculate total spending and average customer satisfaction for each bracket.
8. Which age group has the highest purchasing power?
9. Compare the average spending behavior between male and female customers within each country. Does one gender consistently spend more?
10. Identify the top 10 highest-value customers across the platform. What country and product category do they belong to?
11. Find all customers older than 45 who spent more than the platform-wide average purchase amount.
12. Determine whether customer ratings vary significantly across different countries. Which country reports the highest average satisfaction? Which reports the lowest?

---

### Part 3: Product Category & Commercial Performance
13. Rank the product categories from highest to lowest by total revenue generated.
14. Which product category records the highest transaction volume (number of purchases)?
15. Calculate the average purchase price per item within each product category.
16. Identify the product category with the lowest average satisfaction rating. Is this category also generating high or low revenue?
17. Determine which product categories are most popular among customers under 30 years old compared to customers over 50 years old.
18. Find transactions where the customer gave a rating below 2.0 but spent above the 75th percentile of purchase amounts. What products are these high-paying, dissatisfied customers buying?

---

### Part 4: Onboarding Trends & Time-Based Dynamics
19. Convert the customer join dates into usable temporal dimensions to extract the onboarding year, month, and day of the week.
20. Calculate the total number of new customer signups per year. Has customer acquisition increased or decreased year-over-year?
21. Which calendar month historically attracts the highest volume of new customer registrations?
22. Do customers who register on weekends spend more on average than customers who register on weekdays?

---

### Part 5: Statistical Distribution & Outlier Analysis
23. Calculate the mean, median, variance, and standard deviation of purchase amounts using array-based numerical calculations.
24. Determine whether purchase amount is normally distributed, right-skewed, or left-skewed by comparing its mean and median.
25. Calculate the 10th, 25th, 50th, 75th, 90th, and 99th percentiles of customer spending.
26. Determine the Interquartile Range (IQR) for purchase amount and detect any statistically anomalous transactions that fall outside the upper and lower threshold boundaries.
27. Normalize the purchase amount attribute to a continuous scale between 0.0 and 1.0 to prepare the metric for a standardized scoring model.
28. Create a customer spending tier classification: label transactions as "Bronze" (bottom 25%), "Silver" (middle 50%), or "Gold" (top 25%).

---

### Part 6: Visual Storytelling & Executive Reporting
29. Plot the distribution of purchase amounts with a density curve, highlighting the mean and median markers on the plot.
30. Create a bar chart comparing total revenue across all product categories, ordered from largest to smallest.
31. Generate a side-by-side demographic breakdown showing customer count by country split by gender.
32. Construct a box plot displaying the spread and median purchase amounts for each product category to identify categories with high price variance.
33. Plot customer age against purchase amount on a scatter plot, using visual styling to distinguish between male and female customers. Is there a visible correlation between age and spending?
34. Compute the correlation matrix across all numeric features (Age, Purchase Amount, Rating) and display it as an annotated heatmap.
35. Create a monthly customer acquisition line chart showing how signups have trended over the entire recorded timeline.
