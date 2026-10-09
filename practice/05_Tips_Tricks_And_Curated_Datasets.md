# 💡 Pro Tips, Tricks & Curated Next-Level Datasets

> **"Mastery doesn't come from memorizing syntax; it comes from learning idioms, debugging instincts, and knowing where to find rich, real-world problems to solve."**

This guide provides battle-tested **pro tips & tricks** across NumPy, Pandas, Matplotlib, and Seaborn, followed by **10 curated real-world datasets** you can download today to build your portfolio.

---

## ⚡ Part 1: Pro Tips & Tricks for the Data Analysis Stack

### 🐼 Pandas Pro Tips

#### 1. Avoid the Dreaded `SettingWithCopyWarning`
```python
# ❌ INCORRECT (Chained indexing triggers a warning and may fail silently):
df[df['sales'] > 1000]['discount'] = 0.2

#  CORRECT (Explicitly use .loc with [row_indexer, col_indexer]):
df.loc[df['sales'] > 1000, 'discount'] = 0.2

#  WHEN SUBSETTING (Always make an explicit copy):
high_value = df[df['sales'] > 5000].copy()
high_value['priority_flag'] = 1
```

#### 2. Slash Memory by 70–90% with Smart Dtypes
When working with large CSVs, string columns with repetitive values (e.g. Country, Ship Mode, Category) consume massive RAM as `object`.
```python
# Check memory usage per column in megabytes
print(df.memory_usage(deep=True) / 1024**2)

# Convert low-cardinality string columns to category
categorical_cols = ['ship_mode', 'segment', 'market', 'category']
df[categorical_cols] = df[categorical_cols].astype('category')

# Downcast 64-bit integers to 32-bit or 16-bit
df['quantity'] = pd.to_numeric(df['quantity'], downcast='integer')
```

#### 3. Use `.query()` for Readable, Expressive Filtering
Instead of messy boolean masks with parentheses and `&` operators:
```python
# ❌ Hard to read and type:
subset = df[(df['country'] == 'India') & (df['sales'] > 1000) & (df['category'].isin(['Technology', 'Furniture']))]

#  Clean, readable SQL-like syntax:
target_country = 'India'
subset = df.query("country == @target_country and sales > 1000 and category in ['Technology', 'Furniture']")
```

#### 4. Faster Top/Bottom Slicing with `.nlargest()` and `.nsmallest()`
```python
# ❌ Slower: Sorts the ENTIRE 50,000-row DataFrame just to take 5 rows
top_orders = df.sort_values(by='sales', ascending=False).head(5)

#  Faster: Uses a min-heap under the hood without sorting everything
top_orders = df.nlargest(5, 'sales')
lowest_profit = df.nsmallest(5, 'profit')
```

#### 5. Method Chaining for Reproducible Data Pipelines
Wrap your transformations in parentheses to write readable pipelines:
```python
clean_summary = (
    df
    .query("sales > 0 and profit_margin >= -1.0")
    .assign(net_revenue=lambda d: d['sales'] - d['shipping_cost'])
    .groupby(['market', 'category'])
    .agg(
        total_revenue=('net_revenue', 'sum'),
        avg_margin=('profit_margin', 'mean'),
        order_count=('order_id', 'count')
    )
    .reset_index()
    .sort_values('total_revenue', ascending=False)
)
```

#### 6. Instant Percentages with `pd.crosstab`
```python
# Contingency table normalized across rows (index) to see proportions directly:
pd.crosstab(df['category'], df['order_priority'], normalize='index') * 100
```

---

### 🔢 NumPy Pro Tips

#### 1. Clean Multi-Condition Branching with `np.select`
Instead of messy nested `if-else` or chained `np.where`:
```python
conditions = [
    df['sales'] < 100,
    (df['sales'] >= 100) & (df['sales'] < 500),
    (df['sales'] >= 500) & (df['sales'] < 2000),
    df['sales'] >= 2000
]
choices = ['Micro', 'Small', 'Mid-Market', 'Enterprise']

df['deal_size'] = np.select(conditions, choices, default='Unknown')
```

#### 2. Instant Outlier Capping with `np.clip`
```python
# Cap values between the 5th and 95th percentiles without losing rows:
lower_cap, upper_cap = np.percentile(df['shipping_cost'], [5, 95])
df['shipping_cost_capped'] = np.clip(df['shipping_cost'], lower_cap, upper_cap)
```

#### 3. NaN-Safe Vectorized Aggregations
Standard NumPy functions return `nan` if even a single value is missing:
```python
arr = np.array([10.0, 20.0, np.nan, 40.0])

# ❌ np.mean(arr) -> nan
#  Use the nan-safe family:
np.nanmean(arr)        # 23.33
np.nanmedian(arr)      # 20.0
np.nanstd(arr)         # 12.47
np.nanpercentile(arr, 75)
```

---

### 📊 Matplotlib & Seaborn Visual Polish Tips

#### 1. Auto-Format Currency & Percentages on Axes
Never leave raw numbers like `1000000` or `0.025` on axis labels.
```python
import matplotlib.ticker as ticker

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=df, x='category', y='sales', ax=ax, estimator='sum', errorbar=None)

# Format y-axis as Dollar amounts with commas ($1,000,000)
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))

# Or for percentages:
# ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=1))
```

#### 2. One-Line Bar Annotations (`bar_label`)
No more manual coordinate math to label bar charts:
```python
fig, ax = plt.subplots(figsize=(8, 5))
bars = sns.barplot(data=df, x='market', y='sales', ax=ax, estimator='mean', errorbar=None)

# Automatically annotates values on top of every bar:
ax.bar_label(ax.containers[0], fmt='$%.0f', padding=3, fontsize=10, weight='bold')
ax.set_ylim(0, df.groupby('market')['sales'].mean().max() * 1.15)  # Add headroom
sns.despine(top=True, right=True)
plt.show()
```

#### 3. High-Impact Visual Styling: Drop "Chart Junk"
```python
# 1. Clean minimalistic theme
sns.set_theme(style="ticks", font_scale=1.1)

# 2. Remove distracting top and right border spines
sns.despine(top=True, right=True)

# 3. Add subtle, purposeful gridlines only on the value axis
ax.grid(axis='y', linestyle='--', alpha=0.5)
```

#### 4. Add Contextual Storytelling (Reference Lines & Spans)
```python
# Draw target / break-even line:
ax.axhline(0, color='crimson', linestyle='--', linewidth=1.5, label='Break-even Target')

# Highlight a recession or promotion period:
ax.axvspan('2013-06', '2013-09', color='gold', alpha=0.2, label='Summer Promo Campaign')
ax.legend()
```

---

## 🌐 Part 2: 10 Curated Real-World Datasets for Next-Level Practice

When you finish the 4 practice tracks in this repository, here are the **best 10 open-source datasets** to download next, along with the project questions you should solve.

---

### 1. Netflix Movies and TV Shows
* **Domain**: Streaming Media & Entertainment
* **Where to get it**: [Kaggle: Netflix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows) (Search keyword: `netflix movies and tv shows shivamb`)
* **What you practice**: String extraction, text cleaning, genre explosion (`.explode()`), temporal release trends.
* **Project Challenges**:
  1. Parse the `duration` column (e.g., separate `"90 min"` into integer minutes, and `"2 Seasons"` into season counts).
  2. Split multi-country and multi-genre strings into individual rows using `.str.split(', ').explode()`.
  3. Analyze the shift in content strategy: Did Netflix pivot from licensing Movies to producing original TV series after 2015?

---

### 2. NYC Airbnb Open Data
* **Domain**: Real Estate, Hospitality & Pricing
* **Where to get it**: [Inside Airbnb](http://insideairbnb.com/get-the-data/) or [Kaggle NYC Airbnb](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)
* **What you practice**: Geospatial scatter plots, logarithmic price scaling, outlier filtering, neighborhood group comparisons.
* **Project Challenges**:
  1. Price distribution analysis: Address extreme price outliers ($0 to $10,000/night) using IQR capping.
  2. Plot a geospatial scatter plot of Latitude vs Longitude, with points colored by `neighbourhood_group` and size scaled by `price`.
  3. Calculate the estimated occupancy rate and revenue potential per listing based on `minimum_nights` and `number_of_reviews`.

---

### 3. Spotify Top 10,000 Streamed Songs / Audio Features
* **Domain**: Music Streaming & Audio Intelligence
* **Where to get it**: [Kaggle Spotify Dataset](https://www.kaggle.com/datasets/paradisejoy/top-hits-spotify-from-2000-2019)
* **What you practice**: Correlation matrices, pairplots, distributions of normalized metrics (danceability, energy, acousticness, tempo).
* **Project Challenges**:
  1. Create a Pearson correlation heatmap across all audio metrics. Is high energy positively or negatively correlated with acousticness?
  2. Track the evolution of tempo (BPM) and valence (musical cheerfulness) over the decades (2000–2023).
  3. Segment songs into distinct mood quadrants using `sns.scatterplot` (e.g. Danceable & Happy vs Sad & Acoustic).

---

### 4. Instacart Market Basket Analysis
* **Domain**: E-Commerce Grocery Logistics
* **Where to get it**: [Kaggle Instacart Market Basket](https://www.kaggle.com/c/instacart-market-basket-analysis/data)
* **What you practice**: Multi-table relational joins (`pd.merge`), reorder rate analytics, customer habit loops.
* **Project Challenges**:
  1. Merge `orders.csv`, `order_products__prior.csv`, and `products.csv`.
  2. At what hour of the day do most orders occur? On which day of the week?
  3. Calculate the **Reorder Ratio** for each grocery department (Produce vs Snacks vs Beverages). Which items have the highest customer loyalty?

---

### 5. Walmart Store Sales Forecasting
* **Domain**: Retail Supply Chain & Department Analytics
* **Where to get it**: [Kaggle Walmart Recruiting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data)
* **What you practice**: Time series decomposition, holiday impact analysis, markdowns, temperature/economic indicators.
* **Project Challenges**:
  1. How much do holiday weeks (Super Bowl, Labor Day, Thanksgiving, Christmas) boost average weekly sales compared to non-holiday weeks?
  2. Analyze the correlation between `Fuel_Price`, `Unemployment`, and `Weekly_Sales`.
  3. Use `pd.pivot_table()` to compare store sales performance across the 45 distinct retail store branches.

---

### 6. Titanic: Machine Learning from Disaster
* **Domain**: Historical Demographics & Survival Modeling
* **Where to get it**: [Kaggle Titanic](https://www.kaggle.com/c/titanic/data)
* **What you practice**: The quintessential beginner-to-intermediate dataset for survival rate cross-tabulations, title extraction from names, and missing age imputation.
* **Project Challenges**:
  1. Extract passenger honorific titles (`Mr`, `Mrs`, `Miss`, `Master`, `Dr`) from the `Name` column using regex (`.str.extract(' ([A-Za-z]+)\.')`).
  2. Impute missing `Age` values based on the median age of the passenger's honorific title and `Pclass`.
  3. Create a multi-factor survival bar plot comparing Survival % across Gender, Ticket Class (`Pclass`), and Family Size (`SibSp + Parch + 1`).

---

### 7. Credit Card Fraud Detection
* **Domain**: FinTech & Risk Analytics
* **Where to get it**: [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **What you practice**: Extreme class imbalance (0.17% fraud rate), PCA feature interpretation, KDE distributions, precision-recall trade-offs.
* **Project Challenges**:
  1. Quantify the exact class distribution and visualize the imbalance using a log-scaled count plot.
  2. Plot overlaid KDE density distributions (`sns.kdeplot`) comparing fraudulent vs legitimate transactions for key PCA features (`V1` to `V28`).
  3. Determine the transaction amount threshold above which fraud probability increases significantly.

---

### 8. IBM HR Analytics Employee Attrition & Performance
* **Domain**: Human Resources & People Analytics
* **Where to get it**: [Kaggle IBM HR Analytics](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
* **What you practice**: Churn analysis, tenure distribution, work-life balance impact, salary disparity.
* **Project Challenges**:
  1. Calculate the attrition rate across job roles (Sales Rep vs Research Scientist vs Manager).
  2. Is overtime work (`OverTime == 'Yes'`) a statistically significant driver of employee departures?
  3. Visualize the relationship between `MonthlyIncome`, `TotalWorkingYears`, and `Attrition` using boxplots and violin plots.

---

### 9. World Happiness Report (Gallup World Poll)
* **Domain**: Global Socioeconomics & Public Policy
* **Where to get it**: [Kaggle World Happiness Report](https://www.kaggle.com/datasets/unsdsn/world-happiness)
* **What you practice**: Multi-variable linear relationships, global regional comparisons, custom scatter annotations.
* **Project Challenges**:
  1. Which factor has the strongest correlation with the Happiness Ladder Score: GDP per capita, Social Support, or Healthy Life Expectancy?
  2. Highlight the top 10 happiest and bottom 10 unhappiest nations with a horizontal lollipop or bar chart.
  3. Identify "outlier nations"—countries with low GDP per capita but surprisingly high happiness ratings.

---

### 10. Bike Sharing Demand (Capital Bikeshare / Washington D.C.)
* **Domain**: Urban Mobility & Smart City Analytics
* **Where to get it**: [Kaggle Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand/data)
* **What you practice**: Hourly time series, weather condition impacts (temperature, humidity, windspeed), casual vs registered user dynamics.
* **Project Challenges**:
  1. Compare the hourly usage curve of **Registered Commuters** (sharp rush hour peaks at 8 AM and 5 PM) vs **Casual Tourists** (smooth bell curve peaking at 2 PM).
  2. Plot rental counts against the "feels-like" temperature (`atemp`) and humidity using Seaborn facet grids.
  3. Determine the drop in ridership during adverse weather (rain/snow vs clear skies).

---

## 🎯 Suggested Next Steps
1. Complete the exercises in **[01_Dataset_01_Customer_Purchases.md](file:///mnt/UBUNTU_DATA/Coding/DATA_ANALYSIS/practice/01_Dataset_01_Customer_Purchases.md)** and **[03_Dataset_03_Global_Superstore_EDA.md](file:///mnt/UBUNTU_DATA/Coding/DATA_ANALYSIS/practice/03_Dataset_03_Global_Superstore_EDA.md)**.
2. Download one external dataset that genuinely excites you (e.g., Netflix for entertainment, Airbnb for travel/housing, or Spotify for music).
3. Build a standalone, well-documented Jupyter notebook portfolio project following the **6-Step Universal Workflow**.
