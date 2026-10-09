# 🎯 Practice Problem Set 03: Global Commercial Performance & Profitability Audit
**Dataset**: `Dataset_03.csv`  
**Scenario**: You are a Commercial Data Analyst at Global Superstore, an international e-commerce corporation fulfilling orders across 147 countries. Executive leadership is concerned that despite growing sales volumes, overall net profits are lagging. You are tasked with auditing the entire transactional database to resolve data formatting defects, diagnose financial leakages, uncover unprofitable product lines, evaluate discounting policies, and analyze global supply chain lead times.

---

### Part 1: Data Integrity & Field Rectification
1. Ingest the raw transactions and inspect the total order count and column schema.
2. Investigate the revenue field to determine why it was imported as text rather than a floating-point numeric value.
3. Clean and convert the revenue column so that it represents a valid numerical metric, and verify that all currency amounts are computable.
4. Convert the order dates and shipping dates into standard calendar timestamps.
5. Derive an operational lead-time metric representing the total shipping duration in days from order placement to final dispatch.
6. Verify whether any transactions contain negative or impossible delivery durations (orders dispatched before being placed).
7. Extract the transaction year, month, and day-of-week into dedicated dimensions to enable time-series slicing.

---

### Part 2: Macro Financial Health & Commercial Metrics
8. Calculate the cumulative gross sales, cumulative net profit, and cumulative shipping freight expenses across the company's recorded history.
9. Determine the blended company-wide profit margin percentage.
10. Calculate the individual profit margin for every single transaction, properly accounting for any orders with zero or negligible revenue.
11. Determine the proportion and count of all transactions that resulted in a net financial loss. What percentage of the company's order volume loses money?
12. Calculate the total aggregate dollar amount lost across all unprofitable transactions.
13. Calculate the unit selling price for every order line item based on total revenue and quantity ordered.
14. Identify the top 10 single largest revenue-generating transactions in company history. Did these mega-orders generate healthy profit margins or steep losses?
15. Uncover the 10 single most disastrous transactions in terms of total dollar loss. What products, countries, and discount levels were involved?

---

### Part 3: Geographic & Market Diagnostics
16. Determine how many unique countries, states, and global regional markets the business operates in.
17. Aggregate sales, net profit, average profit margin, and average shipping cost across each global market.
18. Rank the global markets from most profitable to least profitable. Which market serves as the company's primary profit engine? Which market has the thinnest profit margin?
19. Identify the top 5 highest-revenue countries and the top 5 most profitable countries. Are they the same?
20. Identify the 5 countries generating the largest total cumulative financial losses. How much money is the company losing in these territories?
21. Break down financial performance by regional territories within each major market to pinpoint specific regional loss centers.

---

### Part 4: Category, Sub-Category & Product Portfolio Audit
22. Calculate total sales revenue, total profit, and overall profit margin for each of the primary product categories.
23. Break down performance to the sub-category level. Rank all sub-categories by total net profit.
24. Identify any sub-category that produces high sales revenue but generates a net cumulative financial loss for the corporation (e.g. Tables or Bookcases).
25. Find the top 10 best-selling individual products by unit volume.
26. Construct a cross-tabulated revenue matrix showing sales distributed across product categories and customer segments (Consumer, Corporate, Home Office).
27. Identify which customer segment is the most lucrative across each category.

---

### Part 5: Discount Cannibalization & Operational Dynamics
28. Group transactions into discrete discount tiers (e.g. No Discount, Minor Discount: 1–20%, Moderate Discount: 21–50%, Steep Discount: >50%).
29. Calculate the average net profit and average profit margin across each discount tier.
30. Determine the exact discount threshold where average transaction profitability turns negative (the "discount cliff").
31. Compare the delivery lead times across the different shipping modes (Standard Class, Second Class, First Class, Same Day).
32. Check whether "Same Day" orders are fulfilled and dispatched on the exact date of order placement. What is the real average fulfillment duration for Same Day orders?
33. Identify transactions where shipping freight costs account for more than 40% of the item's total sales price.

---

### Part 6: Seasonality, Growth Trends & Visual Dashboards
34. Aggregate sales and profit into monthly intervals across the multi-year timeline to track commercial trajectory.
35. Calculate the quarter-over-quarter percentage revenue growth rate.
36. Identify seasonal patterns across the calendar year: which month consistently experiences the annual sales peak? Is there evidence of a Q4 holiday surge?
37. Construct a multi-panel visual report containing:
   * A bar chart showing total revenue by global market, ordered by size.
   * A horizontal bar chart of net profit across all sub-categories, visually distinguishing profitable categories from loss-making categories.
   * A time-series trend line plotting monthly revenue and profit over the entire recorded timeline.
   * A scatter plot illustrating the relationship between applied discount percentage and transaction profit margin, annotated with a break-even reference line.
38. Perform a customer revenue concentration analysis (Pareto distribution): determine what percentage of total company revenue is generated by the top 20% of customers.
