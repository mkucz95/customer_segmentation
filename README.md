# Identifying Customer Segments
#### Udacity Data Science - Project 3

In this project, I work with real-life data provided to by Bertelsmann partners AZ Direct and Arvato Finance Solution. The data here concerns a company that performs mail-order sales in Germany. Their main question of interest is to identify facets of the population that are most likely to be purchasers of their products for a mailout campaign. I use unsupervised learning techniques to organize the general population into clusters, then use those clusters to see which of them comprise the main user base for the company. Prior to applying the machine learning methods, I assess and clean the data in order to convert the data into a usable form.


The unsupervised learning branch of machine learning is key in the organization of large and complex datasets. While unsupervised learning lies in contrast to supervised learning in the fact that unsupervised learning lacks objective output classes or values, it can still be important in converting the data into a form that can be used in a supervised learning task. Dimensionality reduction techniques can help surface the main signals and associations in your data, providing supervised learning techniques a more focused set of features upon which to apply their work. Clustering techniques are useful for understanding how the data points themselves are organized. These clusters might themselves be a useful feature in a directed supervised learning task. This project will give you hands-on experience with a real-life task that makes use of these techniques, focusing on the unsupervised work that goes into understanding a dataset.

In addition, the dataset presented in this project requires a number of assessment and cleaning steps before you can apply your machine learning methods.


## Write-Up For Project

### 1.1 Assessing Missing Data in Each Column
There were some columns with significant amounts of missing data. Out of the total of 891,221 observations, 6 of the features had more than 200,000 missing values. I decided that these were the outliers since close to 25% of the data was missing. Most of these were supposed to hold information on a personal or household level. Possibly people were uncomfortable disclosing this kind of information. The columns excluded from the dataset are shown above.

It is also interesting to note that many columns had identical numbers of missing values. The most common was 116515 missing values which occurred 7 time, including the following key/value paris of (#NaN,#occurences): (4854, 6), (133324, 6), (73499, 4), (93148, 4), (111196, 3), (99352, 3), (93740, 3), (77792, 2), (158064, 2), (97375, 2). Since so many features had identical numbers of missing values it is most likely that it was the same entries (rows) that didn't fill in particular information (perhaps because it was sensitive). This might also mean that these columns contain similar data and it might not be necessary to keep all the features in this case.

For example, the 7 features that all have 116515 are all from the `PLZ8` macro_cell features. This is all to do with information regarding the building types and family houses of the region a person lives in. Maybe all the people with mssing data live in the same region for which there is no data.

The features that each have 4845 missing values mostly come from different feature groupings/categories: `CJT_GESAMTTYP`, `GFK_URLAUBERTYP`, `RETOURTYP_BK_S`, `ONLINE_AFFINITAET`, apart from `LP_STATUS_FEIN` and `LP_STATUS_GROB` which are the same feature on a fine and rough scale. Since they are mostly different features describing different things maybe certain people here didn't feel comfortable disclosing this information.

This means that there are feasibly two reasons columns would have identical numer of NaN's:

they have similar meaning and if it is hard to get data on one then it is hard to get data for all features in that category (like RR3 micro-cell)

certain people didn't feel comfortable disclosing certain information and these people are likely to be more withdrawn in answering among various features.

![Missing Values in Features](https://github.com/mkucz95/customer_segmentation/blob/master/nan_cols.png)


### 1.2 Assess Missing Data in Each Row
I looked at 6 features where the data had no missing values, and quite interestingly some of these features had even distributions of values between the two datasets, and some were quite uneven.

The highest difference in distributions was seen in the columns:

- `FINANZ_ANLEGER`
- `FINANZ_SPARER`
- `FINANZ_VORSORGER`
This means that 3 out of the 6 features I looked at have a very different distribution of values between the two splits of data- NaN heavy data and NaN light data. This means that it might not be the best idea to drop a lot of the NaN datum, as it could disort the data. We should revist the high NaN rows later as some seem to be qualitatively different.

![Missing Values in Rows](https://github.com/mkucz95/customer_segmentation/blob/master/nan_rows.png)

![Distribution of Values Between Features](https://github.com/mkucz95/customer_segmentation/blob/master/nan_dist.png)

### 1.3 Re-Encode Categorical Features
- The non-numerical binary variable is: `OST_WEST_KZ`
- `CAMEO_DEU_2015` and `CAMEO_DEUG_2015` are a multi-level categorical: alphanumeric
- `SOHO_KZ` and `GREEN_AVANTGARDE` is a regular binary variable
- `ANREDE_KZ` is binary, but takes values `[1,2]`


The variables that need to be re-encoded are:
- non-numerical binary var: `OST_WEST_KZ` will be converted to regular binary
- all multi-level categoricals will be one hot encoded
- The variables I dropped are:
      > 1. `KK_KUNDENTYP`
      > 2. `TITEL_KZ`
      > 3. `AGER_TYP`
      
I dropped them because they had very few (relative to length of dataset) non-missing values. Which wouldn't be useful for analysis anyway.

**Engineering Steps**
For alphanumeric binary var `OST_WEST_KZ`:

converted `OST_WEST_KZ` to binary variable: `WEST_KZ`
W maps to 1 (signifies True/1 for "west")
O maps to 0
For categorical variables: `['CJT_GESAMTTYP', 'FINANZTYP','GFK_URLAUBERTYP', 'LP_FAMILIE_FEIN', 'LP_FAMILIE_GROB', 'LP_STATUS_FEIN', 'LP_STATUS_GROB', 'NATIONALITAET_KZ', 'SHOPPER_TYP', 'ZABEOTYP', 'GEBAEUDETYP', 'CAMEO_DEUG_2015', 'CAMEO_DEU_2015']`

First, I imputed each variable column to prevent getting a one hot encoded column for NaN - missing values.

to do this I find the highest frequency value in each common (most likely category), and then fill the missing values with that instead of a mean for example. The mean would be a bad choice to impute categorical variables as it makes the variable continuous rather than discrete.

### 1.4 Engineer Mixed-Type Features
Engineering `PRAEGENDE_JUGENDJAHRE`
created mapping from the current variable to a categorical `YOUTH_DECADE` variable denotes which decade the persons childhood was in. for example 60s became 60 created mapping from the current variable to a binary `AVANTGARDE` variable if the predominant movement during this persons childhood was avantgarde, the binary var denotes 1.0 otherwise it denotes 0.0

**Engineering  CAMEO_INTL_2015**
first defined two mapping functions: decode_wealth and decode_life

- `decode_wealth()` returns the first digit of the orginial variable which shows wealth status
- `decode_life()` returns the second digit of the original variable showing life stage
I applied this function to the data set and created a new ordinal variable for each wealth and life stage
-- `CAMEO_INTL_2015_WEALTH`
-- `CAMEO_INTL_2015_LIFE`

***Other mixed variables***
the remaining mixed variables that were still in data set after engineering step were: `['LP_LEBENSPHASE_FEIN', 'LP_LEBENSPHASE_GROB', 'WOHNLAGE','PLZ8_BAUMAX']`

- `LP_LEBENSPHASE_FEIN` this variable essentially encodes the same things as `CAMEO_INTL_2015`, so I decided to drop it
- `LP_LEBENSPHASE_GROB` this variable is the same as the variable above but on rougher scale, dropping as well
-`CAMEO_DEU_2015` contains similar but more defined information. `WOHNLAGE` is encoded as 1: rural, 0: not-rural in variable `RURAL`
- `PLZ8_BAUMAX` this variable is encapsulated by the information in other PLZ8 variables, is dropped.

Secondly, I used pandas.get_dummies to one hot encode each column, dropped the original column from the dataset, and then added the new encoded data columns to the dataset.

### 2.1 Apply Feature Scaling
I first had to make sure that there were no null values in the data set. Most of the data had very few nulls left. Only two features had more than 20,000 NaN, which is only about 5% of all the values.

I decided that the best imputation method was mode instead of mean. Most of the values are categorical, and even some of the numeric values like birth year wouldnt make sense if they were continuous. Therefore, I imputed using most_frequent
I used the `StandardScaler()` provided by `sklearn.preprocessing` to scale all the values to mean 0 and stdDev = 1. This was suggested in the writeup.


### 2.2 Perform Dimensionality Reduction
The Principal Component Analysis with all of the Principal Components showed that the first 6 Principal Components each explain more than 3% of the total variance, with the highest explained variance for a single Principal Component being ~8%. The first 15 Principal Components explain 40% of all the variance. I am retaining 75 of the 185 Prinicpal Components for the analysis, as they explain about 75% of the total variance.

It is interesting to note that the last 50 components (#125-175) explain very little of the variance.

![PCA Variance Explanation](https://github.com/mkucz95/customer_segmentation/blob/master/pca_var.png)

### 2.3: Interpret Principal Components
Each principal component is a unit vector that points in the direction of highest variance (after accounting for the variance captured by earlier principal components). The further a weight is from zero, the more the principal component is in the direction of the corresponding feature. If two features have large weights of the same sign (both positive or both negative), then increases in one tend expect to be associated with increases in the other. To contrast, features with different signs can be expected to show a negative correlation: increases in one variable should result in a decrease in the other.

#### 1st Principal Component
The first principal component is explained mostly by the `LP_STATUS_GROB_1.0` variable which shows the social status: low-income earner of a specific person. The next three features with highest weight: `HH_EINKOMMEN_SCORE`, `CAMEO_INTL_2015_WEALTH`, `PLZ8_ANTG3`, are all also a certain measure of wealth or income. We can reconcile this with the fact that features with large weights in the same sign are a sign of correlation between the variables. Realistically, we would expect that each feature portraying wealth would be correlated. The three most negatively correlated features are: `FINANZ_MINIMALIST`, `KBA05_ANTG1`, `MOBI_REGIO`. For example: `FINANZ_MINIMALIST`, shows the lack of interest of someone in finance, which has an obvious negative correlation to wealth.

#### 2nd Principal Component
The second principal component is explained mostly by: `ALTERSKATEGORIE_GROB`, `FINANZ_VORSORGER`, `ZABEOTYP_3`, and the least by: `YOUTH_DECADE`, `FINANZ_SPARER`, `SEMIO_REL`. This is a mixture of age, financial awareness, and environmental/religious awareness respectively.

#### 3rd Principal Component
I find this third component most interesting, as many of the features that explain this principal component relate to personality traits. The most explanatory feature is `SEMIO_VERT` which is a measure of 'dreamfulness', followed by `SEMIO_FAM`: family-mindedness, `SEMIO_SOZ`: socially-mindedness and so forth. The least explanatory features, and those most negatively correlated are interestingly: `ANREDE_KZ` which shows gender, `SEMIO_KAEM` which is a combative-minded feature, `SEMIO_DOM` which is a dominant-minded feature and so forth.

What is interesting is that as we move through the principal components, at least the first three, the weights on features become larger. This means that those features more strongly explain the principal component, after accounting for the variance captured by earlier principal components.


###  3.1: Apply Clustering to General Population
When investigating clustering and drawing scree-plot, I did not find a specific kink in the graph, but rather the average distance kept decreasing. The largest decrease happened in the first 5 clusters, before the benefit of adding an extra cluster starts levelling out. This is why I chose to segment the population into 10 clusters, a nice round number where the average distance is starting to level out on the scree plot.

### 3.3: Compare Customer Data to Demographics Data
When including the population count for all including those we dropped for analysis as they had missing data, the culsters: 1,3,10 were the most over represented in the customer data compared to the general population data (2.48%, 0.2%, 1.43%. This was at a higher percentage for low-NaN data: 6.32%, 2.9% and 3.09% respectively). Based on the full general population data, it seems that cluster 3 is represented similarly in both general population and customer data, so we can exclude it from the 'over-represented' category. Other than clusters 7, and 8 were also very closely represented.

The must under-represented compared to full population data were clusters: 4, 5, by a substantial 4.72% and 6.17% respectively.

***Over-Represented Cluster: #5***
Based on the cluster centers, the highest value principal components are `PC8`, `PC7`, and `PC5` whereas the lowest value principal component are `PC2`, `PC1`, and `PC6`

*Most Value*
- `PC8` is correlated the most with families with children, especially teenagers, as well as aspiring average to low-income earners.
- `PC7` is correlated most with independent workers and title-holder households, as well as residing in a mixed use building
- `PC5`: correlated with number of adults in household, family living in two-generational household, as well as an inverse correlation with region purchasing power (more expensive regions)

*Least Value*
- `PC2`: is correlated with age, low financial preparedness.
- `PC1`: correlated with low income, and a region with higher share of 6-10 family homes
- `PC6`: average earning villagers who live in multi generational households, and is correlated with lack of home-ownership.
This means that generally speaking aspiring independent families with children, who live in mixed-use mulit-generational housing and in areas where purchasing power is low are over represented in the customer cluster data compared to the general population. These are the people that are most important to the business of the company

***Under-Represented Cluster: #10***
Based on the cluster centers, the highest value principal components are: `PC7`, `PC6` , `PC13`, and the lowest value principal components are: `PC1`, `PC11`, and `PC19`

*Most Value*
- `PC6` is correlated with average earning villagers who live in multi generational households, and is correlated with lack of home-ownership.
- `PC13` is correlated with working-class, multi-cultural youth.

*Least Value*
- `PC11` is most correlated with single-parents, high share of 10+ family houses, and an established middle class
- `PC19` is most correlated with older upper class people.
This means that the most under represented cluster in our analysis contains people who are younger and don't own homes, as well as being more working class and multi-cultured.

