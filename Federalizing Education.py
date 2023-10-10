#!/usr/bin/env python
# coding: utf-8

# # A Glimpse on Federalizing Public Education
# 
# ### This is meant as a hypothetical test, based solely upon theory and existing data. The political ramifications of such are not discussed, as it provides more of a glimpse into a potential alternative theoretical pathway to addressing education rather than a realistic possibility of being implemented. Various Python packages are used, and you will find integrated Tableau visualizations as well as an exported SQL database created to host the code.
# 
# 
# #### The data is sourced from the NAEP, BEA, and NCES, and contains the following: data on 8th grade achievement scores by state (denoted by variable AS), broken into demographics of Ethnicity, Subject (math and reading), and Reduced Lunch status. It also contains data on nationwide Per-Pupil Expenditure (PPE) by state, and state Regional Price Parities (RPP) data to be used to standardize PPE into a value that can be compared accross different costs of living - this variable is SPPE. The demographic data has been aggregated into four categories: Not White or Asian and Reduced Lunch, Not White or Asian and No Reduced Lunch, White or Asian and Reduced Lunch, and White or Asian and No Reduced Lunch. 

# In[104]:


# We will begin by importing libraries of course
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from IPython.core.getipython import get_ipython

## We now need to load in our data 
# Load Per-Pupil Expenditure data
ppe_data = pd.read_excel("/Users/kolbedumas/Documents/Python/Federalizing Education/PPE2019.xlsx")

# Load Regional Price Parities data
rpp_data = pd.read_csv("/Users/kolbedumas/Documents/Python/Federalizing Education/COLI2019.csv", skiprows=range(1, 5), delimiter='\t')  # where x is the line number to start reading data from


# The test score data is broken up into four parts, lets load it then clean it up
math_ethnicity_data = pd.read_excel("/Users/kolbedumas/Documents/Python/Federalizing Education/MathRace2019.Xls", skiprows=8)
math_lunch_data = pd.read_excel("/Users/kolbedumas/Documents/Python/Federalizing Education/MathLunch2019.Xls", skiprows=8)
reading_ethnicity_data = pd.read_excel("/Users/kolbedumas/Documents/Python/Federalizing Education/ReadRace2019.Xls", skiprows=8)
reading_lunch_data = pd.read_excel("/Users/kolbedumas/Documents/Python/Federalizing Education/ReadLunch2019.Xls", skiprows=8)

# Adding categorical variables to the data frames to show if math or reading and if based on ethnicity or lunch status
math_ethnicity_data['Subject'] = 'Math'
math_ethnicity_data['Demographic'] = 'Ethnicity'

math_lunch_data['Subject'] = 'Math'
math_lunch_data['Demographic'] = 'Lunch_Status'

reading_ethnicity_data['Subject'] = 'Reading'
reading_ethnicity_data['Demographic'] = 'Ethnicity'

reading_lunch_data['Subject'] = 'Reading'
reading_lunch_data['Demographic'] = 'Lunch_Status'

# Merging the test score data 
test_score_data = pd.concat([math_ethnicity_data, math_lunch_data, reading_ethnicity_data, reading_lunch_data], ignore_index=True)


# In[86]:


# Data cleaning time 
# Split the single string column into three new columns
rpp_data[['Code', 'State', 'RPP']] = rpp_data['SARPP Regional price parities by state'].str.split(',', expand=True)

# Drop the original concatenated column
rpp_data.drop('SARPP Regional price parities by state', axis=1, inplace=True)

# Drop the code column, not needed
rpp_data.drop('Code', axis=1, inplace=True)

# Select only the 'PPE15' and 'STABR' columns
ppe_data = ppe_data.loc[:, ['STABR', 'PPE15']]

# Rename the columns to 'State' and 'PPE'
ppe_data.rename(columns={'STABR': 'State', 'PPE15': 'PPE'}, inplace=True)

# Dropping rows from the PPE DataFrame
ppe_data.drop(index=range(51, 55), inplace=True)

# Dropping rows from the RPP DataFrame
rpp_data.drop(index=range(51, 52), inplace=True)

# Resetting the index after dropping rows
ppe_data.reset_index(drop=True, inplace=True)
rpp_data.reset_index(drop=True, inplace=True)

# For example, if the index of the row to be deleted is 'x', you would use:
ppe_data.drop(index=51, inplace=True)
rpp_data.drop(index=51, inplace=True)

# Reset index after dropping
ppe_data.reset_index(drop=True, inplace=True)
rpp_data.reset_index(drop=True, inplace=True)

# Rename columns using the `rename` method
test_score_data.rename(columns={
    'Jurisdiction': 'State',
    'Race/ethnicity using 2011 guidelines, school-reported': 'Race',
    'National School Lunch Program eligibility, 3 categories': 'Lunch_Status',
    'Average scale score': 'AS'
}, inplace=True)

# Convert AS to a numeric value
test_score_data['AS'] = pd.to_numeric(test_score_data['AS'], errors='coerce')

# Remove the unwanted lunch status rows
test_score_data = test_score_data[test_score_data['Lunch_Status'] != 'Information not available']
 
# Rename lunch categories
test_score_data['Lunch_Status'] = test_score_data['Lunch_Status'].replace({
    'Eligible': 'Reduced Lunch',
    'Not eligible': 'No Reduced Lunch'
})

# Isolate desired columns
filtered_test_score_data = test_score_data.loc[:, ['State', 'Race', 'AS', 'Lunch_Status', 'Subject']]

# Drop unwanted last columns
filtered_test_score_data.drop (index=range(1082,1085), inplace=True)
filtered_test_score_data.reset_index(drop=True, inplace=True)
filtered_test_score_data.drop (index=975, inplace=True)
filtered_test_score_data.reset_index(drop=True, inplace=True)

# Drop NA values
final_test_score_data = filtered_test_score_data[filtered_test_score_data['AS'].notna()]


# In[87]:


## Lets merge Race and Lunch Status into a new singular column for readability
# Step 1: Create a deep copy
final_test_score_data_copy = final_test_score_data.copy(deep=True)

# Step 2: Fill missing values in 'Race' with 'Lunch_Status' and create a new 'Demographics' column
final_test_score_data_copy['Demographics'] = final_test_score_data_copy['Race'].fillna(final_test_score_data_copy['Lunch_Status'])

## Lets now consolidate the races into either White or Asian or Not White or Asian
# Step 3: Create aggregated categories 'WOA' and 'NWOA'
final_test_score_data_copy['Demographics'].replace({'White': 'White or Asian', 'Asian': 'White or Asian'}, inplace=True)
nwoa_list = ['Black', 'Hispanic', 'American Indian/Alaska Native', 'Native Hawaiian/Other Pacific Islander', 'Two or more races']
final_test_score_data_copy['Demographics'].replace(nwoa_list, 'Not White or Asian', inplace=True)

# Step 4: Drop the 'Race' and 'Lunch_Status' columns as they are no longer needed
final_test_score_data_copy.drop(['Race', 'Lunch_Status'], axis=1, inplace=True)

# Step 5: Perform aggregation based on 'State' and 'Demographics'
aggregated_data = final_test_score_data_copy.groupby(['State', 'Demographics'])['AS'].mean().reset_index()




# In[88]:


# Lets Merge all the data
state_dict = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
    'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
    'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
    'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
    'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
    'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
    'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
    'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
    'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
    'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
    'WI': 'Wisconsin', 'WY': 'Wyoming'
}

ppe_data['State'] = ppe_data['State'].map(state_dict)
final_merged_data = pd.merge(aggregated_data, ppe_data, on='State', how='inner')
final_merged_data = pd.merge(final_merged_data, rpp_data, on='State', how='inner')

# Now lets create a standardized PPE based on RPP for each state
# Convert PPE and RPP to numeric type if they are not
final_merged_data['PPE'] = pd.to_numeric(final_merged_data['PPE'], errors='coerce')
final_merged_data['RPP'] = pd.to_numeric(final_merged_data['RPP'], errors='coerce')

# Create the new SPPE column
final_merged_data['SPPE'] = final_merged_data['PPE'] / final_merged_data['RPP']

# Optionally, check for NaNs in the new SPPE column
nan_rows = final_merged_data[final_merged_data['SPPE'].isnull()]
if not nan_rows.empty:
    print("Warning: NaN values detected in the SPPE column. Please investigate.")

# Multiply the SPPE column by 100 to scale the values back to the thousands
final_merged_data['SPPE'] = (final_merged_data['PPE'] / final_merged_data['RPP']) * 100
 
pivot_df = final_merged_data.pivot_table(values='AS', index=['State'], columns=['Demographics'], aggfunc=np.mean).reset_index()

pivot_df['White or Asian + Reduced Lunch'] = (pivot_df['White or Asian'] + pivot_df['Reduced Lunch']) / 2
pivot_df['Not White or Asian + Reduced Lunch'] = (pivot_df['Not White or Asian'] + pivot_df['Reduced Lunch']) / 2
pivot_df['White or Asian + No Reduced Lunch'] = (pivot_df['White or Asian'] + pivot_df['No Reduced Lunch']) / 2
pivot_df['Not White or Asian + No Reduced Lunch'] = (pivot_df['Not White or Asian'] + pivot_df['No Reduced Lunch']) / 2

melted_df = pd.melt(pivot_df, id_vars=['State'], value_vars=['White or Asian + Reduced Lunch', 'Not White or Asian + Reduced Lunch', 'White or Asian + No Reduced Lunch', 'Not White or Asian + No Reduced Lunch'], var_name='Aggregated_Demographics', value_name='AS')

# Merge the melted dataframe back with the original pivot_df to get PPE, SPPE, RPP for each state
melted_df = pd.melt(pivot_df, id_vars=['State'], value_vars=['White or Asian + Reduced Lunch', 'Not White or Asian + Reduced Lunch', 'White or Asian + No Reduced Lunch', 'Not White or Asian + No Reduced Lunch'], var_name='Aggregated_Demographics', value_name='AS')

final_melted_df = pd.merge(melted_df, ppe_data, on='State', how='inner')
final_melted_df = pd.merge(final_melted_df, rpp_data, on='State', how='inner')

# Now lets create a standardized PPE based on RPP for each state
# Convert PPE and RPP to numeric type if they are not
final_melted_df['PPE'] = pd.to_numeric(final_melted_df['PPE'], errors='coerce')
final_melted_df['RPP'] = pd.to_numeric(final_melted_df['RPP'], errors='coerce')

# Create the new SPPE column
final_melted_df['SPPE'] = final_melted_df['PPE'] / final_melted_df['RPP']
# Multiply the SPPE column by 100 to scale the values back to the thousands
final_melted_df['SPPE'] = (final_melted_df['PPE'] / final_melted_df['RPP']) * 100


# In[89]:


# Lets port this data set into SQL to be able to update in the future
database_url = "postgresql://postgres:Devils6!@localhost/state_as_data"
engine = create_engine(database_url)
final_melted_df.to_sql("state_as_data_table", engine, if_exists='replace', index=False)
# Lets Export to a CSV to use in Tableau
final_melted_df.to_csv('state_as_data.csv', index=False)


# In[90]:


## Time for some exploratory data analysis
def get_top_states_by_as(df, top_n=5):
   
    #This function takes the DataFrame and an optional parameter to define the number of top states you're interested in.
    #It returns a list of top states based on the mean Achievement Score (AS).
   
    state_as_means = df.groupby('State')['AS'].mean().reset_index()
    sorted_state_as_means = state_as_means.sort_values(by='AS', ascending=False)
    top_states = sorted_state_as_means.head(top_n)['State'].tolist()
    
    return top_states

# Assuming final_melted_df is your DataFrame with the relevant data
top_as_states = get_top_states_by_as(final_melted_df, top_n=10)  # You can change the number 5 to however many states you wish to consider


# ### The following is an initial scatter plot of SPPE vs AS, as the graph shows, there does not seem to be any correlation between a state's spending on education and achievement scores. 

# In[91]:


# Plot SPPE against AS 
sns.scatterplot(x='SPPE', y='AS', data=final_melted_df)
plt.show()
# Does not seem to be any direct link between the two


# ### To take it a step further, we can see from the following regression analysis that SPPE is not statistically significant;however, we see that the demographic groups are. Not White or Asian + Reduced lunch groups see a significant drop in test scores compared to White or Asian + No Reduced Lunch; a difference in roughly 25 points. 

# In[92]:


# Regression Time 

# Create dummy variables for 'Demographics'
final_merged_data_with_dummies = pd.get_dummies(final_melted_df, columns=['Aggregated_Demographics'], prefix='Aggregated_Demographics')
# Convert Boolean columns to integers
final_merged_data_with_dummies[['Aggregated_Demographics_Not White or Asian + Reduced Lunch', 'Aggregated_Demographics_White or Asian + Reduced Lunch', 'Aggregated_Demographics_Not White or Asian + No Reduced Lunch', 'Aggregated_Demographics_White or Asian + No Reduced Lunch']] = final_merged_data_with_dummies[['Aggregated_Demographics_Not White or Asian + Reduced Lunch', 'Aggregated_Demographics_White or Asian + Reduced Lunch', 'Aggregated_Demographics_Not White or Asian + No Reduced Lunch', 'Aggregated_Demographics_White or Asian + No Reduced Lunch']].astype(int)

# Select dependent variable
y = final_merged_data_with_dummies['AS']

# Select predictor variables (independent variables)
X1 = final_merged_data_with_dummies[['SPPE', 'Aggregated_Demographics_Not White or Asian + Reduced Lunch', 'Aggregated_Demographics_White or Asian + Reduced Lunch', 'Aggregated_Demographics_Not White or Asian + No Reduced Lunch', 'Aggregated_Demographics_White or Asian + No Reduced Lunch']]

# Add constant term for intercept
X1 = sm.add_constant(X1)

# Fit the model
model1 = sm.OLS(y, X1).fit()

# Get summary statistics
print("Summary for Model 1")
print(model1.summary())

# Data shows us that there while SPPE is not statistically significant, the rest of the demographic variables are. As expected, Not White or Asian and Reduced lunch groups are underperforming considerably against White or Asian and No Reduced Lunch groups. Notably, being in one of either Not White or Asian or Reduced Lunch creates somewhat of a normalized score in the middle, although it appears having reduced lunch means you will perform slightly worse. 


# ### Below are three visualizations crafted in Tableau that can be filtered through to see specific data on state AS scores for the various demographics as well as the state's SPPE for reference.

# In[93]:


# Embed a Tableau Visualization
from IPython.core.display import display, HTML
display(HTML("<div class='tableauPlaceholder' id='viz1693541169947' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;St&#47;StatesAchievementScoresbyDemographicData-2&#47;AllStatesScoresSPPE&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='StatesAchievementScoresbyDemographicData-2&#47;AllStatesScoresSPPE' /><param name='tabs' value='yes' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;St&#47;StatesAchievementScoresbyDemographicData-2&#47;AllStatesScoresSPPE&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1693541169947');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"))


# ### To get an idea of the disparity differences between the highest and lowest performing groups in different states, a new column "disparity" was created to sort by. We can see below the states who are closest in terms of how well thier groups do and those who are the furthest.

# In[94]:


# Lets see what States have the lowest disparity between the top performing groups and the bottom 
# First, filter the DataFrame to include only the two demographic groups you're interested in.
filtered_df = final_melted_df[final_melted_df['Aggregated_Demographics'].isin(['Not White or Asian + Reduced Lunch', 'White or Asian + No Reduced Lunch'])]

# Next, pivot the DataFrame to get each demographic group as a separate column.
pivot_df = filtered_df.pivot_table(index='State', columns='Aggregated_Demographics', values='AS', aggfunc='mean').reset_index()

# Calculate the disparity column
pivot_df['Disparity'] = pivot_df['White or Asian + No Reduced Lunch'] - pivot_df['Not White or Asian + Reduced Lunch']

# Sort by disparity
sorted_df = pivot_df.sort_values(by='Disparity', ascending=True)

# Display the top 10 and bottom 10 in terms of disparity 
sorted_df.head(10)

#Vermont could be an outlier, along with Jersey, so we wont use those in developing a model


# In[95]:


sorted_df.tail(10)


# ### From the data we have, we can see there are essentially a few key factors; some states are very good in terms of overall scoring, however, they lack when it comes to narrowing the achievement gap as the disparity between the top performing groups and the bottom performing groups is large. There are also states who do fairly well at minimizing the achievement gap, but their overall scores lack. The idea is the following; if education were to be federalized, and thorough research was done on the top 5 states in each of these categories (lowest disparity + highest average AS score), could you combine the approaches to education these states take into one singular approach that combines them? This model attempts to do so, giving more weight to reducing the achievement score (0.7 to 0.3).

# In[96]:


# Now lets see how much we could decrease the Achievement Gap if we were to apply models of what these top performing states in terms of disparity are doing

# Assume final_melted_df is your primary dataframe containing 'State', 'Aggregated_Demographics', and 'AS' among other columns

# Step 0: Define your top-performing states in terms of disparity and achievement score (AS)
top_disparity_states = ['West Virginia', 'Oklahoma', 'Hawaii', 'Indiana', 'Wyoming']
top_as_states = ['Massachusetts', 'New Jersey', 'Connecticut', 'Georgia', 'Illinois']  # Replace with the actual top states based on AS

# Step 1: Create segregated dataframes based on these lists
top_disparity_df = final_melted_df[final_melted_df['State'].isin(top_disparity_states)]
top_as_df = final_melted_df[final_melted_df['State'].isin(top_as_states)]
other_states_df = final_melted_df[~final_melted_df['State'].isin(top_disparity_states + top_as_states)]

# Step 2: Calculate Benchmark Disparity and Mean AS
# I'll stick to your existing calculation methods for simplicity
benchmark_disparity = top_disparity_df.groupby('State')['AS'].std().mean()
benchmark_mean_as = top_as_df['AS'].mean()

# Step 3: Calculate State-Wide Mean AS for other states
state_means = other_states_df.groupby('State')['AS'].mean().reset_index()

# Step 1-3: (As in your original code)

# Step 4: Project Scores with Reduced Disparity and General Increase for other states
projected_scores_df = other_states_df.copy()

# Merging the state_means with other_states_df for adjusted calculations
projected_scores_df = pd.merge(projected_scores_df, state_means, on='State', suffixes=('', '_mean_state'))

# Calculate Projected AS
projected_scores_df['Projected_AS'] = projected_scores_df['AS_mean_state'] + (benchmark_mean_as - projected_scores_df['AS_mean_state'].mean())

# Calculate Reduced Disparity AS
# Assuming that 'Aggregated_Demographics' is your demographic column
# This operation will reduce the gap between the projected AS and the state mean AS, thereby reducing disparity
projected_scores_df['Reduced_Disparity_AS'] = projected_scores_df['AS'] + (benchmark_disparity - projected_scores_df.groupby('State')['AS'].transform('std'))

print(projected_scores_df[['AS', 'Projected_AS', 'Reduced_Disparity_AS']].describe())
 


# In[97]:


# Lets examine

# Combine the lists of top states based on AS and disparity
combined_top_states = set(top_as_states + top_disparity_states)

# Create a new DataFrame by excluding rows corresponding to the top states
final_filtered_df = final_melted_df[~final_melted_df['State'].isin(combined_top_states)]

# Define a weight alpha, that places larger emphasis on equity than it does higher test scores
alpha = 0.7

# Calculate the composite score
projected_scores_df['Composite_Score'] = alpha * projected_scores_df['Projected_AS'] + (1 - alpha) * projected_scores_df['Reduced_Disparity_AS']

# Descriptive statistics for Composite_Score
print("Current Scores Descriptive Statistics:")
print(final_filtered_df['AS'].describe())

print("Composite Scores Descriptive Statistics:")
print(projected_scores_df['Composite_Score'].describe())

# Proceed with Regression Analysis for both current and projected scores

demographic_columns = [
    'Aggregated_Demographics_Not White or Asian + Reduced Lunch', 
    'Aggregated_Demographics_White or Asian + Reduced Lunch', 
    'Aggregated_Demographics_Not White or Asian + No Reduced Lunch', 
    'Aggregated_Demographics_White or Asian + No Reduced Lunch'
]
subset_df = final_filtered_df_with_dummies[demographic_columns]

# For final_filtered_df
final_filtered_df_with_dummies = pd.get_dummies(final_filtered_df, columns=['Aggregated_Demographics'], prefix='Aggregated_Demographics')
final_filtered_df_with_dummies[demographic_columns] = final_filtered_df_with_dummies[demographic_columns].astype(int)

# For projected_scores_df
projected_scores_df_with_dummies = pd.get_dummies(projected_scores_df, columns=['Aggregated_Demographics'], prefix='Aggregated_Demographics')
projected_scores_df_with_dummies[demographic_columns] = projected_scores_df_with_dummies[demographic_columns].astype(int)

# Regression for final_filtered_df
y_filtered = final_filtered_df_with_dummies['AS']
X_filtered = final_filtered_df_with_dummies[['SPPE'] + demographic_columns]
X_filtered = sm.add_constant(X_filtered)
model_filtered = sm.OLS(y_filtered, X_filtered).fit()
print("Summary for the Filtered Model")
print(model_filtered.summary())

# Regression for projected_scores_df
y_projected = projected_scores_df_with_dummies['Composite_Score']
X_projected = projected_scores_df_with_dummies[['SPPE'] + demographic_columns]
X_projected = sm.add_constant(X_projected)
model_projected = sm.OLS(y_projected, X_projected).fit()
print("Summary for the Projected Model")
print(model_projected.summary())




# ### The model shows what the numbers could look like in this theoretical sitatution; we see average scores raise, and the disparity lowers significantly compared to what it was before. While top scores might come down a bit, this would be a necessary price to pay. Below are visualizations of the current numbers and the projected numbers. A random forest was also ran to get a potential SPPE value that would be standard. 

# In[98]:


# Visualizations


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(final_filtered_df['AS'], bins=20, alpha=0.7, label='Current AS', color='blue')
plt.title('Distribution of Current AS')
plt.xlabel('Scores')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(projected_scores_df['Composite_Score'], bins=20, alpha=0.7, label='Composite Scores', color='green')
plt.title('Distribution of Composite Scores')
plt.xlabel('Scores')
plt.ylabel('Frequency')

plt.tight_layout()

import seaborn as sns

plt.figure(figsize=(10, 6))

sns.boxplot(data=[final_filtered_df['AS'], projected_scores_df['Composite_Score']], notch=True)
plt.xticks([0, 1], ['Current AS', 'Composite Scores'])
plt.title('Boxplot of Current AS and Composite Scores')
plt.ylabel('Scores')




# In[99]:


# Load the shapefile for U.S. states. For demonstration, I am using Geopandas' built-in dataset
filepath = "/Users/kolbedumas/Documents/Python/Federalizing Education/ne_110m_admin_1_states_provinces"
gdf = gpd.read_file(filepath)

# Exclude Alaska and Hawaii
gdf = gdf.loc[~gdf['name'].isin(['Alaska', 'Hawaii'])]

# Merge the geopandas DataFrames
merged_filtered = gdf.merge(final_filtered_df, how='left', left_on='name', right_on='State')
merged_projected = gdf.merge(projected_scores_df, how='left', left_on='name', right_on='State')

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(20, 15))

# Add an axes for the colorbar
cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])  # [left, bottom, width, height]

# Define a colormap and a normalization
cmap = plt.cm.viridis
norm = mcolors.Normalize(vmin=255, vmax=285)

# Plotting for final_filtered_df
merged_filtered.boundary.plot(ax=axes[0], linewidth=1)
merged_filtered.plot(column='AS', ax=axes[0], legend=False, cmap=cmap, norm=norm)

# Plotting for projected_scores_df
merged_projected.boundary.plot(ax=axes[1], linewidth=1)
merged_projected.plot(column='Composite_Score', ax=axes[1], legend=False, cmap=cmap, norm=norm)

# Create the colorbar
cbar = mcolorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')

# Set titles
axes[0].set_title('Actual Educational Scores by State')
axes[1].set_title('Projected Educational Scores by State')

# Show the plots
plt.show()



# In[101]:


# To handle SettingWithCopyWarning, you can use .copy() explicitly to create a new DataFrame
projected_scores_df = projected_scores_df.copy()
final_filtered_df = final_filtered_df.copy()

# Compute statistical features for both DataFrames
for df, score_column in [(projected_scores_df, 'Composite_Score'), (final_filtered_df, 'AS')]:
    df.loc[:, 'Mean_' + score_column] = df[score_column].mean()
    df.loc[:, '25th_Percentile_' + score_column] = df[score_column].quantile(0.25)
    df.loc[:, '75th_Percentile_' + score_column] = df[score_column].quantile(0.75)


# Join the dataframes on a common key
merged_df = final_filtered_df.merge(projected_scores_df, on='State', suffixes=('_Actual', '_Projected'))

# Create feature matrix and target vector
X = merged_df[['Mean_AS', '25th_Percentile_AS', '75th_Percentile_AS', 
               'AS_mean_state', 'Projected_AS', 'Reduced_Disparity_AS']]
y = merged_df['SPPE_Actual']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

# Make predictions
merged_df['Aggregate_SPPE'] = rf.predict(X)

# If you need to evaluate the model
from sklearn.metrics import mean_squared_error
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("Predictions:")
print(merged_df['Aggregate_SPPE'].head())


# In[1]:





# In[ ]:




