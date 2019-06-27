# Some imports will be used in the notebook.
import pandas as pd
import numpy as np
import seaborn as sns
import re
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.utils import shuffle

"""Script for Cleaning data from 
https://www.doh.wa.gov/DataandStatisticalReports/DataSystems/BehavioralRiskFactorSurveillanceSystemBRFSS
for the years 2011 - 2017.

The dataset is 200mbs in entirety and consists of 464 columns. Many are near duplicate survey questions asked
in different years and are here combined where able.
"""

# Read in data to pandas DataFrame
df = pd.read_stata("C:\\Users\Paul\PycharmProjects\BlogPost\data\WA_BRFSS_11to17_B.dta",
                   convert_missing=False)

# Rename age column.
df['Age'] = df['_ageg5yr']

# 77 and 99 are coded as unknown and refused respectively. I'm assuming someone filled out the survey and so score a 1.
df['numadult'] = df['numadult'].map(lambda x: 1 if x in [77, 99] else x)
df['hhadult'] = df['hhadult'].map(lambda x: 1 if x in [77, 99] else x)

df['numadult'] = df['numadult'].fillna(1)
df['hhadult'] = df['hhadult'].fillna(1)

# In case of duplicate entries for the same year, take the larger.
df['adults'] = np.maximum(df['hhadult'], df['numadult'])
df['Adults'] = df['adults'].map(lambda x: 1 if x == 0 else x)

# 88 and 99 are coded as unknown and refused respectively. An adult is assumed to be answering and no
# children are assumed.
df['Children'] = df['children'].map(lambda x: 0 if x in [88, 99] else x)

df['Ownership'] = df['renthom1'].replace('Refused', 'Unknown')
df['Ownership'] = df['Ownership'].replace('Don\'t Know', 'Unknown')

df['Total Household'] = df['Adults'] + df['Children']

# Change values for nicer plotting.
df['Race'] = df['_race'].map({'White NH': 'White', 
                              'Hispanic': 'Hisp.', 
                              'Asian NH': 'Asian', 
                              'Black NH': 'Black',
                              'DK/Refused': 'Don\'t\nKnow', 
                              'AIAN NH': 'AIAN', 
                              'Other NH': 'Other',
                              'NHOPI NH': 'NHOPI'})

# Simplify into boolean column.
df['employment'] = pd.concat([df['employ'], df['employ1']], join='inner', ignore_index=True, sort=False)
df['employment'] = df['employment'].cat.add_categories('Unknown')
df['Employment'] = df['employment'].fillna('Unknown')
df['Employment'] = df['Employment'].replace('Refused', 'Unknown')

# Rename value for nicer plotting.
df['Income'] = df['income2'].replace('Don\'t Know', 'Unknown')

# King County's median income in 2015 was $75k
df['Over Median Income'] = (df['income2'] == '$75+').astype(str)

# Use calculated BMI.
df['_bmi5cat'] = df['_bmi5cat'].cat.add_categories('Unknown')
df['Overweight'] = df['_bmi5cat'].fillna('Unknown')

# Clean zipcode column.
df['zips'] = df['zipcode'] + df['zipcode1']
df['zips'] = df['zips'].str.strip()
df['zips'] = [int(x) if x != '' else np.nan for x in df['zips']]


def seattle_zip(x):
    if not re.match(r'98\d\d\d[.]0', str(x)):
        return 99999
    else:
        return x


# Remove any entries with zipcodes outside of the Seattle area.
df['Zip Code'] = df['zips'].map(seattle_zip)
df = df[(df['Zip Code'] != 99999) & (df['Zip Code'] != 77777)]

# USDA food desert locations are at the census tract level and can be found at:
# https://www.ers.usda.gov/data-products/food-access-research-atlas
#
# Zipcodes cross referenced with census tracts designated as food deserts can be found at:
# https://www.huduser.gov/portal/datasets/usps_crosswalk.html
zips = [98103, 98115, 98166, 98146, 98188, 98138, 98168, 98178, 98057, 98055, 98058, 98198, 98002, 98001, 98047, 98047]
desert_zips = list(set(zips))
df['In Food Desert'] = df['Zip Code'].map(lambda x: True if x in desert_zips else False)
df['Zip Code'] = df['Zip Code'].astype(int).astype(str)

# Relabel columns with respondent's self assesment.
df['Good Health'] = df['genhlth'].replace('DK', 'Unknown')
df['Good Health'] = df['Good Health'].replace('Missing', 'Unknown')

# Convert nan code to nan.
df['Sleep Hrs'] = df['sleptim1'].map(lambda x: np.nan if x in [99, 77] else x)

# Insurance status.
df['Insurance'] = df['_hcvu651'].replace('Missing', 'Unknown')

# Did you forego medical care do to cost?
df['Dr Too Much'] = df['medcost'].replace('DK', 'Unknown')
df['Dr Too Much'] = df['Dr Too Much'].replace('Refused', 'Unknown')

# Time since last apt. Categorical.
df['Recent Dr Visit'] = df['checkup1'].replace('DK', 'Unknown')
df['Recent Dr Visit'] = df['Recent Dr Visit'].replace('Refused', 'Unknown')

# Smoking history.
df['Smoker'] = df['_smoker3'].replace('Missing', 'Unknown')

# Simplify Pre diabetic and diabetic columns to boolean.
df['prediab1'] = df['prediab1'].fillna('DK')
df['pre_diab1'] = df['prediab1'].map(lambda x: True if x == 'Yes' else False)
df['pre_diab2'] = df['diabete3'].map(lambda x: True if x == 'Borderline/Pre-Diabetes' else False)
df['Pre Diabetic'] = ((df['pre_diab1'] == True) | (df['pre_diab2'] == True)).astype(str)

df['Diabetic'] = df['diabete3'].map(lambda x: True if x  == 'Yes' else False)

# Add active column for meeting activity recomendations, boolean .
df['active'] = pd.concat([df['_paindex'], df['_paindx1']], join='inner', ignore_index=True, sort=False)
df['active'] = df['active'].cat.add_categories('Unknown')
df['Active'] = df['active'].fillna('Unknown')

# Clean vegetable and fruit data to reflex frequency of consumption by day, week and month.
# Any unknowns we assume are zero to avoid inflating our good eating habits.
# df['vegeda1_'] = df['vegeda1_'].fillna(0)

# df['vegeda2_'] = df['vegeda2_'].fillna(0)

# df['frutda1_'] = df['frutda1_'].fillna(0)

# df['frutda2_'] = df['frutda2_'].fillna(0)

# df['ftjuda1_'] = df['ftjuda1_'].fillna(0)

# df['ftjuda2_'] = df['ftjuda2_'].fillna(0)

# df['_vegesum'] = df['_vegesum'].fillna(0)

# df['_vegesu1'] = df['_vegesu1'].fillna(0)

# df['_frutsum'] = df['_frutsum'].fillna(0)

# df['_frutsu1'] = df['_frutsu1'].fillna(0)

# # Combine columns from different years.
# df['veg_day'] = df['vegeda1_'] + df['vegeda2_']

# df['fruit_day'] = df['frutda1_'] + df['frutda2_']

# df['juice_day'] = df['ftjuda1_'] + df['ftjuda2_']

# df['veg_sum'] = df['_vegesum'] + df['_vegesu1']

# df['fruit_sum'] = df['_frutsum'] + df['_frutsu1']

# # Assume survey takers skipping the fruit and veg every day questions are filled with shame at not doing so.
# df['_frtlt1'] = df['_frtlt1'].map({'One or more per day': 1, 'Less than once per day': 0, 'Missing': 0})
# df['_frtlt1a'] = df['_frtlt1a'].map({'One or more per day': 1, 'Less than once per day': 0, 'Missing': 0})

# df['_veglt1'] = df['_veglt1'].map({'One or more per day': 1, 'Less than once per day': 0, 'Missing': 0})
# df['_veglt1a'] = df['_veglt1a'].map({'One or more per day': 1, 'Less than once per day': 0, 'Missing': 0})

# # Combine years.
# df['Fruit Daily'] = (df['_frtlt1'] == 1) | (df['_frtlt1a'] == 1)
# df['Veg Daily'] = (df['_veglt1'] == 1) | (df['_veglt1a'] == 1)

# # Exclude people claiming to eat over 16 servings of fruit or 23 servings of vegetables.
# df = df[((df['_vegetex'] == 'Included') | (df['_vegete1'] == 'Included')) &
#         ((df['_fruitex'] == 'Included') | (df['_fruite1'] == 'Included'))]


# Functions to decode weekly and monthly frequency responses.
def nothing(x):
    # code 555
    if re.match(r'^555$', str(x).strip()):
        return True
    else:
        return False

def etoh(x):
    # Code in the 100s.
    week = re.match(r'1(\d\d)', str(x).strip())
    month =  re.match(r'2(\d\d)', str(x).strip())
    unknown = re.match(r'7|8', str(x).strip())
    none = re.match(r'9', str(x).strip())
    if week:
        return str(int(week.group(1)) * 4)
    if month:
        if month.group(1)[0] == '0':
            return month.group(1)[1]
        else:
            return month.group(1)
    if unknown:
        return 'Unknown'
    elif none:
        return '0'

# def per_week(x):
#     # Code in the 200s.
#     else:
        return False


# def per_week_17(x):
#     # Code in the 200s, but not 200.
#     else:
#         return False


def fruit(x):
    # Code in the 300s.
    if re.match(r'1', str(x)):
        return 'Daily'
    if re.match(r'2', str(x)):
#         print('week')
        return 'Weekly'
    if re.match(r'3', str(x)):
        return 'Monthly'
    if re.match(r'5', str(x)):
        return 'None'
    else:
        return ''


# def per_month_17(x):
#     # Code in the 300s, but not 300.
#     else:
#         return False

# Alcohol consumption.
df['Alcohol'] = df['alcday5'].map(etoh)


# Make a column for no fruit or veg at all.
# df['No Veg'] = df['vegetab1'].map(nothing) | df['vegetab2'].map(nothing)
# df['No Fruit'] = df['fruit1'].map(nothing) | df['fruit2'].map(nothing)

df['Vegetables'] = df['vegetab1'].map(fruit) + df['vegetab2'].map(fruit)
df['Vegetables'] = df['Vegetables'].map(lambda x: 'Unknown' if x == '' else x)

df['Fruit'] = df['fruit1'].map(fruit) + df['fruit2'].map(fruit)
df['Fruit'] = df['Fruit'].map(lambda x: 'Unknown' if x == '' else x)

# And one for monthly intake, unless daily or weekly boxes are checked. No double counting.
# df['veg_month'] = df['vegetab1'].map(per_month) | df['vegetab2'].map(per_month_17)
# df['Veg Monthly'] = ((df['veg_month'] == 1) & (df['Veg Weekly'] == 0) & (df['Veg Daily'] == 0))

# df['fruit_month'] = df['fruit1'].map(per_month) | df['fruit2'].map(per_month_17)
# df['Fruit Monthly'] = ((df['fruit_month'] == 1) & 
#                        (df['Fruit Weekly'] == 0) & 
#                        (df['Fruit Daily'] == 0))


# The columns we want to keep in some sort of order.
final_colunns = [
                 'Age',
                 'Race',
                 'Income',
                 'Over Median Income',
                 'Ownership',
#                  'Adults',
#                  'Children',
                 'Total Household',
#                  'Zip Code',
#                  'In Food Desert',
                 'Employment',
                 'Active',
                 'Overweight',
#                  'Pre Diabetic',
                 'Diabetic',
                 'Good Health',
#                  'No Fruit', 
#                  'No Veg',
#                  'Fruit Daily',
#                  'Veg Daily',
#                  'Fruit Weekly',
#                  'Veg Weekly',
#                  'Fruit Monthly',
#                  'Veg Monthly',
                 'Fruit',
                 'Vegetables',
                 'Sleep Hrs',
                 'Insurance',
                 'Dr Too Much',
                 'Recent Dr Visit',
                 'Smoker',
                 'Alcohol',
                 ]

df = df[final_colunns]

df = shuffle(df,
             random_state=42
             )
df.to_csv('model/model_data.csv', index_label='index')
# Finally, separate into different years.
# df11 = df[df['Year'] == 2011]

# df13 = df[df['Year'] == 2013]

# df15 = df[df['Year'] == 2015]

# df17 = df[df['Year'] == 2017]
