"""Script for Cleaning data from
https://www.doh.wa.gov/DataandStatisticalReports/DataSystems/BehavioralRiskFactorSurveillanceSystemBRFSS
for the years 2011 - 2017.

The dataset is 200mbs in entirety and consists of 464 columns. Many are near duplicate survey questions asked
in different years and are here combined where able.
"""

import pandas as pd
import numpy as np
import re
from sklearn.utils import shuffle


# Read in data to pandas DataFrame
df = pd.read_stata("C:\\Users\Paul\PycharmProjects\BlogPost\data\WA_BRFSS_11to17_B.dta",
                   convert_missing=False)

# Rename age column.
df['Age range'] = df['_ageg5yr']

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
df['Home ownership'] = df['Ownership'].replace('Don\'t Know', 'Unknown')

df['Total number living in household'] = df['Adults'] + df['Children']

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
df['Employment status'] = df['Employment'].replace('Refused', 'Unknown')

# Rename value for nicer plotting.
df['Income range'] = df['income2'].replace('Don\'t Know', 'Unknown')

# King County's median income in 2015 was $75k
df['Income over $75K'] = (df['income2'] == '$75+').astype(str)

# Use calculated BMI.
df['_bmi5cat'] = df['_bmi5cat'].cat.add_categories('Unknown')
df['Body Mass Index'] = df['_bmi5cat'].fillna('Unknown')

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

# Relabel columns with respondent's self assessment.
df['Good Health'] = df['genhlth'].replace('DK', 'Unknown')
df['Good Health'] = df['Good Health'].replace('Missing', 'Unknown')
df['General Health'] = df['Good Health'].replace('Excelent', 'Excellent')

# Convert nan code to nan.
df['Average hours of sleep'] = df['sleptim1'].map(lambda x: np.nan if x in [99, 77] else x)

# Insurance status.
df['Insurance status'] = df['_hcvu651'].replace('Missing', 'Unknown')

# Did you forgo medical care do to cost?
df['Dr Too Much'] = df['medcost'].replace('DK', 'Unknown')
df['Skipped a Dr visit because of money'] = df['Dr Too Much'].replace('Refused', 'Unknown')

# Time since last apt. Categorical.
df['Recent Dr Visit'] = df['checkup1'].replace('DK', 'Unknown')
df['Last Dr visit'] = df['Recent Dr Visit'].replace('Refused', 'Unknown')

# Smoking history.
df['Smoking status'] = df['_smoker3'].replace('Missing', 'Unknown')

# Simplify Pre diabetic and diabetic columns to boolean.
df['prediab1'] = df['prediab1'].fillna('DK')
df['pre_diab1'] = df['prediab1'].map(lambda x: True if x == 'Yes' else False)
df['pre_diab2'] = df['diabete3'].map(lambda x: True if x == 'Borderline/Pre-Diabetes' else False)
df['Pre Diabetic'] = ((df['pre_diab1'] == True) | (df['pre_diab2'] == True)).astype(str)

df['Diabetic'] = df['diabete3'].map(lambda x: True if x == 'Yes' else False)

# Add active column for meeting activity recommendations, boolean .
df['active'] = pd.concat([df['_paindex'], df['_paindx1']], join='inner', ignore_index=True, sort=False)
df['active'] = df['active'].cat.add_categories('Unknown')
df['Activity status'] = df['active'].fillna('Unknown')


# Functions to decode weekly and monthly frequency responses.
def nothing(x):
    if re.match(r'^555$', str(x).strip()):
        return True
    else:
        return False


def etoh(x):
    """Decode alcohol related columns"""
    week = re.match(r'1(\d\d)', str(x).strip())
    month = re.match(r'2(\d\d)', str(x).strip())
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


def fruit(x):
    if re.match(r'1', str(x)):
        return 'Daily'
    if re.match(r'2', str(x)):
        return 'Weekly'
    if re.match(r'3', str(x)):
        return 'Monthly'
    if re.match(r'5', str(x)):
        return 'None'
    else:
        return ''


# Alcohol consumption.
df['Estimated number of alcoholic drinks per month'] = df['alcday5'].map(etoh)

df['Vegetables'] = df['vegetab1'].map(fruit) + df['vegetab2'].map(fruit)
df['Vegetable voraciousness'] = df['Vegetables'].map(lambda x: 'Unknown' if x == '' else x)

df['Fruit'] = df['fruit1'].map(fruit) + df['fruit2'].map(fruit)
df['Fruit fanaticism'] = df['Fruit'].map(lambda x: 'Unknown' if x == '' else x)

# The columns we want to keep in some sort of order.
final_colunns = [
    'Age range',
    'Race',
    'Income range',
    'Income over $75K',
    'Home ownership',
    'Total number living in household',
    'Employment status',
    'Activity status',
    'Body Mass Index',
    'Diabetic',
    'General Health',
    'Fruit fanaticism',
    'Vegetable voraciousness',
    'Average hours of sleep',
    'Insurance status',
    'Skipped a Dr visit because of money',
    'Last Dr visit',
    'Smoking status',
    'Estimated number of alcoholic drinks per month',
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
