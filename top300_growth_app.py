#!/usr/bin/env python
# coding: utf-8

# Data analysis

# Goal: Generate a list of top 300 subscription apps that had the best
# rank growth over the past 365 days.



# read in data
from joblib import dump, load
import numpy as np
# for linear regression
from scipy.stats import linregress
import pandas as pd




app_info = load("app_info_df.pkl")

app_rank = load("app_rank_df.pkl")

app_info.head()

# Look at the shape of the data. 

app_info.shape

app_rank.shape


# Define a function to determine if an app is subscription-based by using keywords 'subscription', 'renewal','subscribe',"renew"


def subscription(description):
    subscribe_str_list = ['subscription', 'renewal','subscribe',"renew"]
    
    # if the description contains any of the words 
    # in subscribe_str_list, then return "Yes"
    for word in subscribe_str_list:
        if word in description:
            return "Yes"
        
    # if none of the words in subscribe_str_list is in description,
    # then return "No"   
    return "No"


# If I was given more time, I can use word2vec to find similar words.  

# Generate a column named `subscription` by using the `subscription` function. 
app_info['subscription'] = app_info['description'].apply(subscription)

app_info['subscription'].value_counts()

# There is a total 11113 non-subscription app and 57721 subscription app. 

# Subset the `app_info` to get a df with all subscription-based apps. 

subscribe_app_index = app_info['subscription'] == "Yes"

subscribe_app_info = app_info[subscribe_app_index]

np.sort(subscribe_app_info.index.values)

app_rank.shape

subscribe_rank_df = app_rank.loc[subscribe_app_info.index]

subscribe_rank2_df = subscribe_rank_df.copy()

subscribe_rank_df.shape


# Any rows that contains all NAN values? 

sum(subscribe_rank2_df.isna().sum(axis = 1) ==849)


# Doing forward fill

subscribe_rank_df.head()

clean_subscribe_rank_df = subscribe_rank2_df.ffill(axis = 1)

# Get the last 365 days of the rank data. 

clean_subscribe_rank_df_365 = clean_subscribe_rank_df.iloc[:,-365:].copy()

# The missing value here comes from the app does not exist in the app store yet. Fill the rest of missing values with 300

clean_subscribe_rank_df_365.fillna(300, inplace = True)

# No more missing values.

sum(clean_subscribe_rank_df_365.isna().sum())


# get the growth rate through a simple linear regression

# There are many approaches to compare growth rate among apps. I choose to do a simple linear regression on the rank data vs timestep. Then, the slope is extracted. A more negative value in slope means a quicker growth while a more positive slope means a decrease in rank. 

def get_growth_rate(rank_data):
    x = range(365)
    y = rank_data
    slope, intercept, r_value, p_value, std_err = linregress(x,y)
    if p_value < 0.05:
        return slope
    elif p_value >= 0.05:
        return 0    

growth_rate_list = [get_growth_rate(clean_subscribe_rank_df_365.iloc[i,]) for i in range(clean_subscribe_rank_df_365.shape[0])]

clean_subscribe_rank_df_365['growth_rate'] = growth_rate_list

# Sort the dataframe by the growth rate and find out the top 300 apps that have the best growth in the past 365 days. 

growth_df = clean_subscribe_rank_df_365.sort_values(by = "growth_rate").iloc[:300,-2:]

app_info_growth_df = growth_df.join(subscribe_app_info , on = "itunes_app_id", how = "left")

app_info_growth_df.head()

# Write the data to disk. 

app_info_growth_df.to_csv("top300_subscription_app_info.csv")

