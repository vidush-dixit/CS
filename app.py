from os import path
import time
from PIL import Image

import pandas as pd
import numpy as np

from statistics import mode
import re

import matplotlib.pyplot as plt
import plotly.graph_objects as go

import streamlit as st
from modules.model import *

#====================================================== User defined functions ========================================================
def check_input(text):
    if re.match(r'^([\d]+)$', text) and int(float(text)) > 0:
        return True
    else:
        return False

def check_dataset():
    if path.exists("./dataset/clean_data.csv") and path.exists("./dataset/final_data.csv"):
        return True
    else:
        return False

# function to return key for any value 
def get_key(val, my_dict): 
    for key, value in my_dict.items(): 
         if val == value: 
             return key 
    return "key doesn't exist"

def visualize_dataset(final_df, cluster_insights_df, cluster_names_dict):
    st.title('Raw Data Insights')
    # 1. Average basket price per customer in each cluster
    st.header('Avg. Basket Price per customer in each cluster:')
    fig = go.Figure(
        data=[
            go.Scatter(
                x = [1,2,3,4],
                y = [cluster_insights_df.loc['Monetary',i] for i in cluster_insights_df.columns],
                mode = 'markers',
                marker = dict(
                    color = [236, 189, 648, 135],
                    size = [58, 42, 144, 30],
                    showscale = True
                )
            )
        ]
    )
    st.plotly_chart(fig, use_column_width=True)
    # 1. End average basket price per customer in each cluster

    # 2. Average frequency per customer in each cluster
    st.header('Avg. Frequency per customer in each cluster:')
    fig = go.Figure(
        data=[
            go.Scatter(
                x = [1,2,3,4],
                y = [cluster_insights_df.loc['Frequency',i] for i in cluster_insights_df.columns],
                mode = 'markers',
                marker = dict(
                    color = [236, 189, 648, 135],
                    size = [58, 42, 144, 30],
                    showscale = True
                )
            )
        ]
    )
    st.plotly_chart(fig, use_column_width=True)
    # 2. End average frequency per customer in each cluster

    # 3. Top 5 countries by Monetary value
    st.header('Geographical Insights:')
    top_country = final_df.groupby('Country')['Monetary'].sum().sort_values(ascending=False)[:10]

    labels = top_country[:5].index
    size = top_country[:5].values

    plt.figure(figsize=(10,7))
    plt.pie(size, labels=labels, explode=[0.05]*5, autopct='%1.0f%%')
    plt.title("Top 5 Countries by Total Sales", size=15)
    plt.axis('equal')
    st.pyplot()
    # 3. Top 5 countries by Monetary value

    #4. Orders per month
    st.header('Seasonal Insights:')
    plt.figure(figsize = (10,7))
    n, bins, patches = plt.hist(clean_df['Month'], bins=12)
    plt.title("Number of orders per month")
    plt.xlabel("Months")
    plt.xticks([rect.get_x()+ rect.get_width() / 2 for rect in patches], [x for x in ['Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']])

    for rect in patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        space = 5
        va = 'bottom'
        label = str(int(y_value))
        
        plt.annotate(
            label,
            (x_value, y_value),
            xytext=(0, space),
            textcoords="offset points",
            ha='center',
            va=va)
    st.pyplot()
    #4. End orders per month

def visualize_customer(clean_df, final_df, cust_id, cluster_names_dict):
    st.title('Customer Insights')
    # 1. Customer cluster info
    cust_clust = list(final_df[final_df['CustomerID'] == int(float(cust_id))]['cluster'])[0]
    st.info('Customer category -> {}'.format(get_key(cust_clust, cluster_names_dict)))
    # 1. End customer cluster info

    # 2. Customer order history
    st.header('Customer Order History:')
    st.write(clean_df.loc[clean_df['CustomerID'] == int(float(cust_id)),:].reset_index(drop=True))
    # 2. End customer order history

    # 3. Customer Insights
    st.header('Customer Insights:')
    temp = final_df.loc[final_df['CustomerID'] == int(float(cust_id)),:].T
    temp.columns = ['Values']
    temp.index.name = 'Features'
    temp.reset_index(inplace=True)
    st.table(temp)
    # 3. End customer Insights

def visualize_cluster(clean_df, final_df, selected_cluster, cluster_insights_df, cluster_names_dict):
    st.title('Cluster Insights')
    cluster_no = cluster_names_dict[selected_cluster]
    cust = list(final_df[final_df['cluster'] == cluster_no]['CustomerID'])
    ith_cluster = clean_df[clean_df['CustomerID'].isin(cust)]

    # 1. Segment size insights using pie chart
    st.header('Segment Size:')
    # Data to plot
    labels = selected_cluster, 'Others'
    sizes = [len(cust), (len(final_df['CustomerID'])- len(cust))]
    colors = ['gold', 'yellowgreen']
    explode = (0.1, 0)  # explode 1st slice
    # Plot
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')
    st.pyplot()
    # 1. End segment size insights using pie chart

    # 2. Cluster columns insights dataframe
    st.header('Cluster Summary:')
    temp = pd.DataFrame(cluster_insights_df[cluster_no])
    temp.columns = ['Values']
    temp.index.name = 'Features'
    temp.reset_index(inplace=True)
    st.table(temp)
    st.warning('Features Explanation')
    st.write(pd.DataFrame({
        'Columns': ['No of Customer', 'RFMScore', 'Rest'],
        'Calculation Insights': ['Count', 'Median', 'Mean(Average)']
        }))
    # 2. End cluster columns insights dataframe

    #3. Top 10 brought product
    st.header('Top 10 products bought:')
    temp = pd.DataFrame(ith_cluster['Description'].value_counts()[:10])
    temp.columns = ['Quantity Purchased']
    temp.index.name = 'Products'
    temp.reset_index(inplace=True)
    st.table(temp)
    #3. End top 10 brought product
    
    #4. Orders per month
    st.header('Seasonal Insights:')
    plt.figure(figsize = (10,7))
    n, bins, patches = plt.hist(ith_cluster['Month'], bins=12)
    plt.title("Number of orders per month")
    plt.xlabel("Months")
    plt.xticks([rect.get_x()+ rect.get_width() / 2 for rect in patches], [x for x in ['Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']])

    for rect in patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        space = 5
        va = 'bottom'
        label = str(int(y_value))
        
        plt.annotate(
            label,
            (x_value, y_value),
            xytext=(0, space),
            textcoords="offset points",
            ha='center',
            va=va)
    st.pyplot()
    #4. End orders per month
    
    #5. Orders per hour
    st.header('Hourly Usage Trends:')
    plt.figure(figsize = (10,7))
    n, bins, patches = plt.hist(ith_cluster['Hour'], bins=ith_cluster['Hour'].nunique())
    plt.title("Number of orders per hour")
    plt.xlabel("Hours")
    plt.xticks([rect.get_x()+ rect.get_width() / 2 for rect in patches], [x for x in (sorted(ith_cluster['Hour'].unique()))])

    for rect in patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        space = 5
        va = 'bottom'
        label = str(int(y_value))
        
        plt.annotate(
            label,
            (x_value, y_value),
            xytext=(0, space),
            textcoords="offset points",
            ha='center',
            va=va)
    st.pyplot()
    #5. End Orders per hour

    # 6. Characteristics and Recommendations
    st.header('Miscellaneous:')
    image = Image.open('./static/img/'+selected_cluster+'.png')
    st.image(image, use_column_width=True)
    # 6. Characteristics and Recommendations
#==================================================== End User defined functions ======================================================

#================================================== Checking and loading dataset files ================================================
# Check if dataset files are missing
if check_dataset() == False:
    default_dataset = "./dataset/data.csv"
    if path.exists(default_dataset):
        st.success('Data found!')
        new_data = pd.read_csv(default_dataset, encoding="unicode_escape")
    else:
        st.warning('No data found!')
        st.title('Upload to continue')
        uploaded_file = st.file_uploader("Choose a CSV file", encoding="unicode_escape", type="csv", key="create_dataset")
        if uploaded_file is not None:
            new_data = pd.read_csv(uploaded_file)
     
    # Processing uploaded / found dataset file
    data_process_state = st.info('Processing data...')
    clean_df, final_df = preprocess_Data(new_data, 'new')
    # End processing dataset files
    # Saving processed files
    data_process_state.info('Saving processed data...')
    clean_df.to_csv("./dataset/clean_data.csv", index=False)
    final_df.to_csv("./dataset/final_data.csv", index=False)
    data_process_state.info('Saving processed data...done')
    # End saving processed files
#================================================ End checking and loading dataset files ==============================================

#========================================================= UI template ================================================================
# Loading dataset files
clean_df = pd.read_csv('./dataset/clean_data.csv', encoding='unicode_escape')
final_df = pd.read_csv('./dataset/final_data.csv', encoding='unicode_escape')
cluster_insights_df, cluster_names_dict = analyse_clusters(final_df)
st.sidebar.success('Data loading... Done!')
# End loading dataset files

# [A] Analyse Dataset
if st.sidebar.checkbox('Raw Data Insights', key='raw_data_toggle', value='True'):
    visualize_dataset(final_df, cluster_insights_df, cluster_names_dict)
# [A] End Analyse Dataset

# [B] Analyse Customer
if st.sidebar.checkbox('Customer Insights', key='customer_toggle'):
    cust_id = st.text_input('Customer ID')
    if check_input( cust_id ) and ( int(float(cust_id)) in set(final_df['CustomerID']) ):
        visualize_customer(clean_df, final_df, cust_id, cluster_names_dict)
    else:
        if cust_id != '':
            st.error('Invalid Customer ID!')
# [B] End Analyse Customer

# [C] Analyse Clusters
if st.sidebar.checkbox('Cluster Insights', key='cluster_toggle'):
    selected_cluster = st.sidebar.selectbox('Select Cluster', ('Loyal', 'Bulls eye', 'Promising', 'Need attention'))
    visualize_cluster(clean_df, final_df, selected_cluster, cluster_insights_df, cluster_names_dict)
# [C] End Analyse Clusters

# [D] Upload updated / new Data
if st.sidebar.checkbox('Upload new data', key='new_upload_toggle'):
    st.title('Upload new data')
    update_data = st.file_uploader("Choose a CSV file", encoding="unicode_escape", type="csv", key="update_dataset")
    if update_data is not None:
        new_data = pd.read_csv(update_data)
        data_process_state = st.info('Processing data...')
        clean_df, final_df = preprocess_Data(new_data, 'update')
        cluster_insights_df, cluster_names_dict = analyse_clusters(final_df)
# [D] End Upload updated / new Data
#====================================================== End of UI template ============================================================