from os import path, makedirs
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
# Validate Customer ID input
@st.cache
def check_input(text):
    if re.match(r'^([\d]+)$', text) and int(float(text)) > 0:
        return True
    else:
        return False

# get key from a dictionary based on value 
@st.cache
def get_key(val, my_dict): 
    for key, value in my_dict.items(): 
         if val == value: 
             return key 
    return "key doesn't exist"

#================================================== Checking and loading dataset files ================================================
# Check if dataset is present or not
@st.cache
def check_dataset():
    # If folder doesn't exist, then create it.
    if not path.exists('dataset'):
        makedirs('dataset')
        return False

    if path.exists("./dataset/clean_data.csv") and path.exists("./dataset/final_data.csv"):
        return True
    else:
        return False
#================================================ End checking and loading dataset files ==============================================

#================================================ Visualize Insights based on Data Files ==============================================
# visualize raw dataset
def visualize_dataset(final_df, cluster_insights_df, cluster_names_dict):
    # 1. Average basket price per customer in each cluster
    st.header('Avg. Basket Price per customer in each cluster:')
    min_monetary = cluster_insights_df.loc['Monetary'].min()
    fig = go.Figure(
        data=[
            go.Scatter(
                x = [get_key(i, cluster_names_dict) for i in cluster_insights_df.columns],
                y = [cluster_insights_df.loc['Monetary',i] for i in cluster_insights_df.columns],
                mode = 'markers',
                marker = dict(
                    color = [236, 189, 648, 135],
                    size = [round((cluster_insights_df.loc['Monetary',i]/min_monetary)*20) for i in cluster_insights_df.columns],
                    showscale = True
                )
            )
        ]
    )
    st.plotly_chart(fig, use_column_width=True)
    # 1. End average basket price per customer in each cluster

    # 2. Average frequency per customer in each cluster
    st.header('Avg. Frequency per customer in each cluster:')
    min_freq = cluster_insights_df.loc['Frequency'].min()
    fig = go.Figure(
        data=[
            go.Scatter(
                x = [get_key(i, cluster_names_dict) for i in cluster_insights_df.columns],
                y = [cluster_insights_df.loc['Frequency',i] for i in cluster_insights_df.columns],
                mode = 'markers',
                marker = dict(
                    color = [236, 189, 648, 135],
                    size = [round((cluster_insights_df.loc['Frequency',i]/min_freq)*20) for i in cluster_insights_df.columns],
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

# visualize each customer based on customer ID input
def visualize_customer(clean_df, final_df, cust_id, cluster_names_dict):
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
    st.info('**Note:** _Quantity_, _QuantityCanceled_, _CancellationRate_ depict the _total_ values for each customer.')
    st.info('**Note:** _r_quartile_, _f_quartile_, _m_quartile_ depicts score of _Recency_, _Frequency_, _Monetary_ from 1(Best) to 4(Worst). _RFM_Sum_ depicts sum of _r_quartile_, _f_quartile_ and _m_quartile_')
    # 3. End customer Insights

# visual each cluster based on dropdown menu
def visualize_cluster(clean_df, final_df, selected_cluster, cluster_insights_df, cluster_names_dict):
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
    st.markdown("\
    <div class='markdown-text-container mb-3'>\
        <h3 class='text-center pt-0 mt-0 text-pink'><u>Feature explanation</u></h3>\
        <table class='table table-bordered table-hover table-danger border-pink w-75 mx-auto pt-0'>\
            <thead class='border-pink text-pink'>\
                <tr><th class=''>Features / Variables</th><th class=''>Calculation Criteria</th></tr>\
            </thead>\
            <tbody>\
                <tr><th scope='row'>No of Customers</th><td>Count</td></tr>\
                <tr><th scope='row'>RFMScore</th><td>Median</td></tr>\
                <tr><th scope='row'>Others</th><td>Mean(Average)</td></tr>\
            </tbody>\
        </table>\
    </div>", unsafe_allow_html=True)
    
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
#============================================== End Visualize Insights based on Data Files ============================================
#==================================================== End User defined functions ======================================================

#====================================================== Loading UI template ============================================================
def home_page(clean_df, final_df, cluster_insights_df, cluster_names_dict):
    st.sidebar.success('Loading Data... Done!')
    # Custom styling rules
    st.markdown("<title>Customer Segmentation</title>\
        <style>\
            .text-pink{color: rgb(246, 51, 102);}\
            .border-pink{border: 2px solid rgb(246, 51, 102);}\
            .section-sep {line-height: 1em; position: relative; outline: 0; outline: none; border: none !important; color: black; text-align: center; height: 1.4em; opacity: .8;}\
            .section-sep:before {content: ''; background: -webkit-gradient(linear, left top, right top, from(transparent), color-stop(#f63366, #fffd80), to(transparent)); background: linear-gradient(to right, transparent, #f63366, transparent); position: absolute; left: 0; top: 50%; width: 100%; height: 4px;}\
            .section-sep:after {content: attr(data-content); position: relative; display: inline-block; color: black; padding: 0 .5em; line-height: 1.5em; color: #818078; background-color: #fff;}\
        </style>", unsafe_allow_html=True)
    # End custom styling rules

    # [A] Analyse Dataset
    if st.sidebar.checkbox('Raw Data Insights', key='raw_data_toggle', value='True'):
        # Title
        st.markdown("<h1 class='text-pink'>Raw Data Insights</h1>", unsafe_allow_html=True)
        visualize_dataset(final_df, cluster_insights_df, cluster_names_dict)
        st.markdown("<hr class='section-sep' data-content='End of Raw Data Insights'/>", unsafe_allow_html=True)
    # [A] End Analyse Dataset

    # [B] Analyse Customer
    if st.sidebar.checkbox('Customer Insights', key='customer_toggle'):
        # Title
        st.markdown("<h1 class='text-pink'>Customer Insights</h1>", unsafe_allow_html=True)
        cust_id = st.text_input('Customer ID')
        if check_input( cust_id ) and ( int(float(cust_id)) in set(final_df['CustomerID']) ):
            visualize_customer(clean_df, final_df, cust_id, cluster_names_dict)
        else:
            if cust_id != '':
                st.error('Invalid Customer ID!')
        st.markdown("<hr class='section-sep' data-content='End of Customer Insights'/>", unsafe_allow_html=True)
    # [B] End Analyse Customer

    # [C] Analyse Clusters
    if st.sidebar.checkbox('Cluster Insights', key='cluster_toggle'):
        selected_cluster = st.sidebar.selectbox('Select Cluster', ('Loyal', 'Bulls eye', 'Promising', 'Need attention'))
        # Title
        st.markdown("<h1 class='text-pink'>Cluster Insights</h1>", unsafe_allow_html=True)
        visualize_cluster(clean_df, final_df, selected_cluster, cluster_insights_df, cluster_names_dict)
        # End
        st.markdown("<hr class='section-sep' data-content='End of Cluster Insights'/>", unsafe_allow_html=True)
    # [C] End Analyse Clusters

    # [D] Upload updated / new Data
    if st.sidebar.checkbox('Upload new data', key='new_upload_toggle'):
        # Title
        st.markdown("<h1 class='text-pink'>Upload New Data</h1>", unsafe_allow_html=True)
        update_data = st.file_uploader("Choose a CSV file", encoding="unicode_escape", type="csv", key="update_dataset")
        if update_data is not None:
            new_data = pd.read_csv(update_data)

            # Processing uploaded / found dataset file
            data_process_state = st.info('Processing data...')
            # updating loaded dataset
            clean_df, final_df = preprocess_Data(new_data, 'update')
            cluster_insights_df, cluster_names_dict = analyse_clusters(final_df)
            # End processing dataset files
            
            # Saving processed files
            data_process_state.info('Saving processed data...')
            # 1. Updating Raw Data
            raw_data_old = pd.read_csv('./dataset/data.csv')
            raw_data_old = pd.concat([raw_data_old, new_data], axis=0)
            raw_data_old.csv('./dataset/data.csv', index=False)
            # 2. Updating Clean Data
            clean_df.to_csv("./dataset/clean_data.csv", index=False)
            # 3. Updating Final Data
            final_df.to_csv("./dataset/final_data.csv", index=False)
            data_process_state.info('Saving processed data...done')
            # End saving processed files
        # End    
        st.markdown("<hr class='section-sep' data-content='End of Upload Data Section'/>", unsafe_allow_html=True)
    # [D] End Upload updated / new Data
#====================================================== End of UI template ============================================================

#====================================================== Main Driver Script ============================================================
if __name__ == "__main__":
    data_found = False
    # Check if dataset files are missing
    if check_dataset() == False:
        default_dataset = "./dataset/data.csv"
        # check for raw data file
        if path.exists(default_dataset):
            data_state = st.success('Data found!')
            new_data = pd.read_csv(default_dataset, encoding="unicode_escape")
            # Processing uploaded / found dataset file
            data_state.info('Processing data...')
            clean_df, final_df = preprocess_Data(new_data, 'new')
            cluster_insights_df, cluster_names_dict = analyse_clusters(final_df)
            # End processing dataset files
            
            # Saving processed files
            data_state.info('Saving processed data...')
            clean_df.to_csv("./dataset/clean_data.csv", index=False)
            final_df.to_csv("./dataset/final_data.csv", index=False)
            data_state.info('Saving processed data...done')
            # End saving processed files
            data_found = True
        else:
            data_state = st.warning('Data not found!')
            st.title('Upload to continue')
            uploaded_file = st.file_uploader("Choose a CSV file", encoding="unicode_escape", type="csv", key="create_dataset")
            if uploaded_file is not None:
                new_data = pd.read_csv(uploaded_file)
                # Processing uploaded / found dataset file
                data_state.info('Processing data...')
                clean_df, final_df = preprocess_Data(new_data, 'new')
                cluster_insights_df, cluster_names_dict = analyse_clusters(final_df)
                # End processing dataset files
                
                # Saving processed files
                data_state.info('Saving processed data...')
                new_data.to_csv('./dataset/data.csv', index=False)
                clean_df.to_csv("./dataset/clean_data.csv", index=False)
                final_df.to_csv("./dataset/final_data.csv", index=False)
                data_state.info('Saving processed data...done')
                # End saving processed files
                data_found = True
    else:
        # Loading dataset files
        clean_df = pd.read_csv('./dataset/clean_data.csv', encoding='unicode_escape')
        final_df = pd.read_csv('./dataset/final_data.csv', encoding='unicode_escape')
        cluster_insights_df, cluster_names_dict = analyse_clusters(final_df)
        # End loading dataset files
        data_found = True

    if data_found == True:
        home_page(clean_df, final_df, cluster_insights_df, cluster_names_dict)
#==================================================== End Main Driver Script ==========================================================