#============================================================== importing libraries and packages ===============================================================
from statistics import mode
import pandas as pd
import numpy as np

# import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# is_noun = lambda pos: pos[:2] == 'NN'

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import MinMaxScaler
#============================================================ End importing libraries and packages =============================================================

#==================================================================== RFM helper functions ====================================================================
# quartile score based on RFM values
def RScore(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4
def FMScore(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1

# customer segments based on rfm
def getRS(x):
  if x>141: return "Inactive"
  elif x<141 and x>=50: return "Temporarily idle"
  elif x<50 and x>=17: return "Frequent"
  else: return "Highly active"
def getFS(x):
  if x>=5: return "Stellar customer"
  elif x<5 and x>=2: return "Regulars"
  elif x<2 and x>=1: return "Returning customers"
  else: return "Potential"
def getMS(x):
  if x>=1627.625000: return "Best"
  elif x<1627.625000 and x>=659.680000: return "Top 50%"
  elif x<659.680000 and x>=303.455000: return "Top 75%"
  else: return "Worst"

# defining rfm levels
def rfm_level(df):
    if df['RFM_Sum'] >= 9:
        return 'Require Activation'
    elif ((df['RFM_Sum'] >= 8) and (df['RFM_Sum'] < 9)):
        return 'Needs Attention'
    elif ((df['RFM_Sum'] >= 7) and (df['RFM_Sum'] < 8)):
        return 'Promising'
    elif ((df['RFM_Sum'] >= 6) and (df['RFM_Sum'] < 7)):
        return 'Potential'
    elif ((df['RFM_Sum'] >= 5) and (df['RFM_Sum'] < 6)):
        return 'Loyal'
    elif ((df['RFM_Sum'] >= 4) and (df['RFM_Sum'] < 5)):
        return 'Champions'
    else:
        return 'Can\'t Loose Them'

# Preparing RFM Dataframe with all the needed Columns
def prepare_rfm_df(df1, today):
    # creating dataframe with RFM raw values and CustomerID column
    custom_aggregation = {
        "InvoiceDate" : lambda x: (today - x.max()).days,
        "InvoiceNo" : lambda x: len(x),
        "TotalPrice" : "sum"
    }
    rfmTable_main = df1.groupby('CustomerID', as_index=False).agg(custom_aggregation)
    rfmTable_main.columns = ["CustomerID","Recency","Frequency","Monetary"]

    # defining RFM quartile values    
    quantiles = rfmTable_main.quantile(q=[0.25,0.5,0.75])
    quantiles = quantiles.to_dict()

    rfmTable_main['r_quartile'] = rfmTable_main['Recency'].apply(RScore, args=('Recency',quantiles,))
    rfmTable_main['f_quartile'] = rfmTable_main['Frequency'].apply(FMScore, args=('Frequency',quantiles,))
    rfmTable_main['m_quartile'] = rfmTable_main['Monetary'].apply(FMScore, args=('Monetary',quantiles,))

    # string concatenation of RFM quartiles for RFM Score
    rfmTable_main['RFMScore'] = rfmTable_main.r_quartile.map(str) + rfmTable_main.f_quartile.map(str) + rfmTable_main.m_quartile.map(str)

    # defining individual levels based on RFM each
    rfmTable_main["R_Level"]    = [getRS(i) for i in rfmTable_main["Recency"]]
    rfmTable_main["F_Level"]  = [getFS(i) for i in rfmTable_main["Frequency"]]
    rfmTable_main["M_Level"]   = [getMS(i) for i in rfmTable_main["Monetary"]]
    
    #adding rfm score
    rfmTable_main['RFM_Sum'] = rfmTable_main[['r_quartile','f_quartile','m_quartile']].sum(axis=1)

    # Create a new variable RFM_Level
    rfmTable_main['RFM_Level'] = rfmTable_main.apply(rfm_level, axis=1)

    return rfmTable_main
#================================================================== End RFM helper functions ==================================================================
"""
#============================================================ Product Categorization helper functions =========================================================
# extract keywords from the description
def keywords_inventory(dataframe, colonne = 'Description'):
    stemmer = nltk.stem.SnowballStemmer("english")
    keywords_roots  = dict()  # collect the words / root
    keywords_select = dict()  # association: root <-> keyword
    category_keys   = []
    count_keywords  = dict()
    icount = 0
    for s in dataframe[colonne]:
        if pd.isnull(s): continue
        lines = s.lower()
        tokenized = nltk.word_tokenize(lines)
        nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 
        
        for t in nouns:
            t = t.lower() ; racine = stemmer.stem(t)
            if racine in keywords_roots:                
                keywords_roots[racine].add(t)
                count_keywords[racine] += 1                
            else:
                keywords_roots[racine] = {t}
                count_keywords[racine] = 1
    
    for s in keywords_roots.keys():
        if len(keywords_roots[s]) > 1:  
            min_length = 1000
            for k in keywords_roots[s]:
                if len(k) < min_length:
                    clef = k ; min_length = len(k)            
            category_keys.append(clef)
            keywords_select[s] = clef
        else:
            category_keys.append(list(keywords_roots[s])[0])
            keywords_select[s] = list(keywords_roots[s])[0]
                   
    return category_keys, keywords_roots, keywords_select, count_keywords

# perform product categorization
def prod_categorization(data, clean_data):
    # obtain no. of unique keywords in Description column
    product_data = pd.DataFrame(data['Description'].unique()).rename(columns = {0:'Description'})
    keywords, keywords_roots, keywords_select, count_keywords = keywords_inventory(product_data)

    # Some words are not used, so we consider only words that have occured atleast 15 times
    list_products = []
    for k,v in count_keywords.items():
        word = keywords_select[k]
        if word in ['pink', 'blue', 'tag', 'green', 'orange']: continue
        if len(word) < 3 or v < 15: continue
        if ('+' in word) or ('/' in word): continue
        list_products.append([word, v])
    list_products.sort(key = lambda x:x[1], reverse = True)

    # encode the data for further processing, so data must be equally distributed
    product_list = clean_data['Description'].unique()
    X = pd.DataFrame()
    for key, occurence in list_products:
        X.loc[:, key] = list(map(lambda x:int(key.upper() in x), product_list))

    threshold = [0, 1, 2, 3, 5, 10]
    label_col = []
    for i in range(len(threshold)):
        if i == len(threshold)-1:
            col = '.>{}'.format(threshold[i])
        else:
            col = '{}<.<{}'.format(threshold[i],threshold[i+1])
        label_col.append(col)
        X.loc[:, col] = 0
    
    for i, prod in enumerate(product_list):
        prix = clean_data[clean_data['Description'] == prod]['UnitPrice'].mean()
        j = 0
        while prix > threshold[j]:
            j+=1
            if j == len(threshold): break
        X.loc[i, label_col[j-1]] = 1
    
    matrix = X.to_numpy()
    # choosing and creating product clusters
    n_clusters = 5
    silhouette_avg = -1
    while silhouette_avg < 0.145:
        kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=30)
        kmeans.fit(matrix)
        clusters = kmeans.predict(matrix)
        silhouette_avg = silhouette_score(matrix, clusters)
        # print average silhouette score
        # print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
    
    # creating product catogories & define cluster of each product and store in categ_product
    corresp = dict()
    for key, val in zip (product_list, clusters):
        corresp[key] = val 
    clean_data['categ_product'] = clean_data.loc[:, 'Description'].map(corresp)

    # calculate the amount spent in each product category 
    for i in range(5):
        col = 'categ_{}'.format(i)
        df_temp = clean_data[clean_data['categ_product'] == i]
        price_temp = df_temp['UnitPrice'] * (df_temp['Quantity'] - df_temp['QuantityCanceled'])
        price_temp = price_temp.apply(lambda x:x if x > 0 else 0)
        clean_data.loc[:, col] = price_temp
        clean_data[col].fillna(0, inplace = True)
    
    return clean_data
#========================================================== End Product Categorization helper functions ========================================================
"""
#============================================================== kmeans clustering helper functions =============================================================
def kmeans_clustering(df, columns, req_cust_clusters):
    # clustering based on passed list of columns
    clustered_fm = df[columns].copy()
    
    # standardizing values
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(clustered_fm)
    data_scaled = pd.DataFrame(x_scaled)

    # applying kmeans for customer based clusters
    kmeans = KMeans(n_clusters = req_cust_clusters, init='k-means++', n_init =10, max_iter = 300)
    kmeans.fit(data_scaled)

    y_kmeans = kmeans.predict(data_scaled)
    return y_kmeans
#============================================================ End kmeans clustering helper functions ===========================================================

#=============================================================== Analyse clusters helper functions =============================================================
def analyse_clusters(final_df):
    # index labels for visualization dataframe
    index_simple = ['No of Customers']
    index_mean = ['Quantity', 'QuantityCanceled', 'CancellationRate', 'Recency', 'Frequency', 'Monetary', 'RFM_Sum']
    index_median = ['RFMScore']
    # column labels for visualization dataframe
    col = sorted(final_df["cluster"].unique())

    # visualization dataframe template preparation
    visual_final = pd.DataFrame(index=index_simple+index_mean+index_median, columns=col)
    # dataframe for cluster label encoding
    encode_cluster = visual_final.copy(deep=True)
    encode_cluster = encode_cluster.T

    # creating dataframe for insights visualization
    for i in col:
        visual_final[i] = pd.concat([final_df.loc[final_df['cluster']==i, 'CustomerID'], final_df.loc[final_df['cluster']==i, index_mean].mean(), final_df.loc[final_df['cluster']==i, index_median].median()])
        visual_final.loc['No of Customers', i] = len(final_df[final_df['cluster']==i]['CustomerID'].unique())

    # creating dataframe for cluster naming
    quantiles = visual_final.T.quantile(q=[0.25,0.5,0.75])
    quantiles = quantiles.to_dict()

    high_is_better = ['No of Customers', 'Quantity', 'Frequency', 'Monetary', 'RFM_Sum']
    # less_is_better = ['QuantityCanceled', 'CancellationRate', 'Recency', 'RFMScore']

    for i in visual_final.T.columns:
        if i in high_is_better:
            encode_cluster[i] = visual_final.T[i].apply(FMScore, args=(i, quantiles,))
        else:
            encode_cluster[i] = visual_final.T[i].apply(RScore, args=(i, quantiles,))

    naming_dictionary = {
        1 : 'Loyal',
        2 : 'Bulls eye',
        3 : 'Promising',
        4 : 'Need attention'
    }

    cluster_names = dict()
    for i in encode_cluster.T:
        cluster_names[naming_dictionary[mode(encode_cluster.T[i])]] = i

    return visual_final, cluster_names
#============================================================= End Analyse clusters helper functions ===========================================================

#=================================================================== Main Driver Functions =====================================================================
#initial cleaning and sanitization of data
def sanitize_data(initial_df):
    # dropping rows with empty ColumnID
    initial_df.dropna(subset = ['CustomerID'], inplace = True)
    # droppping duplicate entries
    initial_df.drop_duplicates(inplace = True)

    # dropping rows with following stock codes ['POST', 'D', 'C2', 'M', 'BANK CHARGES', 'PADS', 'DOT']
    # as above rows won't be needed for understanding data
    for i in ['POST', 'D', 'C2', 'M', 'BANK CHARGES', 'PADS', 'DOT']:
        initial_df = initial_df[initial_df['StockCode'] != i]
    
    # Cleaning Dataframe by analyzing Cancelling of Orders 
    clean_df = initial_df.copy(deep = True)
    clean_df['QuantityCanceled'] = 0

    # The cancelled orders with counterparts
    entry_to_remove = [] 
    # The cancelled orders without counterparts
    doubtfull_entry = []

    for index, col in initial_df.iterrows():
        if (col['Quantity'] > 0) or col['Description'] == 'Discount': continue        
        test_data = initial_df[
            (initial_df['CustomerID'] == col['CustomerID']) &
            (initial_df['StockCode'] == col['StockCode']) & 
            (initial_df['InvoiceDate'] < col['InvoiceDate']) & 
            (initial_df['Quantity'] > 0)
        ].copy()
        # Here we calculate the customers who have cancelled without counterparts
        # (i.e. without ordering any other product in cancelled order's place)
        if (test_data.shape[0] == 0): 
            doubtfull_entry.append(index)
        
        # Here we calculate the customers who have cancelled with counterparts
        # (i.e. ordering any other product in cancelled order's place)
        elif (test_data.shape[0] == 1): 
            index_order = test_data.index[0]
            clean_df.loc[index_order, 'QuantityCanceled'] = -col['Quantity']
            entry_to_remove.append(index)  

        # If various counterparts are present, first one is removed
        elif (test_data.shape[0] > 1): 
            test_data.sort_index(axis=0 ,ascending=False, inplace = True)        
            for ind, val in test_data.iterrows():
                if val['Quantity'] < -col['Quantity']: continue
                clean_df.loc[ind, 'QuantityCanceled'] = -col['Quantity']
                entry_to_remove.append(index) 
                break
    
    # As doubtful entries are less in number as compared to total records
    # So, we drop both
    clean_df.drop(entry_to_remove, axis = 0, inplace = True)
    clean_df.drop(doubtfull_entry, axis = 0, inplace = True)

    remaining_entries = clean_df[(clean_df['Quantity'] < 0) & (clean_df['StockCode'] != 'D')]
    clean_df.drop(remaining_entries.index, axis = 0, inplace = True)
    
    return clean_df

# preprocessing of data -> 1. adding new columns 2. grouping invoices and customers 3. adding RFM columns
def preprocess_Data(raw_data, mode):
    # print('Before Sanitization: {}'.format(raw_data.shape))
    # cleaning and sanitization of Data
    clean_df = sanitize_data(raw_data)
    # print('After Sanitization: {}'.format(clean_df.shape))
    """# product categorization -> new columns: (categ_product and categ_N (0<=N<5))
    clean_df = prod_categorization(raw_data, clean_df)
    # print('After product categorization: {}'.format(clean_df.shape))"""
    # Total Price = Unit Price * (Quantity - QuantityCanceled)
    clean_df['TotalPrice'] = clean_df['UnitPrice'] * (clean_df['Quantity'] - clean_df['QuantityCanceled'])
    # For orders per month and per hour insights
    clean_df['InvoiceDate'] = pd.to_datetime(clean_df['InvoiceDate'])
    clean_df['Month'] = clean_df["InvoiceDate"].map(lambda x: x.month)
    clean_df['Hour'] = clean_df["InvoiceDate"].map(lambda x: x.hour)

    # concat new data with old clean data here

    # grouping products of each invoice
    custom_aggregation = {
        # "categ_0" : "sum",
        # "categ_1" : "sum",
        # "categ_2" : "sum",
        # "categ_3" : "sum",
        # "categ_4" : "sum",
        "Quantity" : "sum",
        "InvoiceDate" : lambda x:x.iloc[0],
        "CustomerID" : lambda x:x.iloc[0],
        "Country" : lambda x:x.iloc[0],
        "QuantityCanceled" : "sum",
        "TotalPrice" : "sum"
    }
    df1 = clean_df.groupby("InvoiceNo",as_index=False).agg(custom_aggregation)
    # print('After product grouping for each invoice: {}'.format(df1.shape))

    # grouping invoices of each customer
    custom_aggregation = {
        # "categ_0" : "sum",
        # "categ_1" : "sum",
        # "categ_2" : "sum",
        # "categ_3" : "sum",
        # "categ_4" : "sum",
        "Quantity" : "sum",
        "Country" : lambda x:x.iloc[0],
        "QuantityCanceled" : "sum"
    }
    df1_final = df1.groupby('CustomerID',as_index=False).agg(custom_aggregation)
    df1_final['CancellationRate'] = (df1_final["QuantityCanceled"]/df1_final["Quantity"]).astype('float64')
    # add cluster column to df1_final here initially with zeros(0)
    df1_final["cluster"] = 0
    
    # print('After invoice grouping for each customer: {}'.format(df1_final.shape))

    # if old dataset is present and append new to it -> clean_df
    if mode == 'update':
        clean_df_old = pd.read_csv('../dataset/clean_data.csv')
        clean_df = pd.concat([clean_df_old, clean_df], axis=0)

    # function to prepare complete RFM Dataframe
    rfm_main = prepare_rfm_df(df1, clean_df['InvoiceDate'].max())
    # print('After preparing RFM Data: {}'.format(rfm_main.shape))

    #merging df1_final and rfmTable_main
    final_df = df1_final.merge(rfm_main, sort=False)

    # if old dataset is present and append new to it -> final_df
    if mode == 'update':
        final_df_old = pd.read_csv('../dataset/final_data.csv')
        final_df = pd.concat([final_df_old, final_df], axis=0)
    
    # refill cluster column values in final_df here
    final_df["cluster"] = kmeans_clustering(final_df, ['Recency',"Frequency","Monetary"], 4)

    return clean_df, final_df
#================================================================= End Main Driver Functions ===================================================================