# snipsets of code TS


# Make some features from date
grid_df['tm_d'] = grid_df['date'].dt.day.astype(np.int8)
grid_df['tm_w'] = grid_df['date'].dt.week.astype(np.int8)
grid_df['tm_m'] = grid_df['date'].dt.month.astype(np.int8)
grid_df['tm_y'] = grid_df['date'].dt.year
grid_df['tm_y'] = (grid_df['tm_y'] - grid_df['tm_y'].min()).astype(np.int8)
grid_df['tm_wm'] = grid_df['tm_d'].apply(lambda x: ceil(x/7)).astype(np.int8)

grid_df['tm_dw'] = grid_df['date'].dt.dayofweek.astype(np.int8)
grid_df['tm_w_end'] = (grid_df['tm_dw']>=5).astype(np.int8)

# Here is the main piece of code to generate the mean\median\whatever statistical feature you want
product_month_sales_median = df.groupby(["id", "month_id"])["sales"].agg("median")
product_month_sales_mean = df.groupby(["id", "month_id"])["sales"].agg("mean")

for months_window in range(1,3):   
 
    print("{} / 2".format(months_window))
    window_month = df["month_id"] - months_window
    monthId_product_list =list(zip(df.id, window_month))
    df["median_{}_months_ago".format(months_window)] =  
      `     pd.Series(monthId_product_list,
                                                      
    index = df.index).map(product_month_sales_median)
    df["median_{}_months_ago".format(months_window)] = 
    df["median_{}_months_ago".format(months_window)].round(1).
                   astype("float32")
    df["mean_{}_months_ago".format(months_window)] =   
                   pd.Series(monthId_product_list,                             
                   index = df.index).map(product_month_sales_mean)
    df["mean_{}_months_ago".format(months_window)] =   
                   df["mean_{}_months_ago".format(months_window)].
                   round(1).astype("float32")
            
            

            
start_time = time.time()

print('Specifics lags')

for ts in tqdm(lag_df['ts_id'].unique()):
    
    l_lag_spec = df_specific_l[df_specific_l['ts_id'] == ts]['lag'][:2]
    
    for idx, s_lag in enumerate(l_lag_spec):
        
        lag_df.loc[lag_df['ts_id'] == ts, 'lag_specific_' + str(idx)] = np.NaN
        lag_df.loc[lag_df['ts_id'] == ts, 'lag_specific_' + str(idx)] = \
            lag_df[lag_df['ts_id'] == ts].groupby(['ts_id'])['sales'].transform(lambda x: x.shift(s_lag))

print('Generics lags')

for g_lag in list_lag_generic:
    lag_df.loc[:, 'lag_genreric_' + str(g_lag)] = \
            lag_df.groupby(['ts_id'])['sales'].transform(lambda x: x.shift(g_lag))   
    
print('%0.2f min: Time for loops' % ((time.time() - start_time) / 60))





from lazypredict.Supervised import (  # pip install lazypredict
    LazyClassifier,
    LazyRegressor,
)
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Load data and split
X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit LazyRegressor
reg = LazyRegressor(
    ignore_warnings=True, random_state=1121218, verbose=False
  )
models, predictions = reg.fit(X_train, X_test, y_train, y_test)  # pass all sets

>>> models.head(10)