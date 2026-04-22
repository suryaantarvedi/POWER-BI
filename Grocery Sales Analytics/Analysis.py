import pandas as pd

def load_and_clean_data(path):
    df=pd.read_csv(path)
    print("Shape:", df.shape)
    print("Duplicate Rows:", df.duplicated().sum())
    df=df.drop_duplicates()
    print("Missing Values:\n", df.isnull().sum())
    df=df.dropna()
    df['Date']=pd.to_datetime(df['Date'], dayfirst=True)
    num_cols=['Quantity', 'Unit_Price_INR', 'Discount_Percent', 'Total_Amount_INR']
    df[num_cols]=df[num_cols].apply(pd.to_numeric)
    df=df[(df['Quantity']>0)&(df['Unit_Price_INR']>0)]
    df=df[(df['Discount_Percent']>=0)&(df['Discount_Percent']<=80)]
    df['High_Discount_Flag']=df['Discount_Percent']>=20
    return df

def detect_outliers(df):
    cols=['Quantity', 'Unit_Price_INR', 'Total_Amount_INR']
    def detect_iqr(col):
        Q1=df[col].quantile(0.25)
        Q3=df[col].quantile(0.75)
        IQR=Q3-Q1
        lower=Q1-1.5*IQR
        upper=Q3+1.5*IQR
        return (df[col]<lower)|(df[col]>upper)
    for col in cols:
        df[col + '_outlier'] = detect_iqr(col)
    df['any_outlier']=(df['Quantity_outlier']|df['Unit_Price_INR_outlier']|df['Total_Amount_INR_outlier'])
    print("Outlier Counts:\n", df[[col + '_outlier' for col in cols]].sum())
    print("Total outlier rows:", df['any_outlier'].sum())
    return df

def feature_engineering(df):
    df['Calculated_Total']=df['Unit_Price_INR']*df['Quantity']*(1-df['Discount_Percent']/100)
    df['Mismatch']=abs(df['Total_Amount_INR']-df['Calculated_Total'])
    df['Mismatch_Flag']=df['Mismatch']>=1
    df['Gross_Revenue']=df['Unit_Price_INR']*df['Quantity']
    df['Discount_Amount']=df['Gross_Revenue']-df['Total_Amount_INR']
    df['Month']=df['Date'].dt.month
    df['Quarter']=df['Date'].dt.quarter
    df['Week']=df['Date'].dt.isocalendar().week.astype(int)
    df['Revenue_Per_Unit']=df['Total_Amount_INR']/df['Quantity']
    return df

# PROBLEM - 1
def discount_analysis(df):
    result = df.groupby('Category').agg({
        'Discount_Percent': 'mean',
        'Total_Amount_INR': 'sum',
        'Quantity': 'sum',
        'Transaction_ID': 'count'
    }).rename(columns={'Transaction_ID': 'Transactions'})
    result['Avg_Rev_Per_Transaction']=result['Total_Amount_INR']/result['Transactions']
    result['Avg_Qty_Per_Transaction']=result['Quantity']/result['Transactions']
    result['Revenue_Loss']=df.groupby('Category')['Discount_Amount'].mean()
    result=result.sort_values(by='Discount_Percent', ascending=False)
    print(result['Discount_Percent'])

    print("\nRevenue Loss by Category:")
    print(result['Revenue_Loss'].sort_values(ascending=False))

    corr=df['Discount_Percent'].corr(df['Quantity'])
    print("\nCorrelation (Discount vs Quantity):", corr)

#PROBLEM - 2
def seasonal_analysis(df):
    monthly_sales=df.groupby('Month')['Total_Amount_INR'].sum()
    quarterly_sales=df.groupby('Quarter')['Total_Amount_INR'].sum()
    print("\nMonthly Sales:\n", monthly_sales.sort_values(ascending=False))
    print("\nQuarterly Sales:\n", quarterly_sales.sort_values(ascending=False))
    best_month=monthly_sales.idxmax()
    best_quarter=quarterly_sales.idxmax()
    print("\nBest Month:",best_month)
    print("Best Quarter:",best_quarter)

    category_monthly=df.groupby(['Category', 'Month'])['Total_Amount_INR'].sum().reset_index()
    pivot=category_monthly.pivot(index='Category', columns='Month', values='Total_Amount_INR').fillna(0)
    pivot['Mean']=pivot.mean(axis=1)
    pivot['Std']=pivot.std(axis=1)
    pivot['Seasonality_Index']=pivot['Std']/pivot['Mean']
    seasonality=pivot[['Seasonality_Index']].sort_values(by='Seasonality_Index', ascending=True)
    print("\nCategory Seasonality Ranking:\n")
    print(seasonality)

    peak_month=pivot.iloc[:, :12].idxmax(axis=1)
    print("\nPeak Sales Month per Category:\n", peak_month)

    # Rank categories by seasonality
    seasonality_rank=pivot[['Seasonality_Index']].sort_values(by='Seasonality_Index', ascending=True)
    # Top 3 most seasonal categories
    print("\nMost Seasonal Categories:")
    print(seasonality_rank.head(3))

    # Least seasonal
    print("\nLeast Seasonal Categories:")
    print(seasonality_rank.tail(3))

#PROBLEM - 3
def city_and_region_performance(df):
    city_revenue=df.groupby('City')['Total_Amount_INR'].sum().sort_values(ascending=False)
    print("\nCity-wise Revenue:")
    print(city_revenue)

    region_revenue=df.groupby('Region')['Total_Amount_INR'].sum().sort_values(ascending=False)
    print("\nRegion-wise Revenue:")
    print(region_revenue)

    city_transactions=df.groupby('City')['Transaction_ID'].count().sort_values(ascending=False)
    print("\nTransactions per City:")
    print(city_transactions)

    print("\nTop City:",city_revenue.idxmax())
    print("Bottom City:",city_revenue.idxmin())

#PROBLE - 4
def customer_analysis(df):
    avg_spend=df.groupby('Customer_Age_Group')['Total_Amount_INR'].mean()
    print("\nAverage Spend per Age Group:\n", avg_spend)

    category_age=df.groupby(['Customer_Age_Group', 'Category'])['Total_Amount_INR'].sum().reset_index()
    category_age=category_age.sort_values(['Customer_Age_Group', 'Total_Amount_INR'], ascending=[True, False])

    top3=category_age.groupby('Customer_Age_Group').head(3)
    top3_list=top3.groupby('Customer_Age_Group')['Category'].apply(list)

    print("\nTop 3 Categories:")
    for age, cats in top3_list.items():
        print(f"{age}: {cats}")

    upi_users=df[df['Payment_Method'] == 'UPI']
    upi_count=upi_users.groupby('Customer_Age_Group').size()
    total_count=df.groupby('Customer_Age_Group').size()
    upi_percentage=(upi_count / total_count)*100
    print("\nUPI Usage Percentage by Age Group:")
    print(upi_percentage)


#PROBLEM - 5
def payment_analysis(df):
    share=(df['Payment_Method'].value_counts()/len(df))*100
    print("\nShare of each payment method:\n", share)

    trend=df[df['Payment_Method'].isin(['UPI', 'Cash'])] \
        .groupby(['Month', 'Payment_Method']).size().unstack()
    print("\nUPI vs Cash Monthly Trend:\n", trend)

    df['Is_Digital']=df['Payment_Method'].isin(['UPI', 'Debit Card', 'Credit Card'])
    city_digital=df.groupby('City')['Is_Digital'].mean() * 100
    print("\nDigital Payment % by City:\n", city_digital.sort_values(ascending=False))


#PROBLEM - 6
def product_analysis(df):
    total_revenue=df['Total_Amount_INR'].sum()
    category_revenue=df.groupby('Category')['Total_Amount_INR'].sum()
    revenue_share=(category_revenue / total_revenue) * 100
    revenue_share_df=revenue_share.reset_index()
    revenue_share_df.columns=['Category', 'Revenue_Share_%']
    revenue_share_df=revenue_share_df.sort_values(by='Revenue_Share_%', ascending=False)
    revenue_share_df['Cumulative_%'] = revenue_share_df['Revenue_Share_%'].cumsum()
    print("\nRevenue Share with Cumulative %:")
    print(revenue_share_df)

    print("\nTop 10 Products:\n", df.groupby('Product_Name')['Total_Amount_INR'].sum().nlargest(10))
    print("\nSlow-moving Products:\n", df.groupby('Product_Name')['Quantity'].sum().nsmallest(10))

#PROBLEM - 7
def store_analysis(df):
    store_revenue=df.groupby('Store_ID')['Total_Amount_INR'].sum().sort_values(ascending=False)
    print("\nTop 10 Stores:\n", store_revenue.head(10))
    print("\nBottom 10 Stores:\n", store_revenue.tail(10))

    avg_txn=df.groupby('Store_ID')['Total_Amount_INR'].mean()
    print("\nAverage Transaction Value per Store:")
    print(avg_txn)
    city_avg = df.groupby('City')['Total_Amount_INR'].mean()
    print("\nCity Average Transaction Value:")
    print(city_avg.sort_values(ascending=False))

    store_city=df[['Store_ID', 'City']].drop_duplicates()
    store_city=store_city.merge(avg_txn, on='Store_ID')
    store_city.rename(columns={'Total_Amount_INR': 'Store_Avg'}, inplace=True)
    store_city['City_Avg']=store_city['City'].map(city_avg)
    below_avg=store_city[store_city['Store_Avg'] < store_city['City_Avg']]
    print("\nStores Below City Avg:\n", below_avg)

# -------------------------------
# EXPORT CLEANED DATASET
# -------------------------------
def export_clean_data(df, path="cleaned_grocery_data.csv"):

    df_export = df.drop(columns=['Calculated_Total', 'Mismatch', 'Quantity_outlier','Unit_Price_INR_outlier', 'Total_Amount_INR_outlier',
    'any_outlier','Mismatch_Flag'], errors='ignore')
    df_export.to_csv(path, index=False)
    print(f"\nCleaned dataset exported successfully")

def main():
    df=load_and_clean_data("data/Grocery_Store_sales_Dataset.csv")
    df=detect_outliers(df)
    df=feature_engineering(df)

    discount_analysis(df)
    seasonal_analysis(df)
    city_and_region_performance(df)
    customer_analysis(df)
    payment_analysis(df)
    product_analysis(df)
    store_analysis(df)
    export_clean_data(df)

if __name__ == "__main__":
    main()