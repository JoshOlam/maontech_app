"""

    Simple Streamlit webserver application for serving developed Regressiom model.

    Author: Joshua Olalemi.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data dependencies
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble._forest import RandomForestRegressor

df = pd.read_csv("data-1671661749260.csv")
#print("Done loading data...1")


def _data_preprocessing(df):
    """Private helper function to preprocess data for model prediction.
    All the codes required for feature engineering/selection are defined here.
    Parameters
    ----------
    df : str
        The data payload received within POST requests sent to our API.
    Returns
    -------
    X_data : A dataframe of all relevant features asides the target variable

    y_data : the target variable; 'ams'.
    """
    df = df[df['region']=="SW"]
    df = df.drop(['region', 'tms', 'year'], axis = 1)#, inplace=True)
    df = df.sort_values(by='month')
    df['ams'] = abs(df['ams'])
    df_dummies = pd.get_dummies(df)#, drop_first=True)

    #reindex the columns to make the target variable the last
    cols = [col for col in df_dummies.columns if col != 'ams'] + ['ams']

    df_dummies = df_dummies.reindex(columns=cols)
    
    X_data = df_dummies[[col for col in df_dummies.columns if col != "ams"]]
    y_data = df_dummies['ams']

    return X_data, y_data

def get_training_months(df:str, month: int):
    """Private helper function to get the training data
    for the previous 3 months.

    Parameters
    ----------
    df : str
        The dataframe to be subset.

    month : int
        The month (within the range of 1 and 12) to get the training data for 

    Returns
    -------
    df_train : A dataframe of the previous 3 months

    df_pred : A dataframe of the month provided
    """
    if month in range(1, 13):
        col = list(reversed(range(1,13)))
        col = col + [12,11,10]
        indx = col.index(month)
        train = col[indx+1:indx+4]
        df_pred = df[df['month']==month]
        df_train = pd.DataFrame()
        for i, j in enumerate(train):
            df_train = pd.concat([df_train,df[df['month']==j]], axis=0)

        return df_train, df_pred
    else:
        raise ValueError(f"Index '{month}' not within the range of 1 and 12")


def monthly_training(location: str, sku: str, month: int):
    """Private helper function to predict the average monthly
    sales (ams) based on a location, item_id and month.

    Parameters
    ----------
    location : str
        The location i.e. depot

    sku : str
        The item_id of the product

    month : int
        The month (within the range of 1 and 12)

    Returns
    -------
    Predictions : The predicted values of "ams"
    """
    X_data, y_data = _data_preprocessing(df)
    X_train, X_test, y_train, y_test = train_test_split(X_data,
                                                        y_data,
                                                        test_size=0.20,
                                                        random_state=42,
                                                        shuffle=False)
    #print(f"Done preprocessing")
    rdf_ = RandomForestRegressor(random_state=42)
    #rdf_.fit(X_train, y_train)
    rdf_.fit(X_data, y_data)
    #y_pred = rdf_.predict(X_test)
    #MSE = mean_squared_error(y_test, y_pred)
    #rmse = round(np.sqrt(MSE), 2)
    #print(f"RMSE: {rmse}")

    pred = df[(df['depot'] == location) & (df['item_no']==sku) & (df['month']==month)]
    #print(f"{pred.head()}")
    x, y = _data_preprocessing(pred)
    
    cols = [col for col in X_data.columns]

    df_pred = pd.DataFrame(columns=cols)
    
    for col in x.columns:
        df_pred[col] = x[col]

    df_pred.fillna(0, inplace=True)

    prediction = rdf_.predict(df_pred)
    
    return prediction

#For data preprocessing
def preprocess_df(df):
    #df = df[df['region']=="SW"]
    print("Done")
    df.drop(['region', 'tms', 'year'], axis = 1, inplace=True)
    df['ams'] = abs(df['ams'])

#raw = preprocess_df(raw)



# The main function where for the actual app
def main():
	st.title("ML Prediction Assessment")
	st.subheader("Average Monthly Sales Predictions")

	# Creating sidebar with selection box 
	options = ["Select here", "Prediction", "Information"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("""Dataset data info

- region - the region where the depot is located, i.e SW, NT, SE, LG
- depot - the name of the depot i.e ABEOKUTA, also known as `location`
- item_no - Product number i.e 10040447, also known as `SKU`.
- AMS - Average monthly sales
- month - Month number, i.e Jan = 1, Feb = 2, ..., Dec = 12
- year - Operation year. i.e 2022.""")

		st.subheader("Sales Prediction")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(df.head()) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("""AMS Prediction with ML Models \n
		Params: Location - The depot
		sku - The Item Number
		Month - The Month""")

		# Preprocess data
		df_cleaned = df.copy()
		cols = [col for col in df_cleaned.columns if col != 'ams'] + ["ams"]
		df_cleaned = df_cleaned.reindex(columns = cols)
		df_cleaned = df_cleaned[df_cleaned['region'] == "SW"]
		df_cleaned.drop(['region', 'tms', 'year', 'ams'], axis=1, inplace=True)
		#df_cleaned['ams'] = abs(df_cleaned['ams'])


		#For depot (also known as location)
		select_location = df_cleaned['depot'].unique()
		#st.write(type(locations))
		location = st.selectbox("Location", options=select_location)
		st.write('`You selected`', location, " `depot`")

		#For product number (also known as sku)
		#sku = [i for i in df_cleaned['item_no'].unique() if i != "FUNT"]
		sku = df_cleaned['item_no'].unique()
		Product_Number = st.selectbox("SKU", options=sku)
		st.write('`You selected`', Product_Number, " `item_no`")

		#For Month
		months = df['month'].unique()
		month = st.selectbox("Month", options=months)
		if st.checkbox("Show hint"):
			st.markdown("""
			⚡ Description of the Encoded Month ⚡ 
			|Month | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
			|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
			|Code  | `1` | `2` | `3` | `4` | `5` | `6` | `7` | `8` | `9` |`10` |`11` | `12`| 
			""")
			st.write("\n")
		st.write('`You selected`', month, " `month`")

		train_df = df_cleaned[(df_cleaned['depot'] == location) & (df_cleaned['item_no']==Product_Number) & (df_cleaned['month']==month)]
		train_df = train_df.set_index("depot")

		st.write(train_df)

		#pred_df = monthly_training(location = location, sku = Product_Number, month = month)

		if st.button("Get Predicted Value"):
			if training_df.shape[0] == 0:
				st.write("The provided parameters do not exist in the training")
			else:
				pred_df = monthly_training(location = location, sku = Product_Number, month = month)
				st.balloons()
				st.success("Predicted as: {}".format(pred_df))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
