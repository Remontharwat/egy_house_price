
import pandas as pd
import streamlit as st
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from category_encoders import BinaryEncoder
from sklearn.preprocessing import StandardScaler , RobustScaler , MinMaxScaler , OneHotEncoder ,PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression ,Ridge ,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV ,cross_validate
from sklearn.ensemble import RandomForestRegressor , VotingRegressor
from sklearn.metrics import mean_absolute_error , mean_squared_error ,r2_score
from xgboost import XGBRFRegressor

pipeline=joblib.load('Price Prediction ML model.h5')
inputs=joblib.load('input.h5')


def predict(Type, Bedrooms, Bathrooms, Area, Furnished, Level,Payment_Option, Delivery_Term):
    test_df=pd.DataFrame(columns=inputs)
    test_df.at[0,'Type']=Type
    test_df.at[0,'Bedrooms']=Bedrooms
    test_df.at[0,'Bathrooms']=Bathrooms
    test_df.at[0,'Area']=Area
    test_df.at[0,'Furnished']=Furnished
    test_df.at[0,'Level']=Level
    test_df.at[0,'Payment_Option']=Payment_Option
    test_df.at[0,'Delivery_Term']=Delivery_Term
    
    log_result= pipeline.predict(test_df)[0]
    result = np.exp(log_result())
    return result

def main():
    st.image('image.jpg')
    st.title('Egyption House Price Prediction')
    Type= st.selectbox('Type',['Apartment','Chalet','Stand Alone Villa','Town House','Twin House','Duplex','Penthouse','Studio'])
    Bedrooms= st.slider('Bedrooms',min_value= 1,max_value= 10,value=1,step=1)
    Bathrooms= st.slider('Bathrooms',min_value= 1,max_value= 10,value=1,step=1)
    Area= st.slider('Area',min_value= 25,max_value= 1000,value=100,step=1)
    Furnished= st.selectbox('Furnished',['No', 'Yes'])
    Level= st.slider('Level',min_value= 0,max_value= 12,value=0,step=1)
    Payment_Option= st.selectbox('Payment_Option',['Cash', 'Cash or Installment', 'Installment'])
    Delivery_Term= st.selectbox('Delivery_Term',['Finished', 'Semi Finished', 'Core & Shell', 'Not Finished'])
    

    if st.button('predict'):
        result=predict(Type, Bedrooms, Bathrooms, Area, Furnished, Level,Payment_Option, Delivery_Term)
        st.write("Unit price is : ".format(result))
if __name__ =='__main__':
    main()
