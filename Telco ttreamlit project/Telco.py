import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import scale, StandardScaler

df = pickle.load(open("Telco.pkl", "rb"))
features = pickle.load(open("features", "rb"))
model = pickle.load(open("log_model", "rb"))
X = pickle.load(open("X", "rb"))
model2 = pickle.load(open("xgb_model", "rb"))

#st.write(df.head())
#st.table(df.head())
#st.dataframe(df.head())


# st.sidebar.title("Churn Probability of a single Customer")
html_temp = """
<div style="background-color:tomato;padding:1.5px">
<h1 style="color:white;text-align:center;">Churn Prediction ML app </h1>
</div><br>"""
st.markdown(html_temp,unsafe_allow_html=True)

tenure=st.sidebar.slider("Number of months the customer has stayed with the company (tenure)", 1, 72, 20, step=1)
MonthlyCharges=st.sidebar.slider("The amount charged to the customer monthly", 0,100, 30, step=1)
TotalCharges=st.sidebar.slider("The total amount charged to the customer", 0,5000, 1500, step=1)
Contract=st.sidebar.selectbox("The contract term of the customer", ('Month-to-month', 'One year', 'Two year'))
OnlineSecurity=st.sidebar.selectbox("Whether the customer has online security or not", ('No', 'Yes', 'No internet service'))
InternetService=st.sidebar.selectbox("Customer’s internet service provider", ('DSL', 'Fiber optic', 'No'))
TechSupport=st.sidebar.selectbox("Whether the customer has tech support or not", ('No', 'Yes', 'No internet service'))


a = df._get_numeric_data()

def single_customer():
    my_dict = {"tenure" :tenure,
        "OnlineSecurity":OnlineSecurity,
        "Contract": Contract,
        "TotalCharges": TotalCharges ,
        "InternetService": InternetService,
        "TechSupport": TechSupport,
        "MonthlyCharges":MonthlyCharges}
 
    for i in a:
        my_dict[i] = (my_dict[i]-a[i].mean())/a[i].std()
   
    df_sample = pd.DataFrame.from_dict([my_dict])
    df_sample = pd.get_dummies(df_sample).reindex(columns=features, fill_value=0)
    return df_sample


df2 = single_customer()
proba = model2.predict_proba(df2)

if st.sidebar.button("Submit"):
    #result = message.title()
    st.sidebar.success(f"The churn probability of selected customer is % {proba[:,1][0]*100:.2f}")



def All_customer():
    pred_prob = model2.predict_proba(X)
    df["Churn Probabilty"] = np.round(pred_prob[:,1]*100, 2)
    return df


if st.checkbox("Churn Probability of Randomly Selected Customer"):
    st.markdown("### How many customer to be selected randomly?")
    a = st.selectbox("Please select the number of customers", (10, 50, 100))
    if st.button("Analize"):
        if a == 10:
            st.success(f"The churn probability of randomly {a} customers")
            st.table(All_customer().sample(a))
        elif a == 50:
            st.success(f"The churn probability of randomly {a} customers")
            st.table(All_customer().sample(a))
        else:
            st.success(f"The churn probability of randomly {a} customers")
            st.table(All_customer().sample(a))
       
    
    
    
if st.checkbox("Top Customer to Churn"):
    st.markdown("### How many customer to be selected?")
    a = st.selectbox("Please select the number of customers ", (10, 50, 100))
    if st.button("Show"):
        if a == 10:
            st.success(f"Top {a} customers to churn")
            st.table(All_customer().sort_values("Churn Probabilty", ascending = False).head(a))
        elif a == 50:
            st.success(f"Top {a} customers to churn")
            st.table(All_customer().sort_values("Churn Probabilty", ascending = False).head(a))
        else:
            st.success(f"Top {a} customers to churn")
            st.table(All_customer().sort_values("Churn Probabilty", ascending = False).head(a))
            
            
   


if st.checkbox("Top N Loyal Customer"):
    st.markdown("### How many customer to be selected?  ")
    a = st.selectbox("Please select the number of customers  ", (10, 50, 100))
    if st.button("Show "):
        if a == 10:
            st.success(f"Top {a} customers to be loyal")
            st.table(All_customer().sort_values("Churn Probabilty").head(a))
        elif a == 50:
            st.success(f"Top {a} customers to be loyal")
            st.table(All_customer().sort_values("Churn Probabilty").head(a))
        else:
            st.success(f"Top {a} customers to be loyal")
            st.table(All_customer().sort_values("Churn Probabilty").head(a))




    
    
 
    