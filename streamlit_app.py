import streamlit as st
import joblib
import time
model = joblib.load("model.pkl")
st.title("ShopIQ AI / Sumit Singh")
age = st.number_input("Enter your Age:")
gender = st.radio("Enter Male(1) , Female(0)",[1,2])
anumal = st.number_input("Enter the Anual Income:")
NumberOfPurchases = st.number_input("Enter number of purchase In single Month:")
TimeSpentOnWebsite = st.number_input("Enter TimeSpentOnWebsite(min): ")
DiscountsAvailed = st.number_input("Enter DiscountsAvailed:")
click = st.button("Predict")
if click:
    with  st.spinner(text="wait..." , show_time=True):
        time.sleep(3)
    st.success("Done!")
    st.button("Rerun")
    begain = model.predict([[age,gender,anumal,NumberOfPurchases,TimeSpentOnWebsite,DiscountsAvailed]])
    if begain == 1:
        st.success("✅Customer is likely to return")
    else:
        st.error("❌ Customer is unlikely to return")





#cd C:\Users\sumit\OneDrive\Pictures\Documents

#python -m streamlit run streamlit_app.py
