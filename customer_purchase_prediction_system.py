import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import joblib
file = pd.read_csv(r"C:\Users\sumit\OneDrive\Pictures\Documents\customer_purchase_data.csv")
frame = pd.DataFrame(file)
inputs = frame[["Age","Gender","AnnualIncome","NumberOfPurchases","TimeSpentOnWebsite","DiscountsAvailed"]]
output = frame["PurchaseStatus"]
x_train,x_test,y_train,y_test = train_test_split(inputs,output,train_size=0.8,test_size=0.2)
model = LGBMClassifier(random_state=42)
model.fit(x_train,y_train)
joblib.dump(model,"model.pkl")
print("done")
