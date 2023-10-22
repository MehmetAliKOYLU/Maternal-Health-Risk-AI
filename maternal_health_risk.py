import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# we import the dataset from kaggle or our github repository 
#https://github.com/MehmetAliKOYLU?tab=repositories


#https://www.kaggle.com/datasets/joebeachcapital/maternal-health-risk
data = pd.read_csv("Maternal Health Risk Data Set.csv")
data.head()

#if the patient is high risk we will get the result = High Risk , middle risk we will get the result = MidRisk , if the patient is low risk we will get the result = LowRisk
highrisk = data[data.RiskLevel == "high risk"]
midrisk = data[data.RiskLevel == "mid risk"]
lowrisk = data[data.RiskLevel == "low risk"]


plt.scatter(lowrisk.Age, lowrisk.BS, color="green", label="Lowrisk", alpha = 0.4)
plt.scatter(midrisk.Age, midrisk.BS, color="blue",label="MidRisk",alpha=0.4)
plt.scatter(highrisk.Age, highrisk.BS, color="red", label="HighRisk", alpha = 0.4)
plt.xlabel("Age")
plt.ylabel("BS")

plt.legend()
plt.savefig('Maternal-health-risk.png', dpi=300)
plt.show()


y= data.RiskLevel.values

x = data.drop(["RiskLevel"],axis=1)



x_train, x_test, y_train, y_test =train_test_split (x,y,test_size = 0.1,random_state=5)







################################################################ for k values
k_values = list(range(1, 10))

mean_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_train, y_train, cv=5)  # 5x cross value score
    mean_scores.append(scores.mean())

# Optimal k values =
optimal_k = k_values[mean_scores.index(max(mean_scores))]
print("Optimal k value:", optimal_k)

#create final model with Optimal k values
final_knn = KNeighborsClassifier(n_neighbors=optimal_k)
final_knn.fit(x_train, y_train)

# Accuracy with optimal k value
accuracy = final_knn.score(x_test, y_test)
print(f"Accuracy with optimal k value: %{accuracy * 100:.2f}")
sc = MinMaxScaler()
sc.fit_transform(x.values)
###############################################################




############################################################### for random forest classifier
RFCmodel = RandomForestClassifier()  
RFCmodel.fit(x_train,y_train)
rfc_pred = RFCmodel.predict(x_test)
rfc_acc = accuracy_score(rfc_pred, y_test)
print ("Random forest test accuracy: {:.2f}%".format(rfc_acc*100))
print( "\n" )
print(classification_report(y_test, rfc_pred))
print( "\n" )
###############################################################




#we are using random forest classifier for training and testing purposes because accuracy score is better than other classifiers
def newprediction():
    v1 = int(input("Enter your age: "))
    v2 = int(input("Enter your Systolic BP: "))
    v3 = int(input("Enter your Diastolic BP: "))
    v4 = float(input("Enter your BS: "))  # Change this from "Blood Sugar" to "BS"
    v5 = int(input("Enter your BodyTemp: "))  # Change this from "Body Temperature" to "BodyTemp"
    v6 = int(input("Enter your HeartRate: "))

    new_prediction = RFCmodel.predict(np.array([[v1, v2, v3, v4, v5, v6]]))

    if new_prediction[0] == "low risk":
        print("You are in the Low Risk category. Take good care of your health.")
    elif new_prediction[0] == "mid risk":
        print("You are in the Mid Risk category. Pay attention to your health and consult a healthcare professional.")
    elif new_prediction[0] == "high risk":
        print("You are in the High Risk category. It's essential to consult a healthcare professional immediately.")
    else:
        print("Invalid prediction.")

while True:
    newprediction()
    choose = input("Do you want to continue? (y/n): ")
    choose = choose.lower()
    if choose == "y":
        continue
    elif choose == "n":
        print("Exiting...")
        break
