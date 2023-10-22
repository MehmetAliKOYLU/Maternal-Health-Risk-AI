# Maternal-Health-Risk-AI
This AI, Shows whether you are at risk of having a maternal health

### These are the libraries We will use for the project
```
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
```
### And i want to see first 5 patients in my dataset
```
data = pd.read_csv("Maternal Health Risk Data Set.csv")
data.head()
```
![image](https://github.com/MehmetAliKOYLU/Maternal-Health-Risk-AI/assets/91757385/bef3b408-5422-4b20-82af-78e2fd688be2)

#### Let's make an example drawing just by looking at breathing speed(BS) for now At the end of our program, our machine learning model will make a prediction by looking not only at BS, but also at all other data
```
plt.scatter(lowrisk.Age, lowrisk.BS, color="green", label="Lowrisk", alpha = 0.4)
plt.scatter(midrisk.Age, midrisk.BS, color="blue",label="MidRisk",alpha=0.4)
plt.scatter(highrisk.Age, highrisk.BS, color="red", label="HighRisk", alpha = 0.4)
plt.xlabel("Age")
plt.ylabel("BS")

plt.legend()
plt.savefig('Maternal-health-risk.png', dpi=300)
plt.show()
```
![image](https://github.com/MehmetAliKOYLU/Maternal-Health-Risk-AI/assets/91757385/49a664eb-00ad-4baa-956a-6f18a884a8e1)


y= data.RiskLevel.values
x = data.drop(["RiskLevel"],axis=1)

## we separate our test data with our train data
### our train data will be used to learn how the system distinguishes between a healthy person and a sick person. if our test data is, let's see if our machine learning model can accurately distinguish between sick and healthy people it will be used for testing...
`
x_train, x_test, y_train, y_test =train_test_split (x,y,test_size = 0.1,random_state=5)
`
### for k values

```
#############################################################
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
sc.fit_transform(x_raw_data.values)
###############################################################
```


### for random forest classifier
```

############################################################
RFCmodel = RandomForestClassifier()  
RFCmodel.fit(x_train,y_train)
rfc_pred = RFCmodel.predict(x_test)
rfc_acc = accuracy_score(rfc_pred, y_test)
print ("Random forest test accuracy: {:.2f}%".format(rfc_acc*100))
print( "\n" )
print(classification_report(y_test, rfc_pred))
print( "\n" )

###############################################################
```
![image](https://github.com/MehmetAliKOYLU/Maternal-Health-Risk-AI/assets/91757385/1efd757a-0928-492f-9a0c-330158649b46)

### we are using random forest classifier for training and testing purposes because accuracy score is better than other classifiers
```
def newprediction():
    v1=int(input("age >> "))
    v2=int(input("SystolicBP >> "))
    v3=int(input("DiastolicBP >> "))
    v4=float(input("BS >> "))
    v5=int(input("BodyTemp >> "))
    v6=int(input("HeartRate >> "))

    new_prediction = RFCmodel.predict(sc.transform(np.array([[v1,v2,v3,v4,v5,v6]])))
    new_prediction[0]
    print(new_prediction[0])
```

### and we use while loop for a new predictions
```
while True:
    newprediction()
    choose=input("do you want to continue ? (y/n)  >>"  )
    choose=choose.lower()
    if choose=="y":
        continue
    elif choose=="n":
        print("exiting...")
        break
```
### this is our last output
![image](https://github.com/MehmetAliKOYLU/Maternal-Health-Risk-AI/assets/91757385/046669b1-b4ec-44c6-9d86-ec08b56a95d6)

