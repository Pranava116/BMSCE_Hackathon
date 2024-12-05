import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
a=input("enter the body temperature:")
b=input("enter the SPO2:")
c=input("enter the Heart Rate:")
data = {
    'body temperature': [22, 25, 30, 35, 20, 18, 27, 29, 32, 34],  
    'SPO2': [97,99,95, 96, 80, 100, 94, 85, 89,92 ],  
    'Heart Rate': [60, 70, 80, 90, 50, 40, 75, 85, 65, 55],     
    'outbreak': [0, 0, 1, 1, 0, 0, 1, 1, 1, 1]              
    
}


df = pd.DataFrame(data)


print(df)
X = df[['body temperature', 'SPO2', 'Heart Rate']] 
y = df['outbreak']                                   


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)




print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()