import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.metrics import confusion_matrix, classification_report , accuracy_score
retail=pd.read_csv('retail_sales.csv')
print(retail.info())
print(retail)
#Drop the unnecessary columns
retail=retail.drop(columns=['Transaction ID','Date', 'Customer ID', 'Gender'])
print(retail.shape)
X =retail.drop("Product Category",axis=1)
y =retail["Product Category"]

# Map diagnosis to numerical values
#retail["Product Category"] = retail["Product Category"].map({"Beauty": 0, "Clothing": 1, "Electronics": 2})

#print(retail.head())
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_test)
print(y_test)
print('this is x train')
print(X_train)
print('this is y train')
print(y_train)

dt=DecisionTreeClassifier(random_state =42)
dt.fit(X_train,y_train)
y_pred=dt.predict(X_test)

DT_acc= accuracy_score(y_test,y_pred)
print(DT_acc)

X_train_norm = preprocessing.normalize(X_train)
X_test_norm = preprocessing.normalize(X_test)

#train or fitting and evaluation
model = KMeans (n_clusters =3,random_state =0,n_init ='auto')
model.fit(X_train_norm)

K=range(2,8)
fits=[]
score=[]
#train the model for current values of k on training data
for k in K:
    model = KMeans(n_clusters=k,random_state =0,n_init ='auto').fit(X_train_norm)
                  #append the model to fits
    fits.append(model)
                  #append the silhouette_score
    score.append(silhouette_score(X_train_norm,model.labels_,metric ='euclidean'))
#Select K based on the best silhoutte score
best_k_idx =score.index(max(score))
best_k =K[best_k_idx]
best_model =fits[best_k_idx]
print(fits)
print(score)
#y_pred=km.predict(X_test)
#km_acc= accuracy_score(y_test,y_pred)
#print(km_acc)

log_reg =LogisticRegression(max_iter=1000)
log_reg.fit(X_train,y_train)
y_pred=log_reg.predict(X_test)
log_acc=accuracy_score(y_test,y_pred)
print(log_acc)

from sklearn.ensemble import RandomForestClassifier
RFmodel=RandomForestClassifier(random_state=0).fit(X_train,y_train)
y_pred=RFmodel.predict(X_test)
RF_acc=accuracy_score(y_test,y_pred)
print(RF_acc)

from sklearn.ensemble import GradientBoostingClassifier
GBmodel=GradientBoostingClassifier(random_state=0).fit(X_train,y_train)
y_pred=GBmodel.predict(X_test)
GB_acc=accuracy_score(y_test,y_pred)
print(GB_acc)

models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree','Random Forest',  
              'GradientBoosting','KMeans'],
    'Score': [log_acc, DT_acc, RF_acc, GB_acc,score[best_k_idx]]})
print(models.sort_values(by='Score', ascending=False))


