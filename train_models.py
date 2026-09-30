import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

DATA="data/crop_yield_modeling.csv"
df=pd.read_csv(DATA)
features=["Crop","State","Cost of Cultivation (`/Hectare) A2+FL","Cost of Cultivation (`/Hectare) C2","Cost of Production (`/Quintal) C2"]
X,y=df[features],df["Yield"]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
num=features[2:]; cat=features[:2]
pre=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("ohe",OneHotEncoder(handle_unknown="ignore"))]),cat)])
models={"linear_regression":LinearRegression(),"decision_tree":DecisionTreeRegressor(max_depth=6,random_state=42),"random_forest":RandomForestRegressor(n_estimators=300,random_state=42),"xgboost":XGBRegressor(n_estimators=300,max_depth=4,learning_rate=.05,subsample=.9,colsample_bytree=.9,objective="reg:squarederror",random_state=42)}
rows=[]
for name,m in models.items():
 p=Pipeline([("prep",pre),("model",m)]); p.fit(Xtr,ytr); pred=p.predict(Xte); rows.append({"Model":name,"RMSE":mean_squared_error(yte,pred)**.5,"R2":r2_score(yte,pred)}); joblib.dump(p,f"models/{name}.joblib")
pd.DataFrame(rows).to_csv("results/model_comparison.csv",index=False)
print(pd.DataFrame(rows))
