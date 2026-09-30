import streamlit as st, pandas as pd, joblib, glob
st.set_page_config(page_title="Crop Yield Predictor",layout="wide")
st.title("🌾 Agriculture Crop Yield Prediction")
st.write("Predict crop yield (Quintal/Hectare) using historical cultivation-cost and location features.")
df=pd.read_csv("data/crop_yield_modeling.csv")
crops=sorted(df.Crop.unique()); states=sorted(df.State.unique())
with st.sidebar:
 crop=st.selectbox("Crop",crops); state=st.selectbox("State",states)
 a2=st.number_input("Cost of Cultivation A2+FL",min_value=0.0,value=float(df.iloc[0][df.columns[2]]))
 c2=st.number_input("Cost of Cultivation C2",min_value=0.0,value=float(df.iloc[0][df.columns[3]]))
 cp=st.number_input("Cost of Production C2",min_value=0.0,value=float(df.iloc[0][df.columns[4]]))
 model_name=st.selectbox("Model",["linear_regression","decision_tree","random_forest","xgboost"])
if st.button("Predict Yield"):
 model=joblib.load(f"models/{model_name}.joblib"); X=pd.DataFrame([{"Crop":crop,"State":state,"Cost of Cultivation (`/Hectare) A2+FL":a2,"Cost of Cultivation (`/Hectare) C2":c2,"Cost of Production (`/Quintal) C2":cp}]); pred=model.predict(X)[0]; st.metric("Predicted Yield",f"{pred:.2f} Quintal/Hectare")
st.subheader("Dataset overview"); st.dataframe(df.head(20),use_container_width=True)
