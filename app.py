"""PixelLens: interactive five-model handwritten-digit classification dashboard."""
from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from metrics_utils import FEATURES, TARGET, LABELS, METRICS, validate_test_csv, evaluate

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="PixelLens | Digit Classification",page_icon="🔢",layout="wide")
st.markdown("""
<style>
.block-container {max-width:1150px; padding-top:1.4rem;}
h1 {color:#28366b;letter-spacing:-.04em;}
[data-testid="stMetric"] {border:1px solid #d8def0;border-radius:12px;padding:13px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    info = json.loads((ROOT/"metadata.json").read_text())
    fitted = {name:joblib.load(ROOT/filename) for name,filename in info["model_index"].items()}
    return info,fitted

@st.cache_data
def packaged_test():
    return pd.read_csv(ROOT/"test_data.csv")

info,models=load_models()
st.title("PixelLens · Classification Studio")
st.caption("Five trained classifiers · UCI-derived handwritten digits · 64 features · 10 classes")
st.write("Upload labeled test data, select any saved classifier and inspect performance. The app never retrains models.")
with st.sidebar:
    st.header("Experiment controls")
    choice=st.selectbox("Model",list(models),index=list(models).index(info["winner_by_weighted_f1"]))
    upload=st.file_uploader("Upload labeled test CSV",type="csv",help="All 64 pixel features and digit target required.")
    st.download_button("Download test_data.csv",(ROOT/"test_data.csv").read_bytes(),"test_data.csv","text/csv")
    st.info("Only predictions and evaluation are performed. No fitting on uploaded data.")

try:
    raw=pd.read_csv(upload) if upload else packaged_test()
    data=validate_test_csv(raw)
except (ValueError,pd.errors.ParserError,UnicodeDecodeError) as error:
    st.error(f"Invalid test CSV: {error}")
    st.stop()

st.caption(f"Evaluating {len(data)} labeled test records using pre-trained saved models.")
results=pd.DataFrame([{"Model":name,**evaluate(model,data)} for name,model in models.items()])
chosen=results.set_index("Model").loc[choice]
cols=st.columns(6)
for col,metric in zip(cols,METRICS):
    val=chosen[metric]
    col.metric(metric,"N/A" if pd.isna(val) else f"{val:.4f}")

tab1,tab2,tab3=st.tabs(["Confusion matrix & report","Compare all five models","Explore a digit"])
with tab1:
    model=models[choice]
    prediction=model.predict(data[FEATURES])
    left,right=st.columns([1.1,1])
    with left:
        fig,ax=plt.subplots(figsize=(6.4,5.3))
        ConfusionMatrixDisplay.from_predictions(
            data[TARGET],prediction,labels=LABELS,ax=ax,cmap="Blues",values_format="d",colorbar=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("Classification report")
        report=classification_report(
            data[TARGET],prediction,labels=LABELS,output_dict=True,zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(3),use_container_width=True)
        st.caption("Weighted precision, recall & F1; weighted one-vs-rest multiclass AUC; multiclass MCC.")
with tab2:
    st.subheader("Same uploaded data · all five saved classifiers")
    st.dataframe(results.set_index("Model").round(4),use_container_width=True)
    st.bar_chart(results.set_index("Model")[["Accuracy","F1"]],horizontal=True)
    st.download_button("Download evaluation CSV",results.to_csv(index=False).encode(),"evaluation.csv","text/csv")
with tab3:
    rowindex=st.slider("Choose test example",0,len(data)-1,0)
    sample=data.iloc[rowindex]
    features=data.iloc[[rowindex]][FEATURES]
    predicted=int(models[choice].predict(features)[0])
    left,right=st.columns(2)
    with left:
        fig,ax=plt.subplots(figsize=(4,4))
        ax.imshow(sample[FEATURES].to_numpy(dtype=float).reshape(8,8),
                  cmap="magma",vmin=0,vmax=16)
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.metric("True digit",str(int(sample[TARGET])))
        st.metric("Predicted digit",str(predicted))
        st.write("Correct" if predicted==int(sample[TARGET]) else "Incorrect")
        top=pd.DataFrame({"Digit":models[choice].classes_,
                          "Probability":models[choice].predict_proba(features)[0]})
        st.dataframe(top.sort_values("Probability",ascending=False).head(5),hide_index=True)
st.caption("Dataset attribution: UCI Optical Recognition of Handwritten Digits, sklearn 1,797-record subset.")
