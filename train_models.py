"""Train five classifiers on scikit-learn's UCI-derived handwritten digits."""
from pathlib import Path
import hashlib
import json
import joblib
import pandas as pd
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from metrics_utils import FEATURES, TARGET, evaluate

ROOT = Path(__file__).resolve().parent
SEED = 42
MODELS = {
 "Logistic Regression": ("logistic_regression", Pipeline([
   ("standardize",StandardScaler()),
   ("classifier",LogisticRegression(max_iter=2200,C=1.0,solver="lbfgs",random_state=SEED))])),
 "Decision Tree": ("decision_tree", DecisionTreeClassifier(
   max_depth=12,min_samples_leaf=2,random_state=SEED)),
 "K-Nearest Neighbors": ("knn", Pipeline([
   ("standardize",StandardScaler()),
   ("classifier",KNeighborsClassifier(n_neighbors=5,weights="distance"))])),
 "Gaussian Naive Bayes": ("gaussian_naive_bayes",GaussianNB(var_smoothing=1e-8)),
 "Support Vector Machine": ("support_vector_machine",Pipeline([
   ("standardize",StandardScaler()),
   ("classifier",SVC(C=8,kernel="rbf",gamma="scale",probability=True,random_state=SEED))]))
}

def main():
    (ROOT/"model").mkdir(exist_ok=True)
    (ROOT/"assets").mkdir(exist_ok=True)
    raw = load_digits()
    x = pd.DataFrame(raw.data,columns=FEATURES)
    y = pd.Series(raw.target.astype(int),name=TARGET)
    xtrain,xtest,ytrain,ytest = train_test_split(
        x,y,test_size=0.2,stratify=y,random_state=SEED)
    full=x.assign(**{TARGET:y})
    test=xtest.assign(**{TARGET:ytest}).reset_index(drop=True)
    full.to_csv(ROOT/"full_dataset.csv",index=False)
    test.to_csv(ROOT/"test_data.csv",index=False)
    index,results={},[]
    for label,(stem,model) in MODELS.items():
        model.fit(xtrain,ytrain)
        joblib.dump(model,ROOT/"model"/(stem+".joblib"),compress=3)
        scores=evaluate(model,test)
        results.append({"Model":label,**scores})
        index[label]="model/"+stem+".joblib"
        print(label,{k:round(v,4) if v is not None else None for k,v in scores.items()})
        disp=ConfusionMatrixDisplay.from_predictions(
            test[TARGET],model.predict(test[FEATURES]),labels=list(range(10)),
            colorbar=False,cmap="Blues",values_format="d")
        disp.ax_.set_title(label)
        disp.figure_.set_size_inches(6,5)
        disp.figure_.tight_layout()
        disp.figure_.savefig(ROOT/"assets"/("cm_"+stem+".png"),dpi=130)
        plt.close(disp.figure_)
    table=pd.DataFrame(results)
    table.to_csv(ROOT/"metrics.csv",index=False,float_format="%.8f")
    winner=table.sort_values(["F1","MCC","AUC"],ascending=False).iloc[0]["Model"]
    metadata={
        "dataset_name":"UCI Optical Recognition of Handwritten Digits (scikit-learn subset)",
        "dataset_source":"https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits",
        "dataset_loader":"sklearn.datasets.load_digits",
        "total_instances":len(full),"feature_count":64,
        "classes":list(range(10)),"train_instances":len(xtrain),
        "test_instances":len(test),"random_state":SEED,
        "test_fraction":0.2,"features":FEATURES,"target_column":TARGET,
        "test_data_sha256":hashlib.sha256((ROOT/"test_data.csv").read_bytes()).hexdigest(),
        "model_index":index,"winner_by_weighted_f1":winner,
        "scikit_learn_version":sklearn.__version__
    }
    (ROOT/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf8")
    print("Winner:",winner)

if __name__ == "__main__":
    main()
