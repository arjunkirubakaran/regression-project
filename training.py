#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score

def split_data(X, y1, y2, seed=42):
    Xtr, Xtmp, y1tr, y1tmp, y2tr, y2tmp = train_test_split(X, y1, y2, test_size=0.3, random_state=seed)
    Xva, Xte, y1va, y1te, y2va, y2te = train_test_split(Xtmp, y1tmp, y2tmp, test_size=2/3, random_state=seed)
    return Xtr, Xva, Xte, y1tr, y1va, y1te, y2tr, y2va, y2te

def scores(y, yhat):
    err = y - yhat
    return {
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAE":  float(mean_absolute_error(y, yhat)),
        "R2":   float(r2_score(y, yhat))
    }

def tune_ridge(Xtr, ytr, Xva, yva, alphas, seed):
    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(random_state=seed))])
    best = None
    for a in alphas:
        pipe.set_params(ridge__alpha=float(a))
        m = pipe.fit(Xtr, ytr)
        r2 = r2_score(yva, m.predict(Xva))
        if not best or r2 > best["r2"]:
            best = {"alpha": float(a), "r2": r2, "model": m}
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ENB2012_data.xlsx")
    ap.add_argument("--out",  default="outputs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alphas", default="0,0.1,1,10,100")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(args.data)
    X = df.drop(columns=["Y1","Y2"]).astype(float)
    y1, y2 = df["Y1"].astype(float), df["Y2"].astype(float)
    Xtr, Xva, Xte, y1tr, y1va, y1te, y2tr, y2va, y2te = split_data(X, y1, y2, args.seed)

    alphas = [float(s) for s in args.alphas.split(",") if s.strip()]

    rows = {}
    for target, ytr, yva, yte in [("Y1", y1tr, y1va, y1te), ("Y2", y2tr, y2va, y2te)]:
        lin = LinearRegression().fit(Xtr, ytr)
        rows[(target,"Linear","Validation")] = scores(yva, lin.predict(Xva))
        lin_final = LinearRegression().fit(pd.concat([Xtr,Xva]), pd.concat([ytr,yva]))
        rows[(target,"Linear","Test")] = scores(yte, lin_final.predict(Xte))

        best = tune_ridge(Xtr, ytr, Xva, yva, alphas, args.seed)
        rows[(target,f"Ridge[a*={best['alpha']}]", "Validation")] = scores(yva, best["model"].predict(Xva))
        ridge_final = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=best["alpha"], random_state=args.seed))]).fit(pd.concat([Xtr,Xva]), pd.concat([ytr,yva]))
        rows[(target,f"Ridge[a*={best['alpha']}]", "Test")] = scores(yte, ridge_final.predict(Xte))

    metrics = pd.DataFrame(rows).T
    metrics.index = pd.MultiIndex.from_tuples(metrics.index, names=["Target","Model","Split"])
    (out/"metrics.csv").write_text(metrics.to_csv())
    (out/"metrics.json").write_text(json.dumps(metrics.reset_index().to_dict(orient="records"), indent=2))

    print("\nResults saved to outputs/")
    print(metrics.round(4).to_string())

if __name__ == "__main__":
    main()
