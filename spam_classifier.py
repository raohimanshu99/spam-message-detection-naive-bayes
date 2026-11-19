"""
Spam Message Detection Using the Naive Bayes Classifier
BCSE306L - Artificial Intelligence | Himanshu Yadav (24BDS0225), Samarth Sahu (24BDS0126)

Pipeline: load data -> preprocess -> TF-IDF -> train/test split -> Multinomial Naive Bayes
          -> evaluation (accuracy, precision, recall, F1, confusion matrix) -> baseline comparison
Run:  python3 spam_classifier.py            (trains, evaluates, saves results.json + figures)
      python3 spam_classifier.py --predict  (then type messages to classify)
"""
import re
import sys
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

DATA_PATH = "data/sms.tsv"
RANDOM_STATE = 42


# ---------------- Module 1: Data Preprocessing ----------------
def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/digits, drop stopwords."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [t for t in text.split() if t not in ENGLISH_STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


def clean_text_v2(text: str) -> str:
    """Same as clean_text, but long digit strings (5+ digits: phone numbers, short codes)
    are first replaced by the marker token 'longnum' instead of being discarded."""
    text = re.sub(r"\d{5,}", " longnum ", text.lower())
    return clean_text(text)


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "message"])
    df["y"] = (df["label"] == "spam").astype(int)
    df["clean"] = df["message"].apply(clean_text)
    df["clean2"] = df["message"].apply(clean_text_v2)
    return df


# ---------------- Modules 2-4: features, model, evaluation ----------------
def metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def main():
    df = load_data()
    print(f"Messages: {len(df)} | spam: {df.y.sum()} ({df.y.mean()*100:.1f}%) | ham: {(1-df.y).sum()}")

    X_train, X_test, y_train, y_test, raw_train, raw_test = train_test_split(
        df["clean"], df["y"], df["message"], test_size=0.2,
        random_state=RANDOM_STATE, stratify=df["y"])
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # Feature extraction (TF-IDF)
    vec = TfidfVectorizer()
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    print(f"Vocabulary size: {Xtr.shape[1]}")

    results = {}

    # Baseline: always predict majority class (ham)
    base = DummyClassifier(strategy="most_frequent").fit(Xtr, y_train)
    results["Baseline (always ham)"] = metrics(y_test, base.predict(Xte))

    # Main model: Multinomial Naive Bayes (alpha = 0.1 fixed in advance; sensitivity to alpha reported separately below)
    nb = MultinomialNB(alpha=0.1).fit(Xtr, y_train)
    pred_nb = nb.predict(Xte)
    results["Multinomial Naive Bayes (TF-IDF)"] = metrics(y_test, pred_nb)

    # Comparison: Naive Bayes with raw word counts (shows why TF-IDF was used)
    cv = CountVectorizer()
    Ctr, Cte = cv.fit_transform(X_train), cv.transform(X_test)
    nb_cnt = MultinomialNB(alpha=0.1).fit(Ctr, y_train)
    results["Multinomial Naive Bayes (Bag-of-Words)"] = metrics(y_test, nb_cnt.predict(Cte))

    # Comparison: Logistic Regression (a discriminative alternative)
    lr = LogisticRegression(max_iter=1000).fit(Xtr, y_train)
    results["Logistic Regression (TF-IDF)"] = metrics(y_test, lr.predict(Xte))

    print("\n=== Test-set results ===")
    for name, m in results.items():
        print(f"{name:42s} acc={m['accuracy']:.4f} prec={m['precision']:.4f} "
              f"rec={m['recall']:.4f} f1={m['f1']:.4f}")

    cm = confusion_matrix(y_test, pred_nb)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion matrix (NB+TF-IDF): TN={tn} FP={fp} FN={fn} TP={tp}")

    # Error analysis
    test_df = pd.DataFrame({"msg": raw_test.values, "y": y_test.values, "pred": pred_nb})
    fps = test_df[(test_df.y == 0) & (test_df.pred == 1)]["msg"].tolist()
    fns = test_df[(test_df.y == 1) & (test_df.pred == 0)]["msg"].tolist()

    # Most spam-indicative words (log-prob ratio)
    vocab = np.array(vec.get_feature_names_out())
    log_ratio = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
    top_spam = vocab[np.argsort(log_ratio)[::-1][:10]].tolist()
    top_ham = vocab[np.argsort(log_ratio)[:10]].tolist()
    print("Top spam words:", top_spam)
    print("Top ham words :", top_ham)

    # Alpha sensitivity (Laplace smoothing)
    alpha_res = {}
    for a in [0.01, 0.05, 0.1, 0.5, 1.0]:
        m = MultinomialNB(alpha=a).fit(Xtr, y_train)
        alpha_res[a] = metrics(y_test, m.predict(Xte))
    print("\nAlpha sensitivity:")
    for a, m in alpha_res.items():
        print(f"  alpha={a:<5} acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f}")


    # 5-fold stratified cross-validation (vectoriser re-fitted inside every fold, so no leakage)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_models = {
        "NB (TF-IDF)": make_pipeline(TfidfVectorizer(), MultinomialNB(alpha=0.1)),
        "NB (Bag-of-Words)": make_pipeline(CountVectorizer(), MultinomialNB(alpha=0.1)),
        "Logistic Regression (TF-IDF)": make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000)),
    }
    cv_res = {}
    print("\n5-fold CV (mean +/- std):")
    for name, mdl in cv_models.items():
        sc = cross_validate(mdl, df["clean"], df["y"], cv=skf,
                            scoring=["accuracy", "precision", "recall", "f1"])
        cv_res[name] = {k: [float(sc[f"test_{k}"].mean()), float(sc[f"test_{k}"].std())]
                        for k in ["accuracy", "precision", "recall", "f1"]}
        r = cv_res[name]
        print(f"  {name:30s} acc={r['accuracy'][0]:.4f}+/-{r['accuracy'][1]:.4f} "
              f"prec={r['precision'][0]:.4f}+/-{r['precision'][1]:.4f} "
              f"rec={r['recall'][0]:.4f}+/-{r['recall'][1]:.4f} "
              f"f1={r['f1'][0]:.4f}+/-{r['f1'][1]:.4f}")

    # How many missed spam messages contain a long digit string (phone / short-code)?
    has_num = lambda m: bool(re.search(r"\d{5,}", m))
    fn_num = sum(has_num(m) for m in fns)
    spam_num = sum(has_num(m) for m in df[df.y == 1]["message"])
    ham_num = sum(has_num(m) for m in df[df.y == 0]["message"])
    print(f"\nFNs with 5+ digit string: {fn_num}/{len(fns)} | spam overall: {spam_num}/{int(df.y.sum())} "
          f"| ham overall: {ham_num}/{int((1-df.y).sum())}")


    # Extension: keep a 'longnum' marker for phone numbers / short codes (motivated by the error analysis)
    X2tr = df.loc[X_train.index, "clean2"]; X2te = df.loc[X_test.index, "clean2"]
    vec2 = TfidfVectorizer(); X2tr_v = vec2.fit_transform(X2tr); X2te_v = vec2.transform(X2te)
    nb2 = MultinomialNB(alpha=0.1).fit(X2tr_v, y_train)
    pred2 = nb2.predict(X2te_v)
    results["NB (TF-IDF) + number marker"] = metrics(y_test, pred2)
    cm2 = confusion_matrix(y_test, pred2); tn2, fp2, fn2, tp2 = cm2.ravel()
    print(f"\nExtension (number marker): acc={results['NB (TF-IDF) + number marker']['accuracy']:.4f} "
          f"prec={results['NB (TF-IDF) + number marker']['precision']:.4f} "
          f"rec={results['NB (TF-IDF) + number marker']['recall']:.4f} "
          f"f1={results['NB (TF-IDF) + number marker']['f1']:.4f} | TN={tn2} FP={fp2} FN={fn2} TP={tp2}")
    sc2 = cross_validate(make_pipeline(TfidfVectorizer(), MultinomialNB(alpha=0.1)), df["clean2"], df["y"],
                         cv=skf, scoring=["accuracy", "precision", "recall", "f1"])
    cv_res["NB (TF-IDF) + number marker"] = {k: [float(sc2[f"test_{k}"].mean()), float(sc2[f"test_{k}"].std())]
                                              for k in ["accuracy", "precision", "recall", "f1"]}
    r2 = cv_res["NB (TF-IDF) + number marker"]
    print("5-fold CV (number marker): " + " ".join(f"{k}={r2[k][0]:.4f}+/-{r2[k][1]:.4f}" for k in r2))
    ext_conf = {"TN": int(tn2), "FP": int(fp2), "FN": int(fn2), "TP": int(tp2)}

    # Save outputs
    out = {
        "n_total": int(len(df)), "n_spam": int(df.y.sum()), "n_ham": int((1 - df.y).sum()),
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "vocab_size": int(Xtr.shape[1]),
        "results": results,
        "confusion": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "false_positives": fps, "false_negatives": fns,
        "top_spam_words": top_spam, "top_ham_words": top_ham,
        "alpha_sensitivity": {str(k): v for k, v in alpha_res.items()},
        "cv_5fold": cv_res,
        "confusion_extension": ext_conf,
        "digit_analysis": {"fn_with_digits": int(fn_num), "n_fn": len(fns), "spam_with_digits": int(spam_num), "ham_with_digits": int(ham_num)},
    }
    with open("results.json", "w") as f:
        json.dump(out, f, indent=2)

    joblib.dump((vec, nb), "spam_model.joblib")

    # Figure: confusion matrix
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Ham", "Spam"]); ax.set_yticklabels(["Ham", "Spam"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    plt.tight_layout()
    plt.savefig("fig_confusion.png", dpi=250)
    plt.close()

    # Figure: model comparison bar chart (F1)
    names = ["Baseline", "NB\n(BoW)", "NB\n(TF-IDF)", "LogReg", "NB+num\nmarker"]
    keys = ["Baseline (always ham)", "Multinomial Naive Bayes (Bag-of-Words)",
            "Multinomial Naive Bayes (TF-IDF)", "Logistic Regression (TF-IDF)",
            "NB (TF-IDF) + number marker"]
    f1s = [results[k]["f1"] for k in keys]
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    bars = ax.bar(names, f1s, color=["#bbbbbb", "#7f9cd6", "#2c3e6b", "#c97b7b", "#3f8f5f"])
    ax.set_ylim(0, 1.05); ax.set_ylabel("F1-score (spam)")
    for b, v in zip(bars, f1s):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
    plt.xticks(fontsize=6.5)
    plt.tight_layout()
    plt.savefig("fig_compare.png", dpi=250)
    plt.close()
    print("\nSaved results.json, spam_model.joblib, fig_confusion.png, fig_compare.png")


# ---------------- Module 5: interactive testing ----------------
def predict_loop():
    vec, nb = joblib.load("spam_model.joblib")
    print("Type a message (blank line to quit).")
    while True:
        msg = input("> ").strip()
        if not msg:
            break
        x = vec.transform([clean_text(msg)])
        p = nb.predict_proba(x)[0][1]
        print(f"  -> {'SPAM' if p >= 0.5 else 'HAM'}  (spam probability {p:.3f})")


if __name__ == "__main__":
    if "--predict" in sys.argv:
        predict_loop()
    else:
        main()
