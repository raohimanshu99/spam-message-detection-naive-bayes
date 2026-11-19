# Spam Message Detection Using Naive Bayes
BCSE306L Artificial Intelligence - Himanshu Yadav (24BDS0225), Samarth Sahu (24BDS0126)

## Run
    pip install scikit-learn pandas matplotlib joblib
    python3 spam_classifier.py              # trains, evaluates, writes results.json + figures + spam_model.joblib
    python3 spam_classifier.py --predict    # type a message, get SPAM/HAM + probability

## Files
- spam_classifier.py : whole pipeline (5 modules)
- data/sms.tsv       : SMS Spam Collection (label <TAB> message), 5,572 rows
- results.json       : every number quoted in the report
- fig_*.png          : figures used in the report

## Dataset credit
SMS Spam Collection: Almeida, T. A., Gomez Hidalgo, J. M., Yamakami, A. (2011). Contributions to the study of SMS spam filtering: new collection and results. ACM DocEng '11, 259-262.
