from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold

def text_process(dt_input):
    text = dt_input
    print("Original text is\n{}".format('\n'.join(text)))
    vectorizer = CountVectorizer(min_df=0)
    vectorizer.fit(text)
    x = vectorizer.transform(text)
    x = x.toarray()
    print(vectorizer.get_feature_names())

def cv_score(clf, X, y, scorefunc):
    result = 0.
    nfold = 5
    for train, test in KFold(nfold).split(X):
        clf.fit(X[train], y[train])
        result += scorefunc(clf, X[test], y[test])
    return result / nfold

def log_likelihood(clf, x, y):
    prob = clf.predict_log_proba(x)
    rotten = y == 0
    fresh = ~rotten
    return prob[rotten, 0].sum() + prob[fresh, 1].sum()