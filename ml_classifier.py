import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, roc_auc_score, recall_score, accuracy_score, make_scorer
import numpy as np
import re
import warnings

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


def stem_words(text):
    words = text.split()
    stemmer = SnowballStemmer(language='english')
    stemmed_text = ''
    for word in words:
        x = stemmer.stem(word)
        stemmed_text += x + ' '
    return stemmed_text.rstrip()

def remove_numbers(text):
    return ''.join([i for i in text if not text.isdigit()])

df = pd.read_csv('HIMYM_data.csv')
df = df.drop('Unnamed: 0',axis=1)

# clean text
df['text'] = df['text'].str.lower()
df['text'] = df['text'].apply(lambda x: remove_numbers(x))
df['text'] = df['text'].apply(lambda x: ''.join([i for i in x if not i.isdigit()]))
df['text'] = df['text'].apply(lambda x: re.sub('[^a-zA-Z0-9 \n\.]', '', x))
df['text'] = df['text'].apply(lambda x: re.sub(' +', ' ', x))
df['text'] = df['text'].apply(lambda x: x.replace('.',''))
df['text'] = df['text'].apply(lambda x: x.lstrip())
df['text'] = df['text'].apply(lambda x: stem_words(x))

df = df[df['character'].isin(['Barney','Marshall'])]
df['character'] = df['character'].replace({'Barney': 0, 'Marshall': 1})
characters = list(df.character.unique())
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['character'], test_size = 0.2,stratify = df['character'].values, random_state = 42)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, roc_auc_score, recall_score, accuracy_score, make_scorer
import numpy as np

char_level = [False]
lowercase = [False]
preprocessor = [None]
stop_words = [None]

ngram_min, ngram_max = 1, 3
ngrams_start_from_one, use_idf = ([True, False],) * 2

def log_range(lower, upper, samples_per_decade, is_int=True):
    samples = 1 + (upper - lower) * samples_per_decade
    samples = int(round(samples))
    if is_int:
        return [int(round(i)) for i in np.logspace(lower, upper, samples)]
    else:
        return list(np.logspace(lower, upper, samples))

min_dfs_logrange = (0, 2, 1)

if type(use_idf) is bool:
    use_idf = [use_idf]
if type(ngrams_start_from_one) is bool:
    ngram_min_max_pairs = [(1 if ngrams_start_from_one else n, n) \
                           for n in range(ngram_min, ngram_max + 1)]
else:
    ngram_min_max_pairs = [(1, n) for n in range(ngram_min, ngram_max + 1)]
    # The next one starts from 2 because if it's one it's aleady
    # included in the previous line
    ngram_min_max_pairs += [(n, n) for n in range(max(2, ngram_min), \
                                                  ngram_max + 1)]

print(f"The ngram ranges are: {ngram_min_max_pairs}")

Tfidf_params = {'feature_extractor': [TfidfVectorizer()],
         'feature_extractor__ngram_range': ngram_min_max_pairs,
         'feature_extractor__use_idf': use_idf,
         'feature_extractor__norm': ['l2'],
         'feature_extractor__min_df': log_range(*min_dfs_logrange),
         'feature_extractor__preprocessor': preprocessor,
         'feature_extractor__stop_words': stop_words,
         'feature_extractor__lowercase': lowercase,
         'feature_extractor__analyzer': ['word']
               }

from sklearn.model_selection import StratifiedKFold
def Merge (dict1, dict2):
  d = {**dict1, **dict2}
  return d

splitter = StratifiedKFold(3, shuffle=True, random_state = 42)

from sklearn.linear_model import LogisticRegression

reg_coef_logrange = (0, 3, 10)
lr_params = {'model': [LogisticRegression()],
         'model__random_state': [42],
         'model__C': log_range(*reg_coef_logrange, is_int=False),
         'model__solver': ['lbfgs', 'liblinear'],
         'model__max_iter': [int(1e6)],
         'model__class_weight': ['balanced']
         }


from sklearn.model_selection import RandomizedSearchCV

full_lr_params = Merge(Tfidf_params, lr_params)

lr_pipeline = Pipeline([
    ('feature_extractor', TfidfVectorizer()),
    ('scaler', 'passthrough'),
    ('model', LogisticRegression())
])

lr_rs = RandomizedSearchCV(estimator = lr_pipeline, param_distributions = full_lr_params, n_iter = 100,
                           scoring = 'roc_auc', n_jobs = -1, cv = splitter, verbose = 2)

warnings.filterwarnings('ignore')
lr_rs.fit(X_train, y_train)

print(f'Best Score:\n{lr_rs.best_score_}\n')
print(f'Best Params:\n{lr_rs.best_params_}')



# vectorizer = TfidfVectorizer()
# vectors = vectorizer.fit_transform(X_train)
# x = pd.DataFrame(vectors.toarray(), columns=vectorizer.get_feature_names())


# NB_pipeline = Pipeline([
#                 ('tfidf', TfidfVectorizer(stop_words=stop_words)),
#                 ('clf', OneVsRestClassifier(MultinomialNB(
#                     fit_prior=True, class_prior=None))),])
#
#
# SVC_pipeline = Pipeline([
#                 ('tfidf', TfidfVectorizer(stop_words=stop_words)),
#                 ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),])
#
#
# LogReg_pipeline = Pipeline([
#                 ('tfidf', TfidfVectorizer(stop_words=stop_words)),
#                 ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),])
#
# for character in characters:
#     print('... Processing {}'.format(character))
#     # train the model using X_dtm & y
#     LogReg_pipeline.fit(X_train, y_train)
#     # compute the testing accuracy
#     prediction = LogReg_pipeline.predict(X_test)
#     print('Test accuracy is {}'.format(accuracy_score(y_test, prediction)))
#
# SVC_pipeline.fit(X_train, y_train)
# pred = SVC_pipeline.predict(X_test)
# print(accuracy_score(y_test, pred))


