# Intro to sci-kit learn (sklearn)
import os
import pandas as pd
import numpy as np
import pickle
from warnings import warn
import matplotlib.pyplot as plt

from urllib.request import urlretrieve
from itertools import chain, cycle
from datetime import datetime as dt
from collections import Counter
from multiprocessing import cpu_count

from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, \
    GridSearchCV, RandomizedSearchCV
from sklearn.metrics import precision_score, roc_auc_score, recall_score, \
    accuracy_score, make_scorer
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, \
    roc_curve, auc

force_download = False
base_folder = r'Malware analysis example'
data_url = r'https://github.com/imperva/mal2vec/raw/master/examples/data/malware.gz'
data_path = r'data sets\imperva malware syscalls dataset.gz'


# %% Getting the data
def get_duration(start, milestone_name=''):
    elapsed = (dt.now() - start).total_seconds()
    if milestone_name:
        print(f"Time elapsed until {milestone_name}: {elapsed // (24 * 3600)} Days, "
              f"{(elapsed // 3600) % 24} Hours, {(elapsed // 60) % 60} Min., "
              f"{elapsed % 60:.2f} Sec.")
    return elapsed


if force_download or not os.path.isfile(data_path):
    urlretrieve(data_url, data_path)
df = pd.read_csv(data_path)
print(df.head(3))
print(df.tail(3))

# Checking how many files, labels and API calls are used
# Sorting everything so ensure consistancy used when I iteratate over these
malware_names = sorted(set(df['filename']))
class_names = sorted(set(df['label']))
syscalls = sorted(set(df['system_call']))
num_malware = len(malware_names)

# %% Checking the data
# Checking if there are any missing syscalls in the data and if so which ones
missing_syscalls = (range(a + 1, b) for a, b in zip(syscalls[:-1], syscalls[1:]) if not a + 1 == b)
missing_syscalls = list(chain.from_iterable(missing_syscalls))
if len(missing_syscalls):
    print(f"Some syscalls are never used. "
          f"The list of syscalls go up to {syscalls[-1]} "
          f"but there are only {len(syscalls)} used.\n"
          f"The {len(missing_syscalls)} missing syscalls are: "
          f"{str(missing_syscalls)[1:-1]}")

# Checking if all the label is identical for all the columns of the same malware
# nunique counts the number of unique values
# The following is equivalent to: all(df.groupby('filename')['label'].apply(lambda x: x.nunique()) == 1)
# The following version is shown as another way to achieve the same outcome
assert all(df.groupby('filename').agg({'label': ['nunique']}) == 1)
# Checking that no syscalls are missing (skipped) within any given file, and
# if they're already sorted by timestamp
are_syscalls_sorted = all(df.groupby('filename')['timestamp'].apply(
    lambda x: np.all(np.diff(x) == 1)))
if not are_syscalls_sorted:
    assert all(df.groupby('filename')['timestamp'].apply( \
        lambda x: np.all(np.diff(x.sort_values()) == 1)))
print(f"The syscalls in every malware are {'' if are_syscalls_sorted else 'NOT '}"
      f"sorted by timestamp")

# %% Curating the data
# We've seen that not all the syscalls numbers are available, and therefore we'll
# have to renumber them to get them into an embedding layer.
# If we're gonna reorder then, let's add some interpretability into that.
# In the new representation, the syscalls will be ordered by their tfidf scores.
# For example, syscall 196 has the highest score, so it'll be syscall 0.
# Syscalls 39, 118, 184 and 235 have the lowest score (each appears once in total),
# so of 275 syscalls they'll be syscalls 271-274.
# This adds some interpretability to the sequences as the number of the syscall
# will be an approximation of how important it is.
start = dt.now()
num_syscall_apperances = df.groupby('system_call').size()
syscall_malware_df = df.groupby(['system_call', 'filename']).size()
pair_calc_time = get_duration(start)
num_syscalls_in_malware = df.groupby('filename').size()  # groupby sorts the index
syscall_frequency = (syscall_malware_df / num_syscalls_in_malware).groupby('system_call').sum()
tf_calc_time = get_duration(start)
num_malware_with_syscall = syscall_malware_df.groupby('system_call').size()
df_calc_time = get_duration(start)
# For tf, by default TfidfVectorizer uses num_apperances_in_the_document. Note that this isn't a frequency, simply a count
# For idf, by default TfidfVectorizer uses 1+np.log((1+num_documents) / (1+num_documents_with_term))
# For tfidf, by default TfidfVectorizer uses tf * idf and then normalizes each document with l2 norm
# This means that the normalization is such that the sum of squares of tfidf weights of all the words in the i-th document is 1
# This is also the reason that tf is a simple count. Using it as a frequency would lead to exactly the same result after normalization
# Here, I avoid normalization to keep the variables interpretable. Instead, I use the syscall frequencies
# The frequency of a syscall calculated as: for malware: (sum(syscall_apperances_in_malware) / num_syscalls_in_malware)
sycalls_tfidf = syscall_frequency \
                * (1 + np.log((1 + len(malware_names)) / (1 + num_malware_with_syscall)))
tfidf_df = pd.concat([syscall_frequency, num_malware_with_syscall], axis=1)
tfidf_df.columns = ('tf', 'df')
tfidf_df['idf'] = (1 + np.log((1 + len(malware_names)) / (1 + tfidf_df['df'])))
tfidf_df['tfidf'] = tfidf_df['tf'] * tfidf_df['idf']
tfidf_df.sort_values('tfidf', ascending=False, inplace=True)
assert tfidf_df.shape[0] == len(syscalls)
tfidf_calc_time = get_duration(start)
# To map from the current numbers to the new ones, we'll use a dict
syscall_order_mapper = {tfidf_df.index[i]: i for i in range(len(syscalls))}
tfidf_mapping_time = get_duration(start)

malware_labels = df.groupby('filename').first()['label']  # groupby sorts the index
get_syscalls_as_text = lambda x: ' '.join((str(syscall_order_mapper[el]) for el in x))
malware_syscalls = df.groupby('filename')['system_call'].apply(get_syscalls_as_text)
curated_df = pd.concat([num_syscalls_in_malware, malware_labels, malware_syscalls], axis=1)
curated_df.columns = 'num_syscalls', 'label', 'syscalls'
print(curated_df.head())


# %% Supervised ML - classification with LR
def get_stratified_split(x, y, test_size=0.2):
    n_folds = int(round(1 / test_size))
    skf = StratifiedKFold(n_splits=n_folds, random_state=0, shuffle=True)
    for tr_ind, test_ind in skf.split(x, y):
        break
    return tr_ind, test_ind


tr_ind, test_ind = get_stratified_split(np.zeros(len(malware_names)),
                                        curated_df['label'], test_size=0.2)

tr_sequences = curated_df['syscalls'].iloc[tr_ind]
test_sequences = curated_df['syscalls'].iloc[test_ind]
tr_labels = curated_df['label'].iloc[tr_ind]
test_labels = curated_df['label'].iloc[test_ind]
reg_coef = 1e-2
ngram_range = (1, 1)
use_idf = True

vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_df=1.0, min_df=1, \
                             norm='l2', use_idf=use_idf, smooth_idf=True)
tr_features = vectorizer.fit_transform(tr_sequences)
test_features = vectorizer.transform(test_sequences)
model = LogisticRegression(C=reg_coef, random_state=0, class_weight='balanced',
                           solver='lbfgs', max_iter=int(1e6), multi_class='auto')
_ = model.fit(tr_features, tr_labels)


def show_scores(model, test_features, test_labels, verbose=True):
    predicted_proba = model.predict_proba(test_features)
    test_predictions = model.classes_[np.argmax(predicted_proba, axis=1)]

    scores = {}
    scores['accuracy'] = accuracy_score(test_labels, test_predictions)
    scores['precision'] = precision_score(test_labels, test_predictions,
                                          average='weighted')
    scores['OVO AUC'] = roc_auc_score(test_labels, predicted_proba, \
                                      average='weighted', multi_class='ovo')
    scores['OVR AUC'] = roc_auc_score(test_labels, predicted_proba, \
                                      average='weighted', multi_class='ovr')
    scores['recall'] = recall_score(test_labels, test_predictions, average='weighted')
    if verbose: print(scores)
    return scores


scores = show_scores(model, test_features, test_labels)


# %% Putting LR in a function
# Now putting it into a single function
# Inputs are orderd by likelihood to be changed from the default
def train_lr_classifier(tr_sequences, test_sequences, tr_labels, ngram_range=(1, 3), \
                        reg_coef=1e-2, use_idf=True, class_weight='balanced', \
                        norm='l2', smooth_idf=True, max_df=1.0, min_df=1, solver='lbfgs', \
                        max_iter=int(1e6), multi_class='auto'):
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, use_idf=use_idf, norm=norm, \
                                 smooth_idf=smooth_idf, max_df=max_df, min_df=min_df)
    tr_features = vectorizer.fit_transform(tr_sequences)
    if test_sequences is not None:
        test_features = vectorizer.transform(test_sequences)
    else:
        test_features = None

    model = LogisticRegression(C=reg_coef, random_state=0, class_weight=class_weight, \
                               solver='lbfgs', max_iter=int(1e6), multi_class='auto')
    _ = model.fit(tr_features, tr_labels)
    return vectorizer, model, tr_features, test_features


vectorizer, model, tr_features, test_features = \
    train_lr_classifier(tr_sequences, test_sequences, tr_labels, ngram_range, \
                        reg_coef, use_idf)
scores = show_scores(model, test_features, test_labels)

# %% Plotting the confusion matrix
# First try, just computing the CF
# test_predictions = model.predict(test_features)
# cf = confusion_matrix(test_labels, test_predictions)
# print(cf)

titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(model, test_features, test_labels,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()

# %% Plotting a ROC curve
proba = model.predict_proba(test_features)
fpr = [None] * len(class_names)
tpr = [None] * len(class_names)
roc_auc = [None] * len(class_names)
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'navy', 'deeppink', \
                'limegreen', 'slategray', 'yellow'])
plt.figure()
line_width = 2
plt.plot([0, 1], [0, 1], color='navy', lw=line_width, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Malware analysis ROC curve')
for i, (label, color) in enumerate(zip(class_names, colors)):
    fpr[i], tpr[i], _ = roc_curve(test_labels == label, proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

    plt.plot(fpr[i], tpr[i], color=color,
             lw=line_width, label=f'ROC curve ({label} = {roc_auc[i]:.2f}')
    plt.legend(loc="lower right")
plt.show()

# %% To potentially for classic NLP (pipeline arguments)
char_level = [False]
lowercase = [True]
preprocessor = [None]
stop_words = [None]

# %% Using a Pipline
max_ngram_min, max_ngram_max = 1, 3
n_experiments, n_jobs, num_val_folds = int(1e1), -2, 5
ngrams_start_from_one, use_idf = ([True, False],) * 2
reg_coef_logrange = (0, 3, 10)
min_dfs_logrange = (0, 2, 1)
score_to_optimize = 'OVO_AUC'
verbose = 10
file_path = r'D:\studies\Search, Information Retrieval and Recommender Systems\Example 1\model.p'


def log_range(lower, upper, samples_per_decade, is_int=True):
    samples = 1 + (upper - lower) * samples_per_decade
    samples = int(round(samples))
    if is_int:
        return [int(round(i)) for i in np.logspace(lower, upper, samples)]
    else:
        return list(np.logspace(lower, upper, samples))


# Defining the scoring functions as regular finction so it'll be possible to
# pickle the GridSearchCV object
def get_ovo_roc_auc(gt, pred):
    return roc_auc_score(gt, pred, average='weighted', multi_class='ovo')


def get_ovr_roc_auc(gt, pred):
    return roc_auc_score(gt, pred, average='weighted', multi_class='ovr')


def get_precision(gt, pred):
    return precision_score(gt, pred, average='weighted')


def get_recall(gt, pred):
    return recall_score(gt, pred, average='weighted')


ovo_auc_scorer = make_scorer(get_ovo_roc_auc, needs_proba=True)
ovr_auc_scorer = make_scorer(get_ovr_roc_auc, needs_proba=True)
precision_scorer = make_scorer(get_precision)
recall_scorer = make_scorer(get_recall)
scorers = {'OVO_AUC': ovo_auc_scorer, 'OVR_AUC': ovr_auc_scorer, \
           'Precision': precision_scorer, 'Recall': recall_scorer, \
           'Accuracy': 'accuracy'}

print(f"Defining the potential parameters for the hyper-parameter search")
if type(use_idf) is bool:
    use_idf = [use_idf]
if type(ngrams_start_from_one) is bool:
    ngram_min_max_pairs = [(1 if ngrams_start_from_one else n, n) \
                           for n in range(max_ngram_min, max_ngram_max + 1)]
else:
    ngram_min_max_pairs = [(1, n) for n in range(max_ngram_min, max_ngram_max + 1)]
    # The next one starts from 2 because if it's one it's aleady
    # included in the previous line
    ngram_min_max_pairs += [(n, n) for n in range(max(2, max_ngram_min), \
                                                  max_ngram_max + 1)]
print(f"The ngram ranges are: {ngram_min_max_pairs}")

print('Defining the pipline', dt.now())
param = {'feature_extractor': [TfidfVectorizer()],
         'feature_extractor__ngram_range': ngram_min_max_pairs,
         'feature_extractor__use_idf': use_idf,
         'feature_extractor__norm': ['l2'],
         'feature_extractor__min_df': log_range(*min_dfs_logrange),
         'feature_extractor__preprocessor': preprocessor,
         'feature_extractor__stop_words': stop_words,
         'feature_extractor__lowercase': lowercase,
         'feature_extractor__analyzer': ['char' if cl else 'word' \
                                         for cl in char_level],
         'model': [LogisticRegression()],
         'model__random_state': [0],
         'model__C': log_range(*reg_coef_logrange, is_int=False),
         'model__solver': ['lbfgs'],
         'model__multi_class': ['auto'],
         'model__max_iter': [int(1e6)],
         'model__class_weight': ['balanced'],
         }

pipeline = Pipeline([
    ('feature_extractor', TfidfVectorizer()),
    ('scaler', 'passthrough'),
    ('model', LogisticRegression())
])
splitter = StratifiedKFold(num_val_folds)
if n_experiments:
    hyper_opt = RandomizedSearchCV(pipeline, param, n_experiments,
                                   scorers, n_jobs, cv=splitter,
                                   refit=score_to_optimize,
                                   return_train_score=False, verbose=verbose)
else:
    hyper_opt = GridSearchCV(pipeline, param, scorers, n_jobs, \
                             cv=splitter, refit=score_to_optimize, \
                             return_train_score=False, verbose=verbose)

# The fitting chunk starts here
# Checking the number of validation folds fits the data
tr_val_sequences, tr_val_labels = tr_sequences, tr_labels
num_label_apperances = tr_val_labels.value_counts()
min_label_apperances = min(num_label_apperances)
if num_val_folds > min_label_apperances:
    warn(f"The num_val_folds ({num_val_folds}) is greater than the "
         f"apperances of the rarest label ({num_label_apperances.idxmin()}: "
         f"{min_label_apperances}). Channging num_val_folds to "
         f"{min_label_apperances}")
    hyper_opt.cv = StratifiedKFold(min_label_apperances)

# Fitting the model
start_time = dt.now()
print(f"Fitting (training) the models and performing hyper parameter search. "
      f"Time: {start_time}")
hyper_opt.fit(tr_val_sequences, tr_val_labels)

print(f"Valuating algorithm performance. Time: {dt.now()}")
best_ind = hyper_opt.best_index_

scores = {}
for scorer in scorers:
    scores[scorer] = hyper_opt.cv_results_['mean_test_' + scorer][best_ind]
scores_str = '\n'.join(f"{k}: {v:.3f}" for k, v in scores.items())
print(f'---Algorithm performance---: \n{scores_str}\n-------------')
end_time = dt.now()
duration = (end_time - start_time).total_seconds()
print(f'Training took {duration} seconds')
with open(file_path, 'wb') as f:
    pickle.dump(hyper_opt, f)

model = hyper_opt
test_features = test_sequences
# hyper_opt.predict_proba(test_sequences)