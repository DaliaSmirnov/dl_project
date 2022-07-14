import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
import tensorflow as tf
import numpy as np



try:
    df = pd.read_csv('HIMYM_data.csv').drop(['Unnamed: 0'], axis=1)
except Exception:
    df = pd.read_csv('HIMYM_data.csv')

character_to_int = {'Marshall': 0, 'Ted': 1, 'Barney': 2, 'Lily': 3, 'Robin': 4}
df['character'] = df['character'].replace(character_to_int)


def normalize_data(df):
    def remove_non_dialogue_text(text):
      return re.sub("[\(\[].*?[\)\]]", "", text)

    def remove_unexpected_chars(text):
      unexpected_re = re.compile(r"[^A-Za-z0-9^,!.\/'+-=]>")
      return unexpected_re.sub('', text)

    def english_shortcut_corrections(text):
        text = text.replace(r"?", "")
        text = text.replace(r"'s", " is")
        text = text.replace(r"ive", "I have")
        text = text.replace(r"didnt", "did not")
        text = text.replace(r"heyy", "hey")
        text = text.replace(r"'ve", " have")
        text = text.replace(r"can't", "cannot")
        text = text.replace(r"cant", "cannot")
        text = text.replace(r"musn't", "must not")
        text = text.replace(r"you're", "you are")
        text = text.replace(r"youre", "you are")
        text = text.replace(r"they're", "they are")
        text = text.replace(r"theyre", "they are")
        text = text.replace(r"yup", "yes")
        text = text.replace(r"n't", " not")
        text = text.replace(r"i'm", "I am")
        text = text.replace(r"im", "I am")
        text = text.replace(r"IM", "I am")
        text = text.replace(r"Im", "I am")
        text = text.replace(r"I'm", "I am")
        text = text.replace(r"'re", " are")
        text = text.replace(r"'d", " would")
        text = text.replace(r"\'ll", " will")
        text = text.replace(r"I'm", "I am")
        text = text.replace(r"\em", "them")

        return text

    # relevant punctuations: whitespace, exclamation-mark and question-mark
    def remove_irrelevant_punctuations(text):
        return re.sub(pattern='[^\w\s\!\?]', repl='', string=text)

    # almost last to limit noise in the text so the function will successfully stem most words
    def stem_words(text):
        words = text.split()
        stemmer = SnowballStemmer(language='english')
        stemmed_text = ''
        for word in words:
            x = stemmer.stem(word)
            stemmed_text += x + ' '
        return stemmed_text.rstrip()

    # saved for last because the text has to be as clean as possible for this function to catch the most stopwords
    def remove_stopwords(text):
        stop_words = set(stopwords.words('english'))
        return ' '.join(word for word in text.split() if word.lower() not in stop_words)

    # apply all functions (order is important) and after strip leftover whitspaces
    df['text'] = df.text.map(lambda x: x.lower())
    df['text'] = df.text.map(lambda x: remove_non_dialogue_text(x))
    df['text'] = df.text.map(lambda x: remove_unexpected_chars(x))
    df['text'] = df.text.map(lambda x: english_shortcut_corrections(x))
    df['text'] = df.text.map(lambda x: remove_irrelevant_punctuations(x))
    df['text'] = df.text.map(lambda x: stem_words(x))
    df['text'] = df.text.map(lambda x: remove_stopwords(x))
    df['text'] = df.text.map(lambda x: x.strip())

    return df


df = normalize_data(df)

X = df['text']
y = df['character']

y = np.utils(y, num_classes=5)

