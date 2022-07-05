import pandas as pd
import re


class ProjectTools:

    def __init__(self):
        self.df = pd.DataFrame()

    @staticmethod
    def _remove_director_notes(text):
        return re.sub(r"[\(\[].*?[\)\]]", "", text)   # remove everything inside parentheses

    def clean_data(self, url):
        self.df = pd.read_csv(url)    # get data
        self.df = self.df.iloc[:, 1:]   # remove redundant column

        # remove leftover html stuff
        self.df['text'] = self.df['text'].apply(lambda x: x.split('<')[0])
        self.df['text'] = self.df['text'].apply(lambda x: x.replace('&quot;', '').replace('"', ''))
        self.df['text'] = self.df.text.map(lambda x: self._remove_director_notes(x))

        return self.df

    @staticmethod
    def get_data_of_character(df, character):
        if character not in ['Marshall', 'Ted', 'Barney', 'Lily', 'Robin']:
            raise ValueError('Character must be a main character in HIMYM')

        try:
            return df.loc[df['character'] == character]['text'].reset_index(drop=True)
        except KeyError as e:
            print(f'Column {e} missing when calling get_data_of_character')