import requests
import os
import pandas as pd


def main():
    path_of_infiles = 'C://DL final project//episode_links_by_season'
    main_characters = ['Marshall', 'Ted', 'Barney', 'Lily', 'Robin']

    character_data = []
    script_data = []

    for filename in os.listdir(path_of_infiles):
        print(f'Currently working on {filename}')
        with open(os.path.join(path_of_infiles, filename), 'r', encoding='utf-8') as f:
            links = f.readlines()
            for link in links:
                doc = requests.get(link).text
                doc = ''.join(doc.split('\n'))

                for line in doc.split('<p>'):
                    line = line[:-4].replace('<strong>', '').replace('</strong>', '')
                    if line.split(':')[0] in main_characters:
                        # split only on first occurance. this will seperate character name
                        data_lst = line.split(':', 1)
                        character_data.append(data_lst[0])
                        script_data.append(data_lst[1].strip())

    outfile = pd.DataFrame({'text': script_data, 'character': character_data})
    outfile.to_csv('HIMYM_data.csv')


if __name__ == "__main__":
    main()


