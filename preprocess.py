import os
from tqdm import tqdm
import json
import pickle as pkl
from pprint import pprint
from pymagnitude import Magnitude, MagnitudeUtils

fasttext_dim = 300
glove_dim = 300

base_dir = os.getcwd()

test_data = []
train_data = []
dev_data = []

# print('Downloading Glove...')
# glove = Magnitude(MagnitudeUtils.download_model('glove/medium/glove.6B.{}d'.format(glove_dim), download_dir = os.path.join(base_dir, 'magnitude')), case_insensitive=True)

# print('Downloading FastText...')
# fasttext = Magnitude(MagnitudeUtils.download_model('fasttext/medium/wiki-news-300d-1M-subword', download_dir = os.path.join(base_dir, 'magnitude')), case_insensitive=True)
glove = Magnitude('./magnitude/glove_medium_glove.6B.300d.magnitude')
fasttext = Magnitude('./magnitude/fasttext_medium_wiki-news-300d-1M-subword.magnitude')

vectors = Magnitude(glove, fasttext)

with open('./data/env.json') as f:
    data = json.load(f)

print('Preparing Train data...')
for i in tqdm(range(len(data['train'])), ascii=True, ncols=100):
    train_data.append([data['train'][i][0], data['train'][i][1], data['train'][i][4], data['train'][i][5]])

print('Preparing Test data...')
for i in tqdm(range(len(data['test'])), ascii=True, ncols=100):
    test_data.append([data['test'][i][0], data['test'][i][1], data['test'][i][4], data['test'][i][5]])

print('Preparing Dev data...')
for i in tqdm(range(len(data['dev'])), ascii=True, ncols=100):
    dev_data.append([data['dev'][i][0], data['dev'][i][1], data['dev'][i][4], data['dev'][i][5]])


print('Writing to file')

with open('./data/train_data.pkl', 'wb') as f:
    pkl.dump(train_data, f)

with open('./data/test_data.pkl', 'wb') as f:
    pkl.dump(test_data, f)

with open('./data/dev_data.pkl', 'wb') as f:
    pkl.dump(dev_data, f)
