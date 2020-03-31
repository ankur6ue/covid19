import os
import pandas as pd
import json
import dask
import dask
import time
import itertools
from dask.distributed import Client
from gensim.summarization.textcleaner import get_sentences
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_tags, strip_punctuation, \
    strip_multiple_whitespaces, strip_numeric, \
    remove_stopwords, strip_short, strip_non_alphanum
from collections import OrderedDict
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import csv
from itertools import islice
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

# to see size on disk of directories
# du -shc ./*
# ls -l ./biorxiv_medrxiv/ | egrep -c '^-'

paths_to_json = ['/home/ankur/dev/apps/ML/Covid 19/comm_use_subset/comm_use_subset',
                 '/home/ankur/dev/apps/ML/Covid 19/noncomm_use_subset/noncomm_use_subset',
                 '/home/ankur/dev/apps/ML/Covid 19/biorxiv_medrxiv',
                 '/home/ankur/dev/apps/ML/Covid 19/custom_license/custom_license']

json_files = []
for path_to_json in paths_to_json:
    for file in os.listdir(path_to_json):
        if file.endswith('.json'):
            json_files.append(os.path.join(path_to_json, file))

CUSTOM_FILTERS = [
    lambda x: x.lower(), strip_tags, strip_punctuation,
    strip_multiple_whitespaces, strip_numeric, strip_non_alphanum,
    remove_stopwords, strip_short
]


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return islice(iterable, n)


def grouper(n, it):
    "grouper(3, 'ABCDEFG') --> ABC DEF G"
    it = iter(it)
    return iter(lambda: list(itertools.islice(it, n)), [])


class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        file_ = open("word2vec_loss.txt", "a+")
        loss = model.get_latest_training_loss()
        print("loss: {0}".format(loss))
        file_.write("%.3f" % loss)
        file_.write("\n")
        file_.close()
        self.epoch += 1


def get_data(articles):
    # document id -> abstract
    id2abstract = {}
    # document id -> title
    id2title = {}
    # document id -> authors
    id2authors = {}
    # list of pre-processed sentences
    sentences = []
    for article in articles:
        id = article['paper_id']
        abstract = article['abstract']
        title = article['metadata']['title']
        authors = article['metadata']['authors']
        bodytext = ''
        # convert: "check out these cool references [1][2]" ->
        # "check out these cool references
        for item in article['body_text']:
            cite_spans = item['cite_spans']
            ref_spans = item['ref_spans']
            text = item['text']
            for cite_span in cite_spans:
                strip_text = cite_span['text']
                # can't use string.strip, because it only strips characters at the beginning and end of a string!
                text = text.replace(strip_text, '')
            for ref_span in ref_spans:
                strip_text = ref_span['text']
                # can't use string.strip, because it only strips characters at the beginning and end of a string!
                text = text.replace(strip_text, '')
            bodytext = bodytext + text
        # preprocess - apply custom filters listed above
        for sentence in get_sentences(bodytext):
            sentences.append(preprocess_string(sentence, CUSTOM_FILTERS))
        id2abstract.update({id: abstract})
        id2title.update({id: title})
        id2authors.update({id: authors})
    return [id2abstract, id2title, id2authors, sentences]


# Reads the contents of json files in files list
def read_json_file(files):
    contents = []
    for file in files:
        with open(file) as json_data:
            data = json.load(json_data)
            contents.append(data)
    return contents


def preprocess_data(client):
    articles = []
    # with dask
    scheduler_info = client.scheduler_info()
    workers = scheduler_info['workers']
    num_workers = len(workers)
    num_articles = 1000
    block_size = int(num_articles / num_workers)
    dask.config.set({'distributed.worker.memory.spill': False})

    # without dask
    articles = []
    num_files = 0
    start = time.time()
    for file in json_files[0:100]:
        articles.append(*read_json_file([file]))
        print('reading file: {0}'.format(num_files))
        num_files = num_files + 1

    print('load time (without dask) to load {0} articles: {1}'.format(len(json_files), time.time() - start))

    # in case we want to only process a subset of articles
    num_articles = len(articles)
    articles_ = articles[0:num_articles]
    # conclusion: even with grouping, reading data from the disk in "parallel" using DASK takes longer than serial read.

    id2abstract = {}
    id2title = {}
    id2authors = {}
    sentences = []
    start = time.time()
    num = 0
    for article in articles_:
        id2abstract_, id2title_, id2authors_, sentences_ = get_data([article])
        id2abstract.update(id2abstract_)
        id2title.update(id2title_)
        id2authors.update(id2authors_)
        print('processing article #{0}'.format(num))
        num = num + 1
        for item in sentences_:
            sentences.append(item)
    print('load time without dask: {0}'.format(time.time() - start))

    # write_to_file('titles.csv', id2title, '\t')
    # write_to_file('abstracts.csv', id2abstract, '\t')
    # write_to_file('sentences.csv', sentences, '\t')
    return sentences


# data must be list of dictionaries
def write_to_file(filename, data, delimiter='\t'):
    with open(filename, "w") as f:
        wr = csv.writer(f, delimiter=delimiter)
        if type(data) is dict:
            for k, v in data.items():
                wr.writerow([k, v])
        if type(data) is list:
            wr.writerows(data)
        # otherwise, unsupported data structure


# for reading sentences. Returns list of words in a sentence
def read_from_file(filename, delimiter='\t'):
    with open(filename, newline='') as f:
        reader = csv.reader(f, delimiter=delimiter)
        data = list(reader)
        return data


# for reading titles and abstracts, returns dictionary
def read_from_file2(filename, delimiter='\t'):
    with open(filename, mode='r') as infile:
        reader = csv.reader(infile, delimiter=delimiter)
        mydict = {rows[0]: rows[1] for rows in reader}
    return mydict


def display_closestwords_tsnescatterplot(model, word, dim=128, num_closest=50):

    arr = np.empty((0, dim), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.similar_by_word(word, num_closest)

    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model.wv[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)

    # find tsne coords for 3 dimensions
    tsne = TSNE(n_components=3, random_state=0)
    np.set_printoptions(suppress=True)
    T = tsne.fit_transform(arr)

    x_coords = T[:, 0]
    y_coords = T[:, 1]
    z_coords = T[:, 2]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # display scatter plot

    sc = ax.scatter(x_coords, y_coords, z_coords)

    for label, x, y, z in zip(word_labels, x_coords, y_coords, z_coords):
        ax.text(x, y, z, label)
        if label is word:
            ax.text(x, y, z, label, color='red')
        # plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min() + 50, x_coords.max() + 50)
    plt.ylim(y_coords.min() + 50, y_coords.max() + 50)
    plt.ylim(z_coords.min() + 50, z_coords.max() + 50)
    plt.show()


# creates a word2vec model using gensim using combination of parameters
def create_model(client):
    iter = 20
    sg = 0
    dim = 128
    sentences = preprocess_data(client)
    sentences = read_from_file('sentences.csv')
    epoch_logger = EpochLogger()
    model = Word2Vec(sentences=sentences, workers=8, size=dim, compute_loss=True, iter=iter, sg=sg,
                     callbacks=[epoch_logger])
    model.save('covid19_w2v_gensim_iter={0}_sg={1}_dim={2}.model'.format(iter, sg, dim))


def compute_wmd(model, target, dict_):
    scores = []
    for k, v in dict_.items():
        if v:
            title_processed = preprocess_string(v, CUSTOM_FILTERS)
            score = model.wmdistance(target, title_processed)
            scores.append({k: score})
    return scores


def aux(args):
    return compute_wmd(*args)


def chunks(data, SIZE=10000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k: data[k] for k in islice(it, SIZE)}


def use_model(client):
    titles = read_from_file2('titles.csv')
    # load models
    iter = 20
    sg = 1
    dim = 128

    model = Word2Vec.load('covid19_w2v_gensim_iter={0}_sg={1}_dim={2}.model'.format(iter, sg, dim))
    target_title = "antibiotics Combating Antimicrobial Resistance in Singapore: A Qualitative Study Exploring the Policy Context, Challenges, Facilitators, and Proposed Strategies"

    target_title_processed = preprocess_string(target_title, CUSTOM_FILTERS)
    n = 3000
    # titles = dict(take(n, titles.items()))
    chunk_size = int(len(titles) / 10)
    scores = []
    start = time.time()
    args = []
    start = time.time()

    for item in chunks(titles, chunk_size):
        args.append([model, target_title_processed, item])

    with Pool(10) as p:
        scores_ = p.map(aux, args)

    scores = [items for item in scores_ for items in item]
    print('wmd compute time with multiprocessing: {0}'.format(time.time() - start))

    # compare with regular calculation (without multiprocessing)
    '''
    start = time.time()
    scores__ = []
    for item in chunks(titles, 200):
        _scores_ = compute_wmd(model, target_title_processed, item)
        scores__.append(_scores_)
    scores__ = [items for item in scores_ for items in item]
    print('wmd compute time without multiprocessing: {0}'.format(time.time() - start))
    pairs = zip(scores, scores__)
    print(any(x != y for x, y in pairs))
    '''
    # convert to dict
    scores = {k: v for r in scores for k, v in r.items()}
    # sort wmd scores in normal and reverse order
    sorted_scores = OrderedDict(sorted(scores.items(), key=lambda t: t[1]))
    reverse_sorted_scores = OrderedDict(sorted(scores.items(), key=lambda t: -t[1]))
    # top 10 matching titles
    num = 0
    for key in sorted_scores:
        print(titles[key])
        num = num + 1
        if num > 10:
            break
    print('***')
    # bottom 200 titles
    num = 0
    for key in reverse_sorted_scores:
        print(titles[key])
        num = num + 1
        if num > 200:
            break

    print(model.most_similar(positive=['covid'], topn=20))


def main(client):
    # create_model(client)
    use_model(client)
    print('good')


if __name__ == "__main__":
    client = Client(threads_per_worker=1, n_workers=10)
    main(client)

    # this code attempts to use dask to parallelize data preprocessing
    '''
    id2abstract = []
    id2title = []
    id2authors = []
    sentences = []
    processed_articles = []
    start = time.time()
    block_size = int(block_size)
    articles_ = articles[0:num_articles]
    groups = list(grouper(block_size, articles_))
    for group in groups:
        group = dask.delayed(group, traverse=False)
        lazy_result = dask.delayed(get_data)(group)
        lazy_result = dask.persist(lazy_result, traverse=False)
        processed_articles.append(lazy_result)

    dask.visualize(processed_articles, filename='dask_preprocessing.svg')
    start = time.time()
    # this persist appears to be needed, not clear why..
    processed_articles = dask.persist(processed_articles, traverse=False)
    processed_articles = dask.compute(processed_articles)
    # now concatenate
    processed_articles_ = processed_articles[0]
    for items in processed_articles_:
        for item_ in items[0]:
            id2abstract.append(item_)
        for item_ in items[1]:
            id2title.append(item_)
        for item_ in items[2]:
            id2authors.append(item_)
        for item_ in items[3]:
            sentences.append(item_)
    print('load time with dask: {0}'.format(time.time() - start))

    # get workers and scatter data to workers
    scheduler_info = client.scheduler_info()
    workers = scheduler_info['workers']
    articles_ = articles[0:1200]
    for k, v in workers.items():
        # get addr and port. Very stupid to have to do this..
        # for tcp://127.0.0.1:23456, port=23456
        pieces = k.split(':')
        host = v['host']
        port = pieces[2]
        client.scatter(articles_, [(host, port)], broadcast=False)
    '''
