from __future__ import print_function

import logging
import sys
import pickle
import errno
import os
import numpy as np
import csv


DEBD = [
    "accidents",
    "ad",
    "baudio",
    "bbc",
    "bnetflix",
    "book",
    "c20ng",
    "cr52",
    "cwebkb",
    "dna",
    "jester",
    "kdd",
    "kosarek",
    "moviereview",
    "msnbc",
    "msweb",
    "nltcs",
    "plants",
    "pumsb_star",
    "tmovie",
    "tretail",
    "voting",
]

DEBD_num_vars = {
    "accidents": 111,
    "ad": 1556,
    "baudio": 100,
    "bbc": 1058,
    "bnetflix": 100,
    "book": 500,
    "c20ng": 910,
    "cr52": 889,
    "cwebkb": 839,
    "dna": 180,
    "jester": 100,
    "kdd": 64,
    "kosarek": 190,
    "moviereview": 1001,
    "msnbc": 17,
    "msweb": 294,
    "nltcs": 16,
    "plants": 69,
    "pumsb_star": 163,
    "tmovie": 500,
    "tretail": 135,
    "voting": 1359,
}


def mkdir_p(path):
    """Linux mkdir -p"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def load_mnist(data_dir):
    """Load MNIST"""

    fd = open(os.path.join(data_dir, "train-images-idx3-ubyte"))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_x = loaded[16:].reshape((60000, 784)).astype(np.float32)

    fd = open(os.path.join(data_dir, "train-labels-idx1-ubyte"))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_labels = loaded[8:].reshape((60000)).astype(np.float32)

    fd = open(os.path.join(data_dir, "t10k-images-idx3-ubyte"))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_x = loaded[16:].reshape((10000, 784)).astype(np.float32)

    fd = open(os.path.join(data_dir, "t10k-labels-idx1-ubyte"))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_labels = loaded[8:].reshape((10000)).astype(np.float32)

    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)

    return train_x, train_labels, test_x, test_labels


def load_debd(data_dir, name, dtype="int32"):
    """Load one of the twenty binary density esimtation benchmark datasets."""

    train_path = os.path.join(data_dir, "datasets", name, name + ".train.data")
    test_path = os.path.join(data_dir, "datasets", name, name + ".test.data")
    valid_path = os.path.join(data_dir, "datasets", name, name + ".valid.data")

    reader = csv.reader(open(train_path, "r"), delimiter=",")
    train_x = np.array(list(reader)).astype(dtype)

    reader = csv.reader(open(test_path, "r"), delimiter=",")
    test_x = np.array(list(reader)).astype(dtype)

    reader = csv.reader(open(valid_path, "r"), delimiter=",")
    valid_x = np.array(list(reader)).astype(dtype)

    return train_x, test_x, valid_x


# print('DEBD_num_vars = {')
# for dataset in DEBD:
#     train_x, test_x, valid_x = load_debd('/homes/mlghomes/datasets/DEBD/', dataset, dtype='int32')
#     print("        '{}': {},".format( dataset, train_x.shape[1]))
# print('        }')


def preprocess_text(
    data,
    tokenizer=None,
    stopwords=None,
    filter_numbers=True,
    # stemmer='snowball',
    stemmer=None,
    filter_pos_tag=False,
    lemmatizer="wordnet",
    # lemmatizer=None,
    wordnet_filtering=False,
    bigrams=False,
    trigrams=False,
    min_df=1,
    max_df=0.3,
):
    """
    Small NLP pipeline:
    0. lower casing
    0. tokenize
    1. stopword removal
    2. lemmatize/stemming
    3. filtering by document frequency
    4. pos tagging filtering
    Reduces documents to lists of adjectives, nouns, and verbs.
    """

    import nltk
    from nltk import word_tokenize
    from nltk.tokenize import RegexpTokenizer
    from nltk.stem import WordNetLemmatizer
    from nltk.stem import LancasterStemmer
    from nltk.stem import PorterStemmer
    from nltk.stem import SnowballStemmer
    from nltk.tokenize.texttiling import TextTilingTokenizer
    from nltk.tokenize import sent_tokenize
    from nltk.corpus import stopwords as stopws
    from nltk.util import ngrams
    from nltk.corpus import wordnet

    from collections import Counter

    if stopwords is None:
        stopwords = set(stopws.words("english"))

    n_docs = len(data)

    if isinstance(min_df, float):
        min_df = min_df * n_docs
    if isinstance(max_df, float):
        max_df = max_df * n_docs
    logging.info("Min doc freq {} and Max doc freq {} (all docs {})".format(min_df, max_df, n_docs))

    def get_stemmer(stemmer="snowball"):

        if stemmer == "porter":
            return PorterStemmer()
        elif stemmer == "lancaster":
            return LancasterStemmer()
        elif stemmer == "snowball":
            return SnowballStemmer("english")
        else:
            raise ValueError("Unrecognized stemmer", stemmer)

    VALID_TAGS = set(
        [
            "FW",  # Foreign word
            "JJ",  # Adjective
            "JJR",  # Adjective, comparative
            "JJS",  # Adjective, superlative
            "NN",  # Noun, singular or mass
            "NNS",  # Noun, plural
            "NNP",  # Proper noun, singular
            "NNPS",  # Proper noun, plural
            "UH",  # Interjection
            "VB",  # Verb, base form
            "VBD",  # Verb, past tense
            "VBG",  # Verb, gerund or present participle
            "VBN",  # Verb, past participle
            "VBP",  # Verb, non-3rd person singular present
            "VBZ",  # Verb, 3rd person singular present
        ]
    )

    doc_freq = Counter()
    processed_data = []
    for i, text in enumerate(data):

        print("preprocessing document {} of {}".format(i, len(data)), end="\t\r")

        # tokens = word_tokenize(text.lower())
        if tokenizer is None:
            tokenizer = RegexpTokenizer(r"\w+")
        tokens = tokenizer.tokenize(text.lower())

        if stopwords:
            tokens = [t for t in tokens if t not in stopwords]

        if filter_numbers:
            tokens = [t for t in tokens if not t.isdigit()]

        if filter_pos_tag:
            tokens = [t for t, tag in nltk.pos_tag(tokens) if tag in VALID_TAGS]

        if lemmatizer is not None:
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(t) for t in tokens]

            if wordnet_filtering:
                # for t in tokens:
                #     print('wn', wordnet.synsets(t))
                tokens = [t for t in tokens if wordnet.synsets(t)]

        elif stemmer is not None:
            _stemmer = get_stemmer(stemmer)
            tokens = [_stemmer.stem(t) for t in tokens]

        tokens_cp = [t for t in tokens]
        if bigrams:
            tokens += ["+".join(t) for t in ngrams(tokens_cp, 2)]

        if trigrams:
            tokens += ["+".join(t) for t in ngrams(tokens_cp, 3)]

        for t in tokens:
            doc_freq[t] += 1

        tokens = [t for t in tokens if min_df <= doc_freq[t] <= max_df]
        processed_data.append(" ".join(t for t in tokens))

    return processed_data


def load_20newsgroups(
    data_path="data/20ng.pickle",
    train_valid_test_splits=[0.7, 0.1, 0.2],
    continuous=False,
    min_words=1,
    max_features=500,
    categories=None,  # ['alt.atheism', 'soc.religion.christian'],
    rand_gen=None,
):
    """

    """

    from sklearn.datasets import fetch_20newsgroups
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer

    if rand_gen is None:
        rand_gen = np.random.RandomState(0)

    try:
        logging.info("trying to load 20newsgroups from {}...".format(data_path))
        with open(data_path, "rb") as f:
            dataset, all_pp_documents = pickle.load(f)
        (train_x, train_y), (test_x, test_y), (valid_x, valid_y) = dataset

    except:
        logging.info("failed loading, fetching and preprocessing 20newsgroups...")
        #
        # retrieving raw text (all= train + test)
        # NOTE quotes include headers
        dataset = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"), random_state=rand_gen)

        #
        # extracting categories and filtering
        class_names = dataset.target_names

        if categories is None:
            categories = class_names

        for c in categories:
            assert c in set(class_names), "\t\t\tunrecognized category {} (available {})".format(c, class_names)
        logging.info("\t\tconsidering categories {} over {}".format(categories, class_names))

        labels = [class_names.index(c) for c in categories]
        classes = list(range(len(categories)))
        classes_map = {c: i for i, c in enumerate(labels)}
        print("considering labels", labels, classes)

        indices = list(np.where(np.isin(dataset.target, labels))[0])
        all_docs = [dataset.data[i] for i in indices]

        logging.info("\t\t\tfor a tot of {} docs over {} total docs".format(len(all_docs), len(dataset.data)))

        #
        # preprocessing the raw text
        all_pp_documents = preprocess_text(all_docs)
        print("LEN", len(all_pp_documents))

        # indices = [i for i in indices if len(all_pp_documents[i].split()) >= min_words]
        # logging.info('\n\nPROCESSED doc\n{}\n\n'.format(all_pp_documents[:20]))
        logging.info(
            "\t\t\tadditional filtering  {} docs over {} total docs".format(len(all_pp_documents), len(all_docs))
        )

        #
        # feature extraction (continuous [TF-IDF] or binary data [bag-of-words])
        vectorizer = None
        if continuous:
            _dtype = np.float64
            vectorizer = TfidfVectorizer(
                lowercase=False, stop_words="stop_words", strip_accents="ascii", max_features=max_features, dtype=_dtype
            )
        else:
            _dtype = np.int8
            vectorizer = TfidfVectorizer(
                lowercase=False,
                strip_accents="ascii",
                stop_words="stop_words",
                max_features=max_features,
                binary=True,
                use_idf=False,
                norm=None,
                dtype=_dtype,
            )
        vectorizer.fit(all_docs)
        X = vectorizer.transform(all_docs)
        #
        # from sparse to dense matrix
        X = X.todense().astype(_dtype)
        print("X", X.shape, X)

        #
        # assigning classes
        y = np.array([classes_map[dataset.target[i]] for i in indices])
        print("Y classes", y)

        assert len(y) == len(X)

        #
        # splitting into train, valid and test
        train_valid_test_splits = np.array(train_valid_test_splits)
        train_valid_test_splits = train_valid_test_splits / train_valid_test_splits.sum()

        trainv_x, test_x, trainv_y, test_y = train_test_split(
            X, y, test_size=train_valid_test_splits[2], random_state=rand_gen
        )

        logging.info("Reserving {} samples for testing".format(len(test_x)))

        train_x, valid_x, train_y, valid_y = train_test_split(
            trainv_x, trainv_y, test_size=train_valid_test_splits[1], random_state=rand_gen
        )
        logging.info("Reserving {} samples for validation".format(len(valid_x)))

        assert len(train_x) + len(valid_x) + len(test_x) == len(X)

        dataset = ((train_x, train_y), (test_x, test_y), (valid_x, valid_y))

        #
        # dumping processed data splits for further reuse
        logging.info("caching preprocessed dataset for further reuse...")
        with open(data_path, "wb") as f:
            pickle.dump((dataset, all_pp_documents), f)
            logging.info("Dumped splits and preprocessed text to {}".format(data_path))

    # self.examples = list(range(len(indices)))
    # self.y = dataset.target[indices]
    # self.documents = [all_pp_documents[i] for i in indices]
    # self.vectorizer = TfidfVectorizer(lowercase=False).fit(self.documents)
    # self.X = self.vectorizer.transform(self.documents)
    # self.full_documents = [dataset.data[i] for i in indices]

    assert len(train_x) == len(train_y)
    assert len(test_x) == len(test_y)
    assert len(valid_x) == len(valid_y)

    print("Loaded dataset with {}/{}/{} train/valid/test splits".format(len(train_x), len(valid_x), len(test_x)))
    return train_x, train_y, test_x, test_y, valid_x, valid_y


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    load_20newsgroups(
        data_path="data/20ng.pickle",
        continuous=False,
        min_words=1,
        max_features=500,
        categories=None,
        # categories=['alt.atheism', 'soc.religion.christian'],
        rand_gen=None,
    )
