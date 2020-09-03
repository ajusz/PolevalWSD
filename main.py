from os import listdir
from os.path import isfile, join
import string
import gensim
import numpy as np
import nltk
from tqdm import tqdm
from collections import defaultdict


class DataStore(object):
    def __init__(self, model_path, embeddings_path):
        self.model = gensim.models.Word2Vec.load(model_path)
        self.word2vec_matrix = self.model.wv.vectors
        self.word2index = {token: token_index for token_index, token in enumerate(self.model.wv.index2word)}
        self.embeddings_path = embeddings_path
        self.base_forms_idf = None
        self.base_forms_embeddings = None
        self.lemma_synsets_mapping = None
        self.synsets = None
        self.load()

    def load(self):
        self.read_synsets('data/wordnet_data/synsets.txt')
        self.read_base_forms_idf('data/base_forms_idf.txt')
        self.base_forms_embeddings = self.read_embeddings()
        self.create_lemma_synsets_mapping('data/wordnet_data/lemmas.txt', 'data/wordnet_data/lexicalunits.txt')

    def read_base_forms_idf(self, filename):
        print('Reading base forms idf')
        self.base_forms_idf = defaultdict(int)
        with open(filename) as file:
            for line in tqdm(file):
                base_form, idf = line.strip().split()
                self.base_forms_idf[base_form] = float(idf)

    def read_embeddings(self):
        print('Reading embeddings')
        embeddings = {}
        with open(self.embeddings_path) as file:
            next(file)
            for line in tqdm(file):
                line = line.split()
                word = line[0].lower()
                vector = np.array(line[1:], dtype=np.float32)
                embeddings[word] = vector
        return embeddings

    def read_synsets(self, synsets_filename):
        print('Reading synsets')
        with open(synsets_filename) as synsets_file:
            self.synsets = [line.strip() for line in tqdm(synsets_file)]

    def create_lemma_synsets_mapping(self, lemmas_filename, lexicalunits_filename):
        print('Creating lemma synset mappings')
        with open(lemmas_filename) as lemmas_file:
            lemmas = [line.strip().rsplit(',', 1)[0] for line in lemmas_file]
        with open(lexicalunits_filename) as lexicalunits_file:
            lexicalunits = [line.strip().split() for line in lexicalunits_file]
        self.lemma_synsets_mapping = defaultdict(list)
        for lemma_id, synset_id in tqdm(lexicalunits):
            self.lemma_synsets_mapping[lemmas[int(lemma_id)-1]].append(int(synset_id)-1)


class Disambiguator(object):
    def __init__(self, data_store):
        self.data_store = data_store

    def find_important_words(self, text):
        important_word_ids = []
        punctuation = string.punctuation + '–'
        stopwords = set(nltk.corpus.stopwords.words('polish'))
        important_words = []
        for i, word in enumerate(text):
            word = word.strip(punctuation)
            if word and word not in stopwords:
                important_words.append(word)
                important_word_ids.append(i)
        return important_words, important_word_ids

    def calculate_embedding(self, i, important_words, k=6, alpha=0.95):
        start = max(i - k, 0)
        end = i + k
        context = important_words[start:i] + important_words[i+1:end]
        embedding = np.zeros(next(iter(self.data_store.base_forms_embeddings.values())).shape)
        j = start
        for base_form in context:
            if j == i:
                j += 1
            if base_form in self.data_store.base_forms_embeddings:
                embedding += self.data_store.base_forms_embeddings[base_form] \
                             * self.data_store.base_forms_idf[base_form] * (alpha**abs(i-j))
            j += 1
        return embedding

    def find_best_sense(self, word2vec_matrix, ids, embedding):
        distances = gensim.models.KeyedVectors.cosine_similarities(embedding, word2vec_matrix[ids])
        best_index = np.argmax(distances)
        return best_index

    def disambiguate(self, text, k=6):
        important_words, important_word_indices = self.find_important_words(text)
        id_sense_mapping = {}
        for i, lemma in zip(important_word_indices, important_words):
            senses = []
            for synset_id in self.data_store.lemma_synsets_mapping[lemma]:
                sense = '{}/{}'.format(lemma, self.data_store.synsets[synset_id])
                if sense in self.data_store.word2index:
                    senses.append(sense)
            if len(senses) > 1:
                embedding = self.calculate_embedding(i, important_words, k)
                sense_indices = [self.data_store.word2index[sense] for sense in senses]
                best_sense_id = self.find_best_sense(self.data_store.word2vec_matrix, sense_indices, embedding)
                best_sense = senses[best_sense_id]
                id_sense_mapping[i] = best_sense.rsplit('/', 1)[1]
        return id_sense_mapping

    def disambiguate_conll_file(self, filename, input_directory, results_directory, k=6):
        text = []
        with open(join(input_directory, filename)) as input_file:
            next(input_file)
            for line in input_file:
                lemma = line.strip().split('\t')[3]
                text.append(lemma.lower())
        id_sense_mapping = self.disambiguate(text, k)
        with open(join(input_directory, filename)) as input_file:
            with open(join(results_directory, filename), 'w') as output_file:
                output_file.write('{}\tWN_ID\n'.format(next(input_file).strip()))
                for line in input_file:
                    id = int(line.strip().split('\t')[0])
                    output_file.write(line.strip())
                    if id in id_sense_mapping:
                        output_file.write('\t{}'.format(id_sense_mapping[id]))
                    else:
                        output_file.write('\t_')
                    output_file.write('\n')


def read_lemmas(lemmas_filename):
    result = set()
    with open(lemmas_filename) as lemmas_file:
        for line in lemmas_file:
            result.add(line.strip().split(',')[-1])
    return result


if __name__ == '__main__':
    input_kpwr_directory = 'data/testdata/kpwr'
    output_kpwr_directory = 'data/testdata_with_labels/testdata/kpwr'
    kpwr_filenames = [f for f in listdir(input_kpwr_directory) if isfile(join(input_kpwr_directory, f))]

    input_sherlock_directory = 'data/testdata/sherlock'
    output_sherlock_directory = 'data/testdata_with_labels/testdata/sherlock'
    sherlock_filenames = [f for f in listdir(input_sherlock_directory) if isfile(join(input_sherlock_directory, f))]

    data_store = DataStore(model_path='models/poleval_lemmas_word2vec_iter8.model',
                           embeddings_path='data/embeddings/lemma_embeddings_iter8.txt')
    disambiguator = Disambiguator(data_store)
    for filename in kpwr_filenames:
        disambiguator.disambiguate_conll_file(filename, input_kpwr_directory, output_kpwr_directory)
    for filename in sherlock_filenames:
        disambiguator.disambiguate_conll_file(filename, input_sherlock_directory, output_sherlock_directory)