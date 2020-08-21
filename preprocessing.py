import in_place
import random
import string
import gensim
import numpy as np
import nltk
from tqdm import tqdm
import time
from collections import defaultdict
from functools import partial


class Tokenizer(object):
    def __init__(self):
        self.punctuation = string.punctuation + '–„”…'
        self.stopwords = set(nltk.corpus.stopwords.words('polish'))

    def contains_number(self, word):
        return any(l.isdigit() for l in word)

    def tokenize(self, text):
        text = text.lower().split()
        cleared_text = []
        for word in text:
            word = word.strip(self.punctuation)
            if word and word not in self.stopwords and not self.contains_number(word):
                cleared_text.append(word)
        return cleared_text

    def tokenize_file(self, input_filename, output_filename):
        with open(output_filename, 'w') as output_file:
            for line in tqdm(open(input_filename)):
                tokenized_line = self.tokenize(line)
                output_file.write('{}\n'.format(' '.join(tokenized_line)))


class DataStore(object):
    def __init__(self, read_definitions=True, read_relations=True):
        self.base_forms = None
        self.definitions = None
        self.embeddings = None
        self.base_forms_embeddings = None
        self.lemma_synsets_mapping = None
        self.synset_lemmas_mapping = None
        self.synsets = None
        self.related_synsets = None
        self.related_lemmas_filename = 'data/wordnet_data/related_lemmas.txt'
        self.load(read_definitions, read_relations)

    def load(self, read_definitions=True, read_relations=True):
        self.read_base_forms('data/polimorfologik/polimorfologik-2.1.txt')
        if read_definitions:
            self.read_definitions('data/wordnet_data/synset_defs_examples.txt')
        self.read_synsets('data/wordnet_data/synsets.txt')
        # self.embeddings = self.read_embeddings('data/embeddings/nkjp+wiki-forms-all-100-cbow-hs.txt')
        # self.base_forms_embeddings = self.read_embeddings('data/embeddings/nkjp+wiki-lemmas-all-100-cbow-hs.txt')
        self.create_lemma_synset_mappings('data/wordnet_data/lemmas.txt', 'data/wordnet_data/lexicalunits.txt')
        if read_relations:
            self.read_relations('data/wordnet_data/synset_rels.txt')

    def read_base_forms(self, filename):
        print('Reading base forms')
        self.base_forms = defaultdict(list)
        with open(filename) as file:
            for line in tqdm(file):
                columns = line.split(';')
                base_form = columns[0].lower()
                word = columns[1].lower()
                self.base_forms[word].append(base_form)

    def read_embeddings(self, filename):
        print('Reading embeddings')
        embeddings = {}
        with open(filename) as file:
            next(file)
            for line in tqdm(file):
                line = line.split()
                word = line[0].lower()
                vector = np.array(line[1:], dtype=np.float32)
                embeddings[word] = vector
        return embeddings

    def save_base_forms_embeddings(self, filename):
        with open(filename, 'w') as file:
            embedding_length = len(next(iter(self.base_forms_embeddings.values())))
            embeddings_count = len(self.base_forms_embeddings)
            file.write('{} {}\n'.format(embeddings_count, embedding_length))
            for base_form, embedding in tqdm(self.base_forms_embeddings.items()):
                file.write('{} {}\n'.format(base_form, str(embedding)[1:-1].replace('\n', '')))

    def read_synsets(self, synsets_filename):
        print('Reading synsets')
        with open(synsets_filename) as synsets_file:
            self.synsets = [line.strip() for line in tqdm(synsets_file)]

    def read_definitions(self, filename):
        print('Reading definitions')
        tokenizer = Tokenizer()
        self.definitions = defaultdict(list)
        with open(filename) as file:
            for line in tqdm(file):
                synset, definition = line.strip().split(maxsplit=1)
                self.definitions[synset] += tokenizer.tokenize(definition)

    def read_relations(self, filename):
        print('Reading relations')
        self.related_synsets = defaultdict(set)
        with open(filename) as file:
            for line in tqdm(file):
                s1, s2, relation = line.strip().split(maxsplit=2)
                important_relations = ['hiperonimia', 'hiponimia', 'część', 'element kolekcji', 'fuzzynimia_synsetów',
                                       'bliskoznaczność', 'meronimia czasownikowa', 'meronimia podsytuacji',
                                       'meronimia sytuacji towarzyszącej', 'holonimia sytuacji towarzyszącej'
                                                                           'holonimia czasownikowa',
                                       'holonimia podsytuacji', 'wartość_cechy']
                if relation in important_relations:
                    synset1 = self.synsets[int(s1) - 1]
                    synset2 = self.synsets[int(s2) - 1]
                    self.related_synsets[synset1].add(synset2)
                    self.related_synsets[synset2].add(synset1)

    def write_related_lemmas(self):
        with open(self.related_lemmas_filename, 'w') as output_file:
            for synset, rs in self.related_synsets.items():
                for lemma1 in self.synset_lemmas_mapping[synset]:
                    for related_synset in rs:
                        for lemma2 in self.synset_lemmas_mapping[related_synset]:
                            output_file.write('{}/{} {}/{}\n'.format(lemma1, synset, lemma2, related_synset))

    def create_lemma_synset_mappings(self, lemmas_filename, lexicalunits_filename):
        print('Creating lemma synset mappings')
        with open(lemmas_filename) as lemmas_file:
            lemmas = [line.strip().rsplit(',', 1)[0] for line in lemmas_file]
        with open(lexicalunits_filename) as lexicalunits_file:
            lexicalunits = [line.strip().split() for line in lexicalunits_file]
        self.lemma_synsets_mapping = defaultdict(list)
        for lemma_id, synset_id in tqdm(lexicalunits):
            self.lemma_synsets_mapping[lemmas[int(lemma_id) - 1]].append(int(synset_id) - 1)
        self.synset_lemmas_mapping = defaultdict(list)
        for lemma_id, synset_id in tqdm(lexicalunits):
            self.synset_lemmas_mapping[self.synsets[int(synset_id) - 1]].append(lemmas[int(lemma_id) - 1])


class SynsetsMatrixCreator(object):
    def __init__(self, data_store):
        self.idf = None
        self.base_forms_idf = None
        self.data_store = data_store
        self.calculate_idf()
        self.calculate_base_forms_idf()

    def increase_df(self, word, terms, df):
        if word in terms:
            return
        terms.add(word)
        df[word] += 1

    def calculate_idf(self):
        df = defaultdict(int)
        N = len(self.data_store.definitions)
        for synset in tqdm(self.data_store.synsets):
            terms = set()
            for word in self.data_store.definitions[synset]:
                self.increase_df(word, terms, df)
            for lemma in self.data_store.synset_lemmas_mapping[synset]:
                for word in lemma.split():
                    self.increase_df(word, terms, df)
        self.idf = {word: np.log(N / df_t) for word, df_t in df.items()}

    def calculate_base_forms_idf(self):
        df = defaultdict(int)
        N = len(self.data_store.definitions)
        for synset in tqdm(self.data_store.synsets):
            terms = set()
            for word in self.data_store.definitions[synset]:
                for base_form in self.data_store.base_forms.get(word, [word]):
                    self.increase_df(base_form, terms, df)
            for lemma in self.data_store.synset_lemmas_mapping[synset]:
                for word in lemma.split():
                    for base_form in self.data_store.base_forms.get(word, [word]):
                        self.increase_df(base_form, terms, df)
        self.base_forms_idf = {word: np.log(N / df_t) for word, df_t in df.items()}

    def add_word_embedding(self, matrix, i, word, multiplier=1.0):
        if word in self.data_store.embeddings:
            matrix[i] += multiplier * self.data_store.embeddings[word] * self.idf[word]
        else:
            added_embeddings_count = 0
            embedding_to_be_added = np.zeros(next(iter(self.data_store.base_forms_embeddings.values())).shape)
            for base_form in self.data_store.base_forms.get(word, [word]):
                if base_form in self.data_store.base_forms_embeddings:
                    embedding_to_be_added += multiplier * self.data_store.base_forms_embeddings[base_form] \
                                             * self.base_forms_idf[base_form]
                    added_embeddings_count += 1
            if added_embeddings_count > 0:
                matrix[i] += embedding_to_be_added / added_embeddings_count

    def add_lemma_embedding(self, matrix, i, lemma, multiplier=1.0):
        for word in lemma.split():
            self.add_word_embedding(matrix, i, word, multiplier)

    def calculate_synsets_matrix(self, filename='synsets_matrix_poleval.npz'):
        synsets_matrix = np.zeros((len(self.data_store.synsets),
                                   next(iter(self.data_store.embeddings.values())).shape[0]))
        for i, synset in enumerate(tqdm(self.data_store.synsets)):
            for word in self.data_store.definitions[synset]:
                self.add_word_embedding(synsets_matrix, i, word)
            for s in [synset] + list(self.data_store.related_synsets[synset]):
                for lemma in self.data_store.synset_lemmas_mapping[s]:
                    self.add_lemma_embedding(synsets_matrix, i, lemma)
        synset_norms = np.linalg.norm(synsets_matrix, axis=1)[:, None]
        synset_norms[synset_norms == 0] = 0.0000001
        synsets_matrix = synsets_matrix / synset_norms
        np.savez_compressed(filename, synsets_matrix)


class SentenceDisambiguator(object):
    def __init__(self, data_store):
        self.data_store = data_store

    def calculate_embedding(self, i, important_words, idf, base_forms_idf, k=6, alpha=0.95):
        start = max(i - k, 0)
        end = i + k
        context = important_words[start:i] + important_words[i + 1:end]
        embedding = np.zeros(next(iter(self.data_store.base_forms_embeddings.values())).shape)
        j = start
        for word in context:
            if j == i:
                j += 1
            if word in self.data_store.embeddings:
                embedding += self.data_store.embeddings[word] * idf[word] * (alpha ** abs(i - j))
            else:
                added_base_forms_embeddings_count = 0
                embedding_to_be_added = np.zeros(next(iter(self.data_store.base_forms_embeddings.values())).shape)
                for base_form in self.data_store.base_forms.get(word, [word]):
                    if base_form in self.data_store.base_forms_embeddings:
                        embedding_to_be_added += self.data_store.base_forms_embeddings[base_form] \
                                                 * base_forms_idf[base_form] * (alpha ** abs(i - j))
                        added_base_forms_embeddings_count += 1
                if added_base_forms_embeddings_count > 0:
                    embedding += embedding_to_be_added / added_base_forms_embeddings_count
            j += 1
        return embedding

    def calculate_lemma_embedding(self, i, lemmatized_text, base_forms_idf, k=6, alpha=0.95):
        start = max(i - k, 0)
        end = i + k
        context = lemmatized_text[start:i] + lemmatized_text[i + 1:end]
        embedding = np.zeros(next(iter(self.data_store.base_forms_embeddings.values())).shape)
        j = start
        for word in context:
            if j == i:
                j += 1
            if word in self.data_store.base_forms_embeddings:
                embedding += self.data_store.base_forms_embeddings[word] \
                             * base_forms_idf[word] * (alpha ** abs(i - j))
            j += 1
        return embedding

    def calculate_word2vec_embedding(self, i, disambiguated_text, word2vec_matrix, word2index,
                                     base_forms_idf, k=6, alpha=0.95):
        start = max(i - k, 0)
        end = i + k
        context = disambiguated_text[start:i] + disambiguated_text[i + 1:end]
        embedding = np.zeros(word2vec_matrix[1].shape)
        j = start
        for word in context:
            if j == i:
                j += 1
            splitted_word = word.split('/')
            if word in word2index:
                embedding += word2vec_matrix[word2index[word]] * base_forms_idf[splitted_word[0]] * (
                            alpha ** abs(i - j))
            j += 1
        return embedding

    def disambiguate_tokenized(self, tokenized_text, synsets_matrix, idf, base_forms_idf, k=6):
        result = []
        lemmatized = []
        for i, word in enumerate(tokenized_text):
            sense_lemmas = defaultdict(list)
            for base_form in self.data_store.base_forms.get(word, [word]):
                for synset_id in self.data_store.lemma_synsets_mapping[base_form]:
                    sense_lemmas[synset_id].append(base_form)
            if len(sense_lemmas) > 1:
                embedding = self.calculate_embedding(i, tokenized_text, idf, base_forms_idf, k)
                best_synset_id = self.find_best_sense(synsets_matrix, list(sense_lemmas.keys()), embedding)
                best_synset = self.data_store.synsets[best_synset_id]
                base_form = random.choice(sense_lemmas[best_synset_id])
                result.append('{base_form}/{synset}'.format(base_form=base_form, synset=best_synset))
            else:
                base_form = random.choice(list(self.data_store.base_forms.get(word, [word])))
                result.append(base_form)
            lemmatized.append(base_form)
        return ' '.join(result), ' '.join(lemmatized)

    def disambiguate_lemmatized(self, lemmatized_text, synsets_matrix, base_forms_idf, k=6):
        result = []
        for i, word in enumerate(lemmatized_text):
            senses = []
            for synset_id in self.data_store.lemma_synsets_mapping[word]:
                senses.append(synset_id)
            if len(senses) > 1:
                embedding = self.calculate_lemma_embedding(i, lemmatized_text, base_forms_idf, k)
                best_synset_id = self.find_best_sense(synsets_matrix, senses, embedding)
                best_synset = self.data_store.synsets[best_synset_id]
                result.append('{lemma}/{synset}'.format(lemma=lemmatized_text[i], synset=best_synset))
            else:
                result.append(word)
        return ' '.join(result)

    def disambiguate_lemmatized_after_word2vec(self, lemmatized_text, word2vec_matrix, word2index,
                                               base_forms_idf, k=6):
        result = []
        for i, word in enumerate(lemmatized_text):
            senses = []
            for synset_id in self.data_store.lemma_synsets_mapping[word]:
                sense = '{}/{}'.format(word, self.data_store.synsets[synset_id])
                if sense in word2index:
                    senses.append(sense)
            if len(senses) > 1:
                embedding = self.calculate_lemma_embedding(i, lemmatized_text, base_forms_idf, k)
                sense_indices = [word2index[sense] for sense in senses]
                best_sense_id = self.find_word2vec_best_sense(word2vec_matrix, sense_indices, embedding)
                best_sense = senses[best_sense_id]
                result.append(best_sense)
            else:
                result.append(word)
        return ' '.join(result)

    def improve_sense_annotations(self, disambiguated_text, word2vec_matrix, word2index,
                                  base_forms_idf, k=6):
        improved_count = 0
        for i, word in enumerate(disambiguated_text):
            splitted_word = word.split('/')
            if len(splitted_word) == 1:
                continue
            elif len(splitted_word) == 2:
                word, old_sense = splitted_word
            else:
                print('This word has too many parts: {}'.format(word))
                continue
            senses = []
            for synset_id in self.data_store.lemma_synsets_mapping[word]:
                sense = '{}/{}'.format(word, self.data_store.synsets[synset_id])
                if sense in word2index:
                    senses.append(sense)
            if len(senses) > 1:
                embedding = self.calculate_word2vec_embedding(i, disambiguated_text, word2vec_matrix, word2index,
                                                              base_forms_idf, k)
                sense_ids = [word2index[sense] for sense in senses]
                best_sense_id = self.find_word2vec_best_sense(word2vec_matrix, sense_ids, embedding)
                best_sense = senses[best_sense_id]
                if best_sense != disambiguated_text[i]:
                    disambiguated_text[i] = best_sense
                    improved_count += 1
        return ' '.join(disambiguated_text), improved_count

    def find_best_sense(self, matrix, ids, embedding, return_index=False):
        distances = self.cosine_similarity(matrix[ids], embedding)
        best_index = np.argmax(distances)
        if return_index:
            return best_index
        return ids[best_index]

    def find_word2vec_best_sense(self, word2vec_matrix, ids, embedding):
        distances = gensim.models.KeyedVectors.cosine_similarities(embedding, word2vec_matrix[ids])
        best_index = np.argmax(distances)
        return best_index

    def cosine_similarity(self, matrix, embedding):
        embedding_norm = np.linalg.norm(embedding)
        if embedding_norm == 0:
            embedding_norm = 0.0000001
        similarity_vector = matrix.dot(embedding.T) / embedding_norm
        return similarity_vector.flatten()


class FileDisambiguator(object):
    def __init__(self, corpora):
        self.sentence_disambiguator = SentenceDisambiguator(corpora.data_store)
        self.corpora = corpora

    def disambiguate_tokenized_file(self, synsets_matrix):
        print('Disambiguating tokenized file')
        with open(self.corpora.sense_annotated_corpora, 'w') as disambiguated_file:
            with open(self.corpora.lemmatized_corpora, 'w') as lemmatized_file:
                for line in tqdm(open(self.corpora.tokenized_corpora)):
                    tokenized_line = line.split()
                    disambiguated_line, lemmatized_line = self.sentence_disambiguator.disambiguate_tokenized(
                        tokenized_line, synsets_matrix, self.corpora.idf, self.corpora.base_forms_idf)
                    disambiguated_file.write('{}\n'.format(disambiguated_line))
                    lemmatized_file.write('{}\n'.format(lemmatized_line))

    def disambiguate_lemmatized_file(self, synsets_matrix):
        print('Disambiguating lemmatized file')
        with open(self.corpora.sense_annotated_corpora, 'w') as disambiguated_file:
            for line in tqdm(open(self.corpora.lemmatized_corpora)):
                lemmatized_line = line.split()
                disambiguated_line = self.sentence_disambiguator.disambiguate_lemmatized(
                    lemmatized_line, synsets_matrix, self.corpora.base_forms_idf)
                disambiguated_file.write('{}\n'.format(disambiguated_line))

    def disambiguate_lemmatized_file_after_word2vec(self, word2vec_matrix, word2index):
        print('Disambiguating lemmatized file after word2vec')
        with open(self.corpora.sense_annotated_corpora, 'w') as disambiguated_file:
            for line in tqdm(open(self.corpora.lemmatized_corpora)):
                lemmatized_line = line.split()
                disambiguated_line = self.sentence_disambiguator.disambiguate_lemmatized_after_word2vec(
                    lemmatized_line, word2vec_matrix, word2index, self.corpora.base_forms_idf)
                disambiguated_file.write('{}\n'.format(disambiguated_line))

    def improve_sense_annotations(self, word2vec_matrix, word2index, max_iter=1):
        print('Improving sense annotations')
        improved_count = np.Inf
        i = 0
        while improved_count > 10000 and i < max_iter:
            improved_count = 0
            with in_place.InPlace(self.corpora.sense_annotated_corpora) as file:
                for line in tqdm(file):
                    line = line.split()
                    improved_line, count = self.sentence_disambiguator.improve_sense_annotations(
                        line, word2vec_matrix, word2index, self.corpora.base_forms_idf)
                    improved_count += count
                    file.write('{}\n'.format(improved_line))
            print('Improved {} sense annotations. Iteration: {}'.format(improved_count, i))
            i += 1


class Corpora(object):
    def __init__(self, data_store, calculate_idf=True, calculate_base_forms_idf=True):
        self.data_store = data_store
        self.corpora = 'data/training/data/train_shuf.txt'
        self.tokenized_corpora = 'data/training/data/train_shuf_tokenized.txt'
        self.lemmatized_corpora = 'data/training/data/train_shuf_lemmatized.txt'
        self.sense_annotated_corpora_base_name = 'data/training/results/train_shuf_lemmatized_sense_annotated'
        self.sense_annotated_corpora = '{}_iter0.txt'.format(self.sense_annotated_corpora_base_name)
        self.idf = self.calculate_idf(self.tokenized_corpora) if calculate_idf else None
        self.base_forms_idf = self.calculate_base_forms_idf(self.tokenized_corpora) if calculate_base_forms_idf \
            else None

    def increase_df(self, word, terms, df):
        if word in terms:
            return
        terms.add(word)
        df[word] += 1

    def calculate_idf(self, filename):
        df = defaultdict(int)
        N = 0
        print('Calculating idf')
        with open(filename) as file:
            for line in tqdm(file):
                terms = set()
                for word in line.split():
                    self.increase_df(word, terms, df)
                N += 1
        return {word: np.log(N / df_t) for word, df_t in df.items()}

    def calculate_base_forms_idf(self, filename):
        df = defaultdict(int)
        N = 0
        print('Calculating base forms idf')
        with open(filename) as file:
            for line in tqdm(file):
                terms = set()
                for word in line.split():
                    for base_form in self.data_store.base_forms.get(word, [word]):
                        self.increase_df(base_form, terms, df)
                N += 1
        return {word: np.log(N / df_t) for word, df_t in df.items()}

    def save_base_forms_idf(self, filename):
        with open(filename, 'w', encoding='utf-8') as file:
            for word, idf in self.base_forms_idf.items():
                file.write('{} {}\n'.format(word, str(idf)))

    def calculate_tf(self):
        print('Calculating new term frequencies')
        tf = defaultdict(int)
        with open(self.sense_annotated_corpora) as file:
            for line in tqdm(file):
                for word in line.split():
                    tf[word] += 1
        return tf

    def calculate_new_embeddings(self, word2vec_matrix, word2index):
        print('Calculating new embeddings')
        tf = self.calculate_tf()
        new_embeddings = defaultdict(partial(np.zeros, word2vec_matrix[1].shape))
        missing_embeddings_count = 0
        all_embeddings_count = 0
        for lemma, synset_ids in tqdm(self.data_store.lemma_synsets_mapping.items()):
            tf_sum = 0
            if len(synset_ids) > 1:
                for synset_id in synset_ids:
                    sense_annotated_lemma = '{}/{}'.format(lemma, self.data_store.synsets[synset_id])
                    all_embeddings_count += 1
                    try:
                        index = word2index[sense_annotated_lemma]
                        new_embeddings[lemma] += word2vec_matrix[index] * tf[sense_annotated_lemma]
                        tf_sum += tf[sense_annotated_lemma]
                    except KeyError:
                        missing_embeddings_count += 1
                if tf_sum != 0:
                    new_embeddings[lemma] /= tf_sum
            else:
                try:
                    all_embeddings_count += 1
                    index = word2index[lemma]
                    new_embeddings[lemma] = word2vec_matrix[index]
                except KeyError:
                    missing_embeddings_count += 1
        additional_embeddings_count = 0
        for word, index in tqdm(word2index.items()):
            if '/' not in word and word not in new_embeddings:
                new_embeddings[word] = word2vec_matrix[word2index[word]]
                additional_embeddings_count += 1
        self.data_store.base_forms_embeddings = new_embeddings
        print('Missing embeddings: {}/{}'.format(missing_embeddings_count, all_embeddings_count))
        print('Additional embeddings count: {}'.format(additional_embeddings_count))


class Sentences(object):
    def __init__(self, filenames):
        self.filenames = filenames

    def __iter__(self):
        for filename in self.filenames:
            for line in open(filename):
                yield line.split()


class Model(object):
    def __init__(self, data_store):
        self.embeddings_base_path = 'data/embeddings/lemma_embeddings'
        self.embeddings_path = '{}_iter0.txt'.format(self.embeddings_base_path)
        self.model_base_path = 'models/poleval_lemmas_word2vec'
        self.model_path = '{}_iter0.model'.format(self.model_base_path)
        self.corpora = Corpora(data_store)
        self.disambiguator = FileDisambiguator(self.corpora)

    def train(self, synsets_matrix_filename='synsets_matrix_poleval.npz',
              iterations=10):
        synsets_matrix = np.load(synsets_matrix_filename)['arr_0']
        self.disambiguator.disambiguate_tokenized_file(synsets_matrix)
        del self.corpora.idf
        self.corpora.base_forms_idf = self.corpora.calculate_idf(self.corpora.lemmatized_corpora)
        sentences = Sentences([self.corpora.sense_annotated_corpora, self.corpora.data_store.related_lemmas_filename])
        t0 = time.time()
        model = gensim.models.Word2Vec(sentences, size=100, workers=8, min_count=3)
        model.save(self.model_path)
        model_training_duration = time.time() - t0
        print('Model training duration: {}'.format(time.strftime("%H:%M:%S", time.gmtime(model_training_duration))))
        print('------------------------------------------------------------------\n')
        word2index = {token: token_index for token_index, token in enumerate(model.wv.index2word)}
        self.corpora.calculate_new_embeddings(model.wv.vectors, word2index)
        self.corpora.data_store.save_base_forms_embeddings(self.embeddings_path)

        for i in range(1, iterations):
            print('Iteration {}:'.format(i))
            self.corpora.sense_annotated_corpora = '{}_iter{}.txt'.format(
                self.corpora.sense_annotated_corpora_base_name, i)
            self.disambiguator.disambiguate_lemmatized_file_after_word2vec(model.wv.vectors, word2index)
            self.disambiguator.improve_sense_annotations(model.wv.vectors, word2index)
            sentences = Sentences(
                [self.corpora.sense_annotated_corpora, self.corpora.data_store.related_lemmas_filename])
            self.model_path = '{}_iter{}.model'.format(self.model_base_path, i)
            t0 = time.time()
            model = gensim.models.Word2Vec(sentences, size=100, workers=8, min_count=3)
            model.save(self.model_path)
            model_training_duration = time.time() - t0
            print('Model training duration: {}'.format(time.strftime("%H:%M:%S", time.gmtime(model_training_duration))))
            self.embeddings_path = '{}_iter{}.txt'.format(self.embeddings_base_path, i)
            word2index = {token: token_index for token_index, token in enumerate(model.wv.index2word)}
            self.corpora.calculate_new_embeddings(model.wv.vectors, word2index)
            self.corpora.data_store.save_base_forms_embeddings(self.embeddings_path)
            print('------------------------------------------------------------------\n')


if __name__ == "__main__":
    data_store = DataStore(read_definitions=False, read_relations=False)
    model = Model(data_store)
    model.train()
