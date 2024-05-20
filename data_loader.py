# -*- coding: utf-8 -*-

import os
from collections import Counter
from sentence_transformers import SentenceTransformer
import nltk
import numpy as np
import scipy.io
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import MinMaxScaler


def load_stackoverflow(data_path="data/stackoverflow/", use_SIF=True):

    # load SO embedding
    with open(data_path + 'vocab_withIdx.dic', 'r', encoding='utf-8') as inp_indx, \
            open(data_path + 'vocab_emb_Word2vec_48_index.dic', 'r', encoding='utf-8') as inp_dic, \
            open(data_path + 'vocab_emb_Word2vec_48.vec', encoding='utf-8') as inp_vec:
        pair_dic = inp_indx.readlines()
        word_index = {}
        for pair in pair_dic:
            word, index = pair.replace("\n", "").split("\t")
            word_index[word] = index

        index_word = {v: k for k, v in word_index.items()}

        del pair_dic

        emb_index = inp_dic.readlines()
        emb_vec = inp_vec.readlines()
        word_vectors = {}
        for index, vec in zip(emb_index, emb_vec):
            word = index_word[index.replace("\n", "")]
            word_vectors[word] = np.array(list((map(float, vec.split()))))

        del emb_index
        del emb_vec

    with open(data_path + 'title_StackOverflow.txt', 'r', encoding='utf-8') as inp_txt:
        all_lines = inp_txt.readlines()[:-1]
        text_file = " ".join([" ".join(nltk.word_tokenize(c)) for c in all_lines])
        word_count = Counter(text_file.split())
        total_count = sum(word_count.values())
        unigram = {}
        for item in word_count.items():
            unigram[item[0]] = item[1] / total_count
            # unigram[item[0]] is equal to the occurence of the word in the corpus divided by the total number of tokens in the corpus

        all_vector_representation = np.zeros(shape=(20000, 48))
        for i, line in enumerate(all_lines):
            word_sentence = nltk.word_tokenize(line)

            sent_rep = np.zeros(shape=(48,))
            j = 0

            for word in word_sentence:
                try:
                    wv = word_vectors[word]
                    j = j + 1
                except KeyError:
                    continue
                if use_SIF:
                    weight = 0.1 / (0.1 + unigram[word])
                    sent_rep += wv * weight # weighted sum of word vectors
                else:
                    sent_rep += wv
            if j != 0: # j represents the number of words in the sentence that have a word vector representation
                all_vector_representation[i] = sent_rep / j
            else:
                all_vector_representation[i] = sent_rep

    if use_SIF:
        pca = PCA(n_components=1)
        pca.fit(all_vector_representation)
        pca = pca.components_ # Principal axes
    # remove the first principal component (all_vector_representation.dot(pca.transpose()) * pca is the projection of all_vector_representation on the 1st principal component)
        XX1 = (
            all_vector_representation
            - all_vector_representation.dot(pca.transpose()) * pca 
        )

        XX = XX1

        scaler = MinMaxScaler()
        XX = scaler.fit_transform(XX)
    else:
        XX = all_vector_representation

    with open(data_path + 'label_StackOverflow.txt', encoding='utf-8') as label_file:
        y = np.array(list((map(int, label_file.readlines()))))
        print(y.dtype)

    return XX, y


def load_stackoverflow_sentence_transformers(model, data_path="data/stackoverflow/"):
    with open(data_path + "SearchSnippets.txt", "r", encoding="utf-8") as inp_indx:
        # read with readline and delete \n in each line
        lines = [line.rstrip() for line in inp_indx]
        x = model.encode(lines)
        with open(
            data_path + "label_StackOverflow.txt", "r", encoding="utf-8"
        ) as label_file:
            y = np.array(list((map(int, label_file.readlines()))))

        return x, y


def load_search_snippet2(data_path="data/SearchSnippets/", use_SIF=True):
    mat = scipy.io.loadmat(data_path + "SearchSnippets-STC2.mat")

    emb_index = np.squeeze(mat["vocab_emb_Word2vec_48_index"])
    emb_vec = mat["vocab_emb_Word2vec_48"]
    y = np.squeeze(mat["labels_All"])

    del mat

    rand_seed = 0

    # load SO embedding
    with open(data_path + 'SearchSnippets_vocab2idx.dic', 'r', encoding='utf-8') as inp_indx:
        pair_dic = inp_indx.readlines()
        word_index = {}
        for pair in pair_dic:
            word, index = pair.replace("\n", "").split("\t")
            word_index[word] = index

        index_word = {v: k for k, v in word_index.items()}

        del pair_dic

        word_vectors = {}
        for index, vec in zip(emb_index, emb_vec.T):
            word = index_word[str(index)]
            word_vectors[word] = vec

        del emb_index
        del emb_vec

    with open(data_path + 'SearchSnippets.txt', 'r', encoding='utf-8') as inp_txt:
        all_lines = inp_txt.readlines()[:-1]
        all_lines = [line for line in all_lines]
        text_file = " ".join([" ".join(nltk.word_tokenize(c)) for c in all_lines])
        word_count = Counter(text_file.split())
        total_count = sum(word_count.values())
        unigram = {}
        for item in word_count.items():
            unigram[item[0]] = item[1] / total_count

        all_vector_representation = np.zeros(shape=(12340, 48))
        for i, line in enumerate(all_lines):
            word_sentence = nltk.word_tokenize(line)

            sent_rep = np.zeros(shape=(48,))
            j = 0
            for word in word_sentence:
                try:
                    wv = word_vectors[word]
                    j = j + 1
                except KeyError:
                    continue
                if use_SIF:
                    weight = 0.1 / (0.1 + unigram[word])
                    sent_rep += wv * weight
                else:
                    sent_rep += wv
            if j != 0:
                all_vector_representation[i] = sent_rep / j
            else:
                all_vector_representation[i] = sent_rep

    if use_SIF:
        svd = TruncatedSVD(n_components=1, n_iter=20)
        svd.fit(all_vector_representation)
        svd = svd.components_

        XX = (
            all_vector_representation
            - all_vector_representation.dot(svd.transpose()) * svd
        )

        scaler = MinMaxScaler()
        XX = scaler.fit_transform(XX)
    else:
        XX = all_vector_representation

    return XX, y


def load_searchsnippets_sentence_transformers(model, data_path="data/SearchSnippets/"):
    with open(data_path + "SearchSnippets.txt", "r", encoding="utf-8") as inp_indx:
        # read with readline and delete \n in each line
        lines = [line.rstrip() for line in inp_indx]
        x = model.encode(lines)

        with open(
            data_path + "SearchSnippets_gnd.txt", "r", encoding="utf-8"
        ) as label_file:
            y = np.array(list((map(int, label_file.readlines()))))

        return x, y


def load_biomedical(data_path="data/Biomedical/", use_SIF=True):
    mat = scipy.io.loadmat(data_path + "Biomedical-STC2.mat")

    emb_index = np.squeeze(mat["vocab_emb_Word2vec_48_index"])
    emb_vec = mat["vocab_emb_Word2vec_48"]
    y = np.squeeze(mat["labels_All"])

    del mat

    rand_seed = 0

    # load SO embedding
    with open(data_path + 'Biomedical_vocab2idx.dic', 'r', encoding='utf-8') as inp_indx:
        # open(data_path + 'vocab_emb_Word2vec_48_index.dic', 'r') as inp_dic, \
        # open(data_path + 'vocab_emb_Word2vec_48.vec') as inp_vec:
        pair_dic = inp_indx.readlines()
        word_index = {}
        for pair in pair_dic:
            word, index = pair.replace("\n", "").split("\t")
            word_index[word] = index

        index_word = {v: k for k, v in word_index.items()}

        del pair_dic

        word_vectors = {}
        for index, vec in zip(emb_index, emb_vec.T):
            word = index_word[str(index)]
            word_vectors[word] = vec

        del emb_index
        del emb_vec

    with open(data_path + 'Biomedical.txt', 'r', encoding='utf-8') as inp_txt:
        all_lines = inp_txt.readlines()[:-1]
        # print(sum([len(line.split()) for line in all_lines])/20000) #avg length
        text_file = " ".join([" ".join(nltk.word_tokenize(c)) for c in all_lines])
        word_count = Counter(text_file.split())
        total_count = sum(word_count.values())
        unigram = {}
        for item in word_count.items():
            unigram[item[0]] = item[1] / total_count

        all_vector_representation = np.zeros(shape=(20000, 48))
        for i, line in enumerate(all_lines):
            word_sentence = nltk.word_tokenize(line)

            sent_rep = np.zeros(shape=(48,))
            j = 0
            for word in word_sentence:
                try:
                    wv = word_vectors[word]
                    j = j + 1
                except KeyError:
                    continue
                if use_SIF:
                    weight = 0.1 / (0.1 + unigram[word])
                    sent_rep += wv * weight
                else:
                    sent_rep += wv
            if j != 0:
                all_vector_representation[i] = sent_rep / j
            else:
                all_vector_representation[i] = sent_rep

    if use_SIF:
        svd = TruncatedSVD(n_components=1, random_state=rand_seed, n_iter=20)
        svd.fit(all_vector_representation)
        svd = svd.components_
        XX = (
            all_vector_representation
            - all_vector_representation.dot(svd.transpose()) * svd
        )

        scaler = MinMaxScaler()
        XX = scaler.fit_transform(XX)
    else:
        XX = all_vector_representation
    return XX, y


def load_biomedical_sentence_transformers(model, data_path="data/Biomedical/"):
    with open(data_path + "Biomedical.txt", "r", encoding="utf-8") as inp_indx:
        # read with readline and delete \n in each line
        lines = [line.rstrip() for line in inp_indx]
        x = model.encode(lines)

        with open(
            data_path + "Biomedical_gnd.txt", "r", encoding="utf-8"
        ) as label_file:
            y = np.array(list((map(int, label_file.readlines()))))

        return x, y


def load_data(dataset_name,use_SIF=True):
    print("load data")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    if dataset_name == "stackoverflow":
        # return load_stackoverflow_sentence_transformers(model=model)
        return load_stackoverflow(use_SIF=use_SIF)
    elif dataset_name == "biomedical":
        # return load_biomedical_sentence_transformers(model=model)
        return load_biomedical(use_SIF=use_SIF)
    elif dataset_name == "searchSnippets":
        # return load_searchsnippets_sentence_transformers(model=model)
        return load_search_snippet2(use_SIF=use_SIF)
    else:
        raise Exception("dataset not found...")
