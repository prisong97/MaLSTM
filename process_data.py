import pandas as pd
import numpy as np
import re
import string
from nltk.tokenize import word_tokenize 
import itertools


def clean_sentence(sentence):
    
    """
    clean each sentence by:
        removing non alpha-numeric characters,
        converting text to lowercase,
        standardising the spacing to unit spacing
        removing all punctuation.
        
    :param sentence: str
    """
        
    sentence = re.sub(r'[^A-Za-z0-9#]', ' ', str(sentence))
    sentence = sentence.lower()
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    sentence = " ".join(re.findall(r"[\w']+", sentence))
    return sentence



def build_vocab_index(vocab):
    
    """
    build a dictionary that assigns each word in the vocabulary to a unique index.
    
    :param vocab: list
    """
    
    index_vocab = {}
    index = 0
    for word in vocab:
        index_vocab[word] = index
        index += 1
    
    return index_vocab


def return_indices_for_sentence(sentence_word_column, index_vocab_):
    
    """
    map the words in each sentence to their unique indices.
    
    :param sentence_word_list: List[str]
    :param index_vocab_: dict[str]-> int
    """
    
    return_indices = []
    for sentence in sentence_word_column:
        sentence_list = []
        for word in sentence:
            sentence_list.append(index_vocab_[word])
        return_indices.append(sentence_list)
    
    return return_indices

def process_text_column(column):
    
    """
    preprocess each sentence to be of the desired form by:
        cleaning each sentence,
        tokenizing the sentence.
        
    :param column: pd.Series[str]
    """
    
    clean_column = column.apply(clean_sentence)
    clean_tokenized_column = clean_column.apply(word_tokenize)
    
    return clean_tokenized_column

def build_vocab(column1, column2):
    
    """
    build the vocabulary based on the words contained in column1 and column2.
    
    :param column1: pd.Series[str]
    :param column2: pd.Series[str]
    """
    
    all_sentences = column1 + column2
    all_sentences_flattened = list(itertools.chain.from_iterable(all_sentences))
    unique_words = list(set(all_sentences_flattened))
    
    return unique_words

def process_df(maxlen):
    
    """
    main function to be run. This function processes the dataframe and returns 5 columns in the following order:
        clean_X1_indices: pd.Series(List[int]) containing the integer mappings of the words contained in each sentence of column X1.  
        clean_X2_indices: pd.Series(List[int])containing the integer mappings of the words contained in each sentence of column X2.  
        X1_sentence_length: pd.Series(List[int]) containing the length of each sentence in column X1. Any sentence with length > maxlen will be (pre)-truncated.
        X2_sentence_length: pd.Series(List[int]) containing the length of each sentence in column X2. Any sentence with length > maxlen will be (pre)-truncated.
        y: pd.Series[int] containing values \in {0,1}, indicating whether each sentence pair is semantically similar. 
        
    :param maxlen: int
    """
    
    df = pd.read_csv("questions.csv")
    X1 = df["question1"]
    X2 = df["question2"]
    y = df["is_duplicate"]
    
    clean_X1 = process_text_column(X1)
    clean_X2 = process_text_column(X2)
    
    vocab = build_vocab(clean_X1, clean_X2)
    vocab_index_dict = build_vocab_index(vocab)
    
    clean_X1_indices = return_indices_for_sentence(clean_X1, vocab_index_dict)
    clean_X2_indices = return_indices_for_sentence(clean_X2, vocab_index_dict)
    
    X1_sentence_length = pd.Series(clean_X1_indices).apply(lambda x: len(x))
    X2_sentence_length = pd.Series(clean_X2_indices).apply(lambda x: len(x))
    
    # truncate sentences with more than the specified maxlen
    X1_over_maxlen_indices = np.argwhere(np.array(X1_sentence_length) > maxlen).reshape(-1)
    X2_over_maxlen_indices = np.argwhere(np.array(X2_sentence_length) > maxlen).reshape(-1)
    
    for sentence_index in X1_over_maxlen_indices:
        clean_X1_indices[sentence_index] = clean_X1_indices[sentence_index][-maxlen:]
        X1_sentence_length[sentence_index] = maxlen
    
    for sentence_index in X2_over_maxlen_indices:
        clean_X2_indices[sentence_index] = clean_X2_indices[sentence_index][-maxlen:]
        X2_sentence_length[sentence_index] = maxlen
        
    return pd.Series(clean_X1_indices), pd.Series(clean_X2_indices), pd.Series(X1_sentence_length),\
         pd.Series(X2_sentence_length), y


        
    
