from sklearn.feature_extraction.text import CountVectorizer
import pickle
import string
import nltk
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk import tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

words_to_exclude = [
    "...", ".."
]
tags_to_remove=["NNP", "VBG", "VBN", "CD"]


def word_embedding_processing(data):

    count_vectorizer = pickle.load(open("models/count_vectorizer.h5", 'rb'))
    X_embed  = count_vectorizer.transform([data])
    return X_embed


def clean_textism(sentence):
    neo_sentence = []
    for word in sentence:
        if word == 'u':
            neo_sentence.append('you')
        elif word == 'r':
            neo_sentence.append('are')
        elif word == 'ur':
            neo_sentence.append('your')
        elif word == 'some1':
            neo_sentence.append('someone')
        elif word == 'yrs':
            neo_sentence.append('years')
        elif word == 'hrs':
            neo_sentence.append('hours')
        elif word == 'mins':
            neo_sentence.append('minutes')
        elif word == 'secs':
            neo_sentence.append('seconds')
        elif word == 'pls' or word == 'plz':
            neo_sentence.append('please')
        elif word == '2morow':
            neo_sentence.append('tomorrow')
        elif word == '2day':
            neo_sentence.append('today')
        elif word == '2nite':
            neo_sentence.append('tonight')
        elif word == '4got' or word == '4gotten':
            neo_sentence.append('forget')
        elif word == 'amp' or word == 'quot' or word == 'lt' or word == 'gt' or word == '½25':
            neo_sentence.append('')
        else:
            neo_sentence.append(word)
    return neo_sentence


def get_wordnet_pos(word):
    """Return the corresponding character for a word use in the lemmatization
    
    Parameters:
    word (str): a word
    
    Returns:
    str: the corresponding character
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def natural_language_processing(data):
    res = ""

    X = data.split()
    X_wo_arobas = [x for x in X if not x.startswith("@")]
    X_new = [x for x in X_wo_arobas if not x.startswith("http")]
    temp_res = ' '.join(X_new)

    # building stopwords list
    stopW = stopwords.words('english')
    stopW.extend(string.punctuation)

    # normalisation
    temp_res = temp_res.lower()
    # tokenization
    tk = tokenize.TweetTokenizer(reduce_len=True)
    temp_res = tk.tokenize(temp_res)

    # clean the sms language to usefull langage
    temp_res = clean_textism(temp_res)

    # remove stopwords
    temp_res = [word for word in temp_res if word not in stopW]

    # lemmatization
    lemmatizer = WordNetLemmatizer()
    temp_res = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in temp_res]

    temp_res = pos_tag(temp_res)
    temp_res = [x[0] for x in temp_res if x[1] not in tags_to_remove]

    temp_res = [x for x in temp_res if x not in words_to_exclude]

    temp_res = ' '.join([x for x in temp_res])

    res = word_embedding_processing(temp_res)

    return res



