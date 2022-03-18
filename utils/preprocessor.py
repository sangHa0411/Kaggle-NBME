import re
from nltk.tokenize import sent_tokenize

def standardize(txt) :
    sen_list = sent_tokenize(txt)
    sens = []
    for sen in sen_list :
        if sen.isupper() :
            sen = sen[0].upper() + sen[1:].lower()
        sens.append(sen)
    return ' '.join(sens)

def clean_spaces(txt):
    txt = re.sub('\n', ' ', txt)
    txt = re.sub('\t', ' ', txt)
    txt = re.sub('\r', ' ', txt)
    return txt

def preprocess(dataset) :
    doc_list = dataset['history']
    batch_size = len(doc_list)

    docs = []
    for i in range(batch_size) :
        doc = doc_list[i]
        doc = clean_spaces(doc)
        doc = standardize(doc)
        docs.append(doc)

    dataset['history'] = docs
    return dataset
