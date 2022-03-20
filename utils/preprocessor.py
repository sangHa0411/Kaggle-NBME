import re
from nltk.tokenize import sent_tokenize

class Preprocessor :
    def __init__(self, ) :
        pass

    def clean_spaces(self, txt):
        txt = re.sub('\n', ' ', txt)
        txt = re.sub('\t', ' ', txt)
        txt = re.sub('\r', ' ', txt)
        return txt

    def split(self, doc, sen_list) :
        sens = [sen_list[0]]
        prev_end = len(sen_list[0])

        gaps = []
        for sen in sen_list[1:] :
            start_id = doc.find(sen, prev_end)
            end_id = start_id + len(sen)

            gap = doc[prev_end:start_id]
            gaps.append(gap)

            if sen.isupper() and len(sen) > 5 :
                sen = sen[0] + sen[1:].lower()
            sens.append(sen)
            prev_end = end_id

        gaps.append(doc[end_id:])
        return sens, gaps        

    def merge(self, sens, gaps) :
        docs = []
        for i in range(len(sens)) :
            docs.append(sens[i] + gaps[i])
        return ''.join(docs)

    def __call__(self, dataset) :
        doc_list = dataset['history']
        batch_size = len(doc_list)

        docs = []
        for i in range(batch_size) :
            prev_doc = doc_list[i]
            prev_doc = self.clean_spaces(prev_doc)
            sen_list = sent_tokenize(prev_doc)

            if len(sen_list) > 1 :
                sens, gaps = self.split(prev_doc, sen_list)
                after_doc = self.merge(sens, gaps)

                assert len(prev_doc) == len(after_doc)
                docs.append(after_doc)
            else :
                docs.append(prev_doc)
        dataset['history'] = docs
        return dataset