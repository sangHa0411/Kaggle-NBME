import re

def process_features(txt):
    txt = re.sub('I-year', '1-year', txt)
    txt = re.sub('-OR-', " or ", txt)
    txt = re.sub('-', ' ', txt)
    return txt

def clean_spaces(txt):
    txt = re.sub('\n', ' ', txt)
    txt = re.sub('\t', ' ', txt)
    txt = re.sub('\r', ' ', txt)
    return txt