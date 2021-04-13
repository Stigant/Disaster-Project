import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
from urllib.parse import urlparse
import phonenumbers as pn

def load_data(database_filepath):
    """Load pandas dataframe from SQL database"""
    engine = create_engine('sqlite:///'+database_filepath)
    df=pd.read_sql_table('DisasterTable',engine).set_index('id')
    return df

def get_wordnet_pos(tag):
    tag=tag[0].upper()
    if tag == 'J':
        return wordnet.ADJ
    elif tag == 'V':
        return wordnet.VERB
    elif tag == 'N':
        return wordnet.NOUN
    elif tag == 'R':
        return wordnet.ADV
    else:
        return wordnet.NOUN

##Patterns and functions for tokenizer
stopWords=stopwords.words('english')
lemmatizer = WordNetLemmatizer()


#Borrowed from https://daringfireball.net/2010/07/improved_regex_for_matching_urls
url_pattern_1 = r"(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'\".,<>?«»“”‘’]))"

url_pattern_2 = r"[a-z0-9.\-]+[.][a-z]{2,4}" #

phone_pattern=r"([\+\d*]?(\(?\d+\)?[\s|-]*)+)"

locations=[None, 'GB', 'USA']

symbols= {'\$': 'USD',
 '\₽': 'RUB',
 '\£': 'GBP',
 '\€': 'EUR',
 '\¥': 'CNY'}

currency_markers= list(symbols.keys())+list(symbols.values())
marker_pattern= '('+'|'.join(currency_markers)+')'
curr_pattern=r""+ marker_pattern+"\s*(\d+[\s|\.|,]*)+"

def reduce_if(char, count):
    if char.isalpha():
        return char*min(2,count)
    else:
        return char*count

def parse_currency(curr):
    symb=curr[0]
    try:
        code=symbols['\\'+symb]
        curr=curr[1:]
    except:
        code=curr[:2]
        curr=curr[3:]
    digits=re.sub('[\s|\.|,]+', '', curr)
    return [code, code+digits, '$']

def parse_phone_number(num, loc=None):
    try:
        dat = pn.parse(num, loc)
        if pn.is_valid_number(dat):
            return str(dat.national_number)
    except:
        return None

def extract_url_data(url):
    dat=urlparse(url)
    if dat.hostname:
        return [x for x in [dat.hostname.strip('www.'), dat.path.strip('/')] if x]
    else:
        dat=urlparse('//'+url)
        return [x for x in [dat.hostname.strip('www.'), dat.path.strip('/')] if x]

def tokenize(text):
    """Clean and tokenize text, then lemmatize"""
    text = text.lower()

    ##urls
    text=re.sub(r"\[|\]", ' ', text) #They break the regex for some reason

    matcher= re.compile(url_pattern_1)
    urls=[match.group() for match in matcher.finditer(text)]
    for url in urls:
        text=text.replace(url, ' ', 1)

    matcher= re.compile(url_pattern_2)
    maybe_urls=[match.group() for match in matcher.finditer(text)]
    for url in maybe_urls:
        text=text.replace(url, ' ')

    url_tokens=sum([extract_url_data(url) for url in (urls+maybe_urls)], [])


    ##Phone Numbers
    maybe_phones=[x[0] for x in re.findall(phone_pattern, text)]
    phone_tokens=[]
    for num in maybe_phones:
        for loc in locations:
            digits=''.join(re.findall(r'\d+', num))
            nat_num=parse_phone_number(num, loc=loc)
            if nat_num:
                phone_tokens.append(nat_num)
                text=text.replace(num, '', 1)
                break
    phone_count='##'+str(len(phone_tokens))
    phone_tokens.append(phone_count)


    ##Currency
    matcher= re.compile(curr_pattern)
    matches=[match.group() for match in matcher.finditer(text)]
    currency_tokens=[]
    for match in matches:
        text=text.replace(match, '', 1)
        currency_tokens+=parse_currency(match)

    ##Remove punctuation and duplicate characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    matcher= re.compile(r'(.)\1*')
    matches=[match.group() for match in matcher.finditer(text)]
    text=''.join([reduce_if(x[0],len(x)) for x in matches])
    word_tokens=word_tokenize(text)
    ##Lemmatise
    word_tokens = [lemmatizer.lemmatize(word,get_wordnet_pos(tag)) for word,tag in pos_tag(word_tokens) if word not in stopWords]
    #word_tokens = [lemmatizer.lemmatize(word) for word in word_tokens if word not in stopWords]
    return url_tokens+phone_tokens+word_tokens+currency_tokens
    #return word_tokens

class const(object):
    def __init__(self, a):
        self.a = a
    def __call__(self, b):
        return self.a
def keep_message(X):
    """Keep only message column"""
    return  X['message']
def keep_genres(X):
    """Keep genre columns"""
    return X[['genre_social', 'genre_news']]
def thresh_fun(z):
    """Calculate custom threshold as 2*mean"""
    return 2*z.mean()
def no_entries_in(X):
    """See if sparse array has entries in columns 0:-2"""
    return np.diff(X[:,:-2].indptr) == 0
