# load libraries
from tqdm import tqdm
tqdm.pandas()
import pandas as pd
import json,os,gc
from time import time
from copy import deepcopy
import nltk
from nltk import sent_tokenize,word_tokenize
from nltk.corpus import wordnet,stopwords
from nltk.stem import WordNetLemmatizer
stopwords = set(stopwords.words('english'))
# nltk.download('all')
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import os,re,multiprocessing,joblib
from multiprocessing import Pool
from collections import defaultdict
from transformers import pipeline


# load dataset
path1 = 'G:/rauf/STEPBYSTEP/Data/Covid-19/kgarrett-covid-19-open-research-dataset/biorxiv_medrxiv/'
path2 = 'G:/rauf/STEPBYSTEP/Data/Covid-19/kgarrett-covid-19-open-research-dataset/comm_use_subset/'
path3 = 'G:/rauf/STEPBYSTEP/Data/Covid-19/kgarrett-covid-19-open-research-dataset/pmc_custom_license/'
path4 = 'G:/rauf/STEPBYSTEP/Data/Covid-19/kgarrett-covid-19-open-research-dataset/noncomm_use_subset/'
paths = [path1,path2,path3,path4]
file_names = []
for path in paths:
  temp_file_names = os.listdir(path)
  file_names.extend([path+file_name for file_name in temp_file_names])
#_>print(len(file_names)) #here we got 13202 articles as jsonfile


# to load json file
def file_content(file_path):
  abstract='';body_text = '';error_count = 0
  if os.path.splitext(file_path)[1]=='.json':
    f = open(file_path)
    f_json = json.load(f)
    try:
      abstract = f_json['abstract'][0]['text']
    except:
      error_count+=1
    for i in f_json['body_text']:
      try:
        body_text= body_text+' '+i['text']
      except:
        error_count+=1
    body_text = body_text.strip()
    f.close()
    return body_text,abstract,error_count
  else:
    return body_text,abstract,error_count


#
## Storing article and related information in data-frame
df = pd.DataFrame({'file_name':[],'body':[],'abstract':[],'error_count':[]})
df['file_name'] = file_names
df['article_no'] = list(range(df.shape[0]))
#_>for ind,info in tqdm(df.iterrows(),total=df.shape[0]):  df.loc[ind,'body'],df.loc[ind,'abstract'],df.loc[ind,'error_count'] = \
  #_>file_content(file_path=info['file_name'])
#_>print(df.head())


# preprocessing functions
corpus_file = 'G:/rauf/STEPBYSTEP/Projects/NLP/Question Answering/Question Answering Covid-19/corpus.txt'
sent_dict_file = 'sent.joblib.compressed'
word_sent_no_dict_file = 'word_sent_no.joblib.compressed'
orig_word_sent_no_dict_file = 'orig_word_sent_no.joblib.compressed'
stopword_file = 'G:/rauf/STEPBYSTEP/Projects/NLP/Question Answering/Question Answering Covid-19/stopwords.txt'


## Lemmatization function
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)
# 1. Init Lemmatizer
lemmatizer = WordNetLemmatizer()

def get_lemmatize(sent):
  return " ".join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in word_tokenize(sent)])

def parallelize_dataframe(df, func, num_partitions, num_cores):
  df_split = np.array_split(df, num_partitions)
  pool = Pool(num_cores)
  df = pd.concat(pool.map(func, df_split))
  pool.close()
  pool.join()
  return df

def fn_lemmatize(data):
  for ind,info in tqdm(data.iterrows(),total=data.shape[0]):
    data.loc[ind,'sentence_lemmatized'] = get_lemmatize(sent = info['sentence'])
  return data

## removing stopwords
def words(text): return re.findall(r'\w+', text.lower())
stopwords = list(set(words(open(stopword_file).read())))

def remove_stopwords(sent):
  ## case conversion - lower case
  word_tokens = words(text=sent)
  #sent = sent.lower()
  #word_tokens = word_tokenize(sent)
  ## removing stopwords
  filtered_sentence = " ".join([w for w in word_tokens if not w in stopwords])
  ## removing digits
  filtered_sentence = re.sub(r'\d+','',filtered_sentence)
  ## removing multiple space
  filtered_sentence = words(text = filtered_sentence)
  return " ".join(filtered_sentence)

def fn_stopword(data):
  for ind,info in tqdm(data.iterrows(),total=data.shape[0]):
    sent = info['sentence_lemmatized']
    data.loc[ind,'sentence_lemma_stop'] = remove_stopwords(sent)
  return data

def fn_stopword_orig(data):
  for ind,info in tqdm(data.iterrows(),total=data.shape[0]):
    sent = info['sentence']
    data.loc[ind,'sentence_stop'] = remove_stopwords(sent)
  return data


# creating sentence dictionary
df['article'] = df['body']+' '+df['abstract']
df['article'].fillna('',inplace=True)
article_no_sent_dict = dict()

for ind,info in tqdm(df.iterrows(),total=df.shape[0]):
  article_no_sent_dict[info['article_no']] = sent_tokenize(str(info['article']))
article_no_list = list();sent_list = list()
df_sent = pd.DataFrame({'article_id':[],'sentence':[]})

for i in tqdm(article_no_sent_dict,total=len(article_no_sent_dict)):
  article_no_list.extend([i]*len(article_no_sent_dict[i]))
  sent_list.extend(article_no_sent_dict[i])

df_sent['article_id'] = article_no_list ; df_sent['sentence'] = sent_list
df_sent['sent_no'] = list(range(df_sent.shape[0]))


# sentence level dictionary
sent_dict = dict()
for ind,info in tqdm(df_sent.iterrows(),total=df_sent.shape[0]):
  sent_dict[info['sent_no']] = info['sentence']
sent_dict[-1] = 'NULL'
sent_dict_file = 'sent.joblib.compressed'
joblib.dump(sent_dict,sent_dict_file, compress=True)


#
## lemmatization over sentence
df1 = deepcopy(df_sent)
df1 = parallelize_dataframe(df=df1, func=fn_lemmatize, num_partitions=27, num_cores=27)
## removing stopword from lemmatized sentence
df1 = parallelize_dataframe(df=df1, func=fn_stopword, num_partitions=30, num_cores=35)
## saving inverse dictionary on lemmatized sentence i.e. word and sentence no
word_sent_no_dict = defaultdict(list)
for ind,info in tqdm(df1.iterrows(),total=df1.shape[0]):
  sent_words = words(info['sentence_lemma_stop'])
  for w in sent_words:
    word_sent_no_dict[w].append(info['sent_no'])
joblib.dump(word_sent_no_dict,word_sent_no_dict_file, compress=True)


#
## saving inverse dictionary on original sentence i.e. word and sentence no
df1 = deepcopy(df_sent)
df1 = parallelize_dataframe(df=df1, func=fn_stopword_orig, num_partitions=35, num_cores=35)
orig_word_sent_no_dict = defaultdict(list)
for ind,info in tqdm(df1.iterrows(),total=df1.shape[0]):
  sent_words = words(info['sentence_stop'])
  for w in sent_words:
    orig_word_sent_no_dict[w].append(info['sent_no'])
joblib.dump(orig_word_sent_no_dict,orig_word_sent_no_dict_file, compress=True)


#
## Corpus - for spelling correction model
outF = open(corpus_file, "w")
for line in tqdm(df_sent['sentence'],total=df_sent.shape[0]):
 # write line to output file
 outF.write(line)
 outF.write("\n")
outF.close()


# Use BERT Model
nlp = pipeline('question-answering',model = 'bert-large-cased-whole-word-masking-finetuned-squad')
query_sample = "How to prevent Corona ?"
relevant_sentence = "When asked why they were wearing masks, several students answered that they were "preventing corona"".
nlp(question = query_sample, context = relevant_sentence)