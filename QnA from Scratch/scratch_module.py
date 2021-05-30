# load libraries
import pandas as pd
from gensim.models import Word2Vec
from gensim.models import FastText
import gensim


# create function to convert json file to pd dataframe
def json_to_df(json_file):
    arrayForDF=[]
    for current_subject in json_file['data']:
        subject = current_subject['title']
        for current_context in current_subject['paragraphs']:
            context = current_context['context']
            for current_question in current_context['qas']:
                question = current_question['question']
                if (len(question)>2):
                    is_impossible = current_question['is_impossible']
                    if (is_impossible==False):
                        for answer in current_question['answers']:
                            answer_text = answers['text']
                            answer_start = answer['answer_start']

                            record = {
                                "answer_text" : answer_text,
                                "answer_start" : answer_start,
                                "question" : question,
                                "context" : context,
                                "subject" : subject
                                }
                            arrayForDF.append(record)

    df = pd.DataFrame(arrayForDF)
    return df


# we shpuld train word embedding model on our context
# loading context
    