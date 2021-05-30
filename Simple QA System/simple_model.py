# load libraries
import pandas as pd


# load data
data = pd.read_csv('qa.csv')


# this function is used to get printable results
def getResults(questions, fn):
    def getResult(q):
        answer, score, prediction = fn(q)
        return [q, prediction, answer, score]
    return pd.DataFrame(list(map(getResult, questions)), columns=["Q", "Prediction", "A", "Score"])


test_data = [
    "What is the population of Egypt?",
    "What is the poulation of egypt",
    "How long is a leopard's tail?",
    "Do you know the length of leopard's tail?",
    "When polar bears can be invisible?",
    "Can I see arctic animals?",
    "some city in Finland"
]


#
def getNaiveAnswer(q):
    # regex helps to pass some punctuation signs
    row = data.loc[data['Question'].str.contains(re.sub(r"[^\w'\s)]+", "", q),case=False)]
    if len(row) > 0:
        return row["Answer"].values[0], 1, row["Question"].values[0]
    return "Sorry, I didn't get you.", 0, ""

getResults(test_data, getNaiveAnswer) # here our model find only two questions on test data # the reason for this poor result is we remove punctuations


# to get better result we use aproximate string matching method
# such as Levinshtein method
from Levenshtein import ratio
def getApproximateAnswer(q):
    max_score = 0
    answer = ""
    prediction = ""
    for idx, row in data.iterrows():
        score = ratio(row["Question"], q)
        if score >= 0.9: # I'm sure, stop here
            return row["Answer"], score, row["Question"]
        elif score > max_score: # I'm unsure, continue
            max_score = score
            answer = row["Answer"]
            prediction = row["Question"]
    if max_score > 0.8:
        return answer, max_score, prediction
    return "Sorry, I didn't get you.", max_score, prediction
getResults(test_data, getApproximateAnswer) # here is better result we got three answer


# to improve gain result for better side we should drop max_score to 0.3 from 0.8
from Levenshtein import ratio
def getApproximateAnswer2(q):
    max_score = 0
    answer = ""
    prediction = ""
    for idx, row in data.iterrows():
        score = ratio(row["Question"], q)
        if score >= 0.9: # I'm sure, stop here
            return row["Answer"], score, row["Question"]
        elif score > max_score: # I'm unsure, continue
            max_score = score
            answer = row["Answer"]
            prediction = row["Question"]
    if max_score > 0.3: # threshold is lowered
        return answer, max_score, prediction
    return "Sorry, I didn't get you.", max_score, prediction
getResults(test_data, getApproximateAnswer2) # finally we get answer for all of our questions


#