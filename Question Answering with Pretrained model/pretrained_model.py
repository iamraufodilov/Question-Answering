# load libraries
from allennlp.predictors.predictor import Predictor


# load pretrained model
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bidaf-model-2020.03.19.tar.gz")


# load the text
mytext = 'G:/rauf/STEPBYSTEP/Projects/NLP/Question Answering/Question Answering with Pretrained model/passage.txt'


# give question and get answer
result=predictor.predict(
  passage=passage,
  question= "When Artificial intelligence was founded as an academic discipline?" #Artificial intelligence was founded as an academic discipline in 1955
)
result['best_span_str'] #here we go we get answer