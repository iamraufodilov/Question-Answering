# load libraries
import wikipedia as wiki
import pprint as pp
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from collections import OrderedDict


# load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2") 
model = AutoModelForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")


# get the question and context from wikipedia
question = 'What is the wingspan of an albatross?'

results = wiki.search(question)
#_>print("Wikipedia search results for our question:\n")
#_>pp.pprint(results)

page = wiki.page(results[0])
text = page.content
#_>print(f"\nThe {results[0]} Wikipedia article contains {len(text)} characters.")


# tokenize question and context
inputs = tokenizer.encode_plus(question, text, return_tensors='pt')
#_>print(f"This translates into {len(inputs['input_ids'][0])} tokens.") 
# as you can see here the amount of tokens are more 8890 which model can be feeded
# maximum 512 tokens in case we should divide inputs to several chunks


# time to chunk!

# identify question tokens (token_type_ids = 0)
qmask = inputs['token_type_ids'].lt(1)
qt = torch.masked_select(inputs['input_ids'], qmask)
print(f"The question consists of {qt.size()[0]} tokens.")

chunk_size = model.config.max_position_embeddings - qt.size()[0] - 1 # the "-1" accounts for
# having to add a [SEP] token to the end of each chunk
print(f"Each chunk will contain {chunk_size - 2} tokens of the Wikipedia article.")

# create a dict of dicts; each sub-dict mimics the structure of pre-chunked model input
chunked_input = OrderedDict()
for k,v in inputs.items():
    q = torch.masked_select(v, qmask)
    c = torch.masked_select(v, ~qmask)
    chunks = torch.split(c, chunk_size)

    for i, chunk in enumerate(chunks):
        if i not in chunked_input:
            chunked_input[i] = {}

        thing = torch.cat((q, chunk))
        if i != len(chunks)-1:
            if k == 'input_ids':
                thing = torch.cat((thing, torch.tensor([102])))
            else:
                thing = torch.cat((thing, torch.tensor([1])))

        chunked_input[i][k] = torch.unsqueeze(thing, dim=0) # here we go now we got 12 tokens of questions and their respective conr=text with 497 tokens


# lets show all chunks
for i in range(len(chunked_input.keys())):
    print(f"Number of tokens in chunk {i}: {len(chunked_input[i]['input_ids'].tolist()[0])}")


# now we can feed this chunks to our model
# but remember only one chunks may contain the answer rest of them may not
# below code to find that chunk nd answer
def convert_ids_to_string(tokenizer, input_ids):
    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids))

answer = ''

# now we iterate over our chunks, looking for the best answer from each chunk
for _, chunk in chunked_input.items():
    answer_start_scores, answer_end_scores = model(**chunk)

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    ans = convert_ids_to_string(tokenizer, chunk['input_ids'][0][answer_start:answer_end])
    
    # if the ans == [CLS] then the model did not find a real answer in this chunk
    if ans != '[CLS]':
        answer += ans + " / "
        
print(answer) 


# time to test
questions = [
    'When was Barack Obama born?',
    'Why is the sky blue?',
    'How many sides does a pentagon have?'
]

reader = DocumentReader("deepset/bert-base-cased-squad2") 

# if you trained your own model using the training cell earlier, you can access it with this:
#reader = DocumentReader("./models/bert/bbu_squad2")

for question in questions:
    print(f"Question: {question}")
    results = wiki.search(question)

    page = wiki.page(results[0])
    print(f"Top wiki result: {page}")

    text = page.content

    reader.tokenize(question, text)
    print(f"Answer: {reader.get_answer()}")
    print() # here we go we got answer



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# CONCLUSION
'''
in this repo we load fine tuned model, such as bert
and try to use it with large context such as wiki article
in case challenge how to feed large data to model
solution is we divide rticle context to several chunks which is below 512
'''