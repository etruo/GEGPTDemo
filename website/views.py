from django.shortcuts import render
from django.http import HttpResponse
from .prompt_completions_functions import *
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt 
import json

# Create your views here.
# this jjawn is the request handler! , not actual "views"


def say_meow(request):
    ##return HttpResponse('meoooooow')


    # debug example
    x = 2

    # example of using with parameters between .html file, rather than just straight text
    
    return render(request, "meow.html", { "name" : "daddy"})

# debugging process:
# 0. breakpoint, run
# 1. open different server host thing
# 2. then go to the url with the specific website/view

# different debug toolbar: 
# follow this: https://django-debug-toolbar.readthedocs.io/en/latest/installation.html

#import packages
import json
import openai 
import pandas as pd
import numpy as np
import pickle
import os
from transformers import GPT2TokenizerFast
from typing import List
import wordfreq



#graham1118@gmail.com Key (AT CAPACITY)
#openai.api_key = "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"
#gks.kenan@gmail.com Key
openai.api_key = "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"


#We need a separate embedding modle for the data and the user's query
MODEL = "curie" #Can be "ada", "davinci", etc.
DOC_EMBEDDINGS_MODEL = f"text-search-{MODEL}-doc-001"
QUERY_EMBEDDINGS_MODEL = f"text-search-{MODEL}-query-001"
COMPLETIONS_MODEL = "text-davinci-003"


#Embeds text
def get_embedding(text: str, model: str) -> List[float]:
    result = openai.Embedding.create(
      model=model,
      input=text)
    return result["data"][0]["embedding"]

#Embed the user's query into a numerical vector for comparison with the wikipedia embeddings
def get_query_embedding(text: str) -> List[float]:
    return get_embedding(text, QUERY_EMBEDDINGS_MODEL)

#distance metric for comparing query and context
def vector_similarity(x, y):
    return np.dot(np.array(x), np.array(y))



#   DESCRIPTION: Searches through each wiki paragraph in doc_embeddings and orders them based on closeness to the query_embedding
#   PARAMETERS:
#      - query: User's Query (string)
#      - doc_embeddings: the dictionary imported in the 1st code cell
#   FUNCTIONS USED:
#      - get_query_embedding()
#      - vector_similarity()

def order_document_sections_by_query_similarity(query, doc_embeddings):
    
    #Turning the query into a numerical vector (An "embedding")
    query_embedding = get_query_embedding(query)
    
    #Super long list concatenation
    document_similarities = sorted([(vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in doc_embeddings.items()], reverse=True)
    
    return document_similarities #dictionary of [distance between vectors: index of vector in dataframe]

import pinecone
import boto3
from smart_open import smart_open

PINECONE_API_KEY = '3662b65e-7d4c-441d-8b03-3e0b5f2070e7'
PINECONE_ENV = 'us-east-1-aws'

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index("index-1") 

AWS_ACCESS_KEY_ID = "AKIASMIRSZ4UMUC3J64S"
AWS_SECRET_KEY = "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"
client = boto3.client(service_name='s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_KEY)

#Varibles used by construct_prompt()
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2") #we need a tokenizer, this works for gpt3/4
SEPARATOR = "\n* " #Need a separator to separate context and query
separator_len = len(tokenizer.tokenize(SEPARATOR))



#   DESCRIPTION: Adds a prompt-engineered header, as well as wikipedia paragraphs to the user's original question
#   PARAMETERS:
#      - question: User's Query (string)
#      - answer_len: How long (in # of tokens) you want the answer to be
#      - Mode: Can be "Q&A" or "Instructions". Lets the model know how it should answer the prompt
#   FUNCTIONS USED:
#      - order_document_sections_by_query_similarity()

def construct_prompt(question, answer_len, custom_instructions, mode = "Q&A"):

  MAX_SECTION_LEN = 4097 - answer_len #GPT-3 has a token limit of 4097 tokens, shared between the prompt and the response
  
  
  #Step 1) Order articles based on similarity to prompt
  query_embedding = get_query_embedding(question)
  matches = index.query(vector=query_embedding, top_k=5)


  #Step 2) Add article sections to the prompt until there are no more tokens left for the prompt
  chosen_sections_len = 0
  chosen_sections_content = []
  chosen_sections_objects = []

  for vec in matches['matches']:
    id = vec['id'] #id is "<csv_number>-<index_in_csv>"

    id_components = id.split("-")
    csv_num = id_components[0]
    csv_index = int(id_components[1])

    path = 's3://{}:{}@foundersxarticles/article_df_{}.csv'.format(AWS_ACCESS_KEY_ID, AWS_SECRET_KEY, csv_num)
    doc_df = pd.read_csv(smart_open(path))

    #Retreive document section from index
    document_section = doc_df.loc[csv_index] #columns are:    Title, DOI, Paragraph
        
    num_tokens = len(tokenizer.tokenize(document_section.Paragraph))

    chosen_sections_len += num_tokens + separator_len
    chosen_sections_objects.append(document_section)
    chosen_sections_content.append(SEPARATOR + document_section.Paragraph.replace("\n", " "))
    
    if chosen_sections_len > MAX_SECTION_LEN:
      break
        
    
  #Construct Prompt with a header, the context, and the question
  #Using Q: and A: lets the model know we want Q&A behavior

  if mode == "General":
    header = ""

  if mode == "Q&A":
    header = """Answer the question below as truthfully as possible using the 
    provided context, and if the answer is not contained 
    within the text below, say "I don't know."\n\nContext:\n"""

  if mode == "Instructions":
    header = """You are a professor of biology and a student in your lab asks you how to do a certain task. 
    Using the factual context below, answer their question using a list of instructional steps which explains how to complete the task.
    If the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""

  if mode == "Workflow":
    header = """
    ###INSTRUCTIONS###
    You are a professor of biology and a student in your lab asks you how to do a certain task. 
    Using the factual context below, answer their question in the following format:

    ###FORMAT###
    <describe_task_overview_in_a_complete_sentence>

    1) <step_1>
    2) <step_2>
    ...
    n) <step_n>

    Necessary Equipment: <comma_separated_list_of_machinery>
    Necessary Materials: <comma_separated_list_of_substances> 
    Estimated time to complete: <estimated_time_to_complete_task>

    
    ###CONTEXT###\n"""

  if custom_instructions != "":
    header = custom_instructions

  #Step 3) Put it all together!
  final_prompt = header + "".join(chosen_sections_content) + "\n\n Q: " + question + "\n A:"
  return final_prompt, chosen_sections_objects

#These parameters are passed into the openai.Completions.create() function
COMPLETIONS_API_PARAMS = {
    "temperature": 0.0, #let temp be 0 to minimize creativity, maximize factualness
    "max_tokens": 700,
    "model": COMPLETIONS_MODEL,
    }

VOCAB_API_PARAMS = {
    "temperature": 0.0, #let temp be 0 to minimize creativity, maximize factualness
    "max_tokens": 200,
    "model": COMPLETIONS_MODEL,
    }



#FOR RESEARCH ARTICLES
def add_sources(doc_objects):    
    print(f"\n\n\nI found {len(doc_objects)} article(s) to inform my answer:")
    for i in range(len(doc_objects)):
        print(str(i+1) + ") " + doc_objects[i].Title + " (" + doc_objects[i].DOI + ")" + "\n" + doc_objects[i].Paragraph)
        print("\n\n")
    

#Strips punctuation from words. Used in add_vocab
def strip_words(word):
    return word.strip('{[]}\"\'.,();:-\n')

def add_vocab(response):
    words_defined = []
    
    response = str(response)
    print("\n\n\nRelevant Vocabulary:")
    words = response.split(' ')
    words = map(strip_words, words)
    threshold_to_define = 3.2 #logarithmic frequency of a word in the english language
    for word in words:
        if wordfreq.zipf_frequency(word, "en", wordlist = 'best') <= threshold_to_define and word not in words_defined:
            word_to_define = "please define the word \"" + word + "\" in 1 sentence."
            definition = openai.Completion.create(prompt=word_to_define, **VOCAB_API_PARAMS)
            definition = definition["choices"][0]["text"].strip(" \n")
            print("\n\t*" + word + ": " + definition)
            words_defined.append(word)


import sys

#   DESCRIPTION: Returns a response given a question
#   PARAMETERS:
#      - query: User's Query (string)
#      - answer_len: How long (in # of tokens) you want the answer to be
#      - mode: Can be "Q&A" or "Instructions". Lets the model know how it should answer the prompt
#      - vocab: If True, defines notable vocab in the response
#      - custom_instructions: If provided, this replaces the "header" in construct_prompt
#   FUNCTIONS USED:
#      - construct_prompt()

def ask(query, answer_len, mode = "Q&A", vocab = True, custom_instructions = ""):
   
    print(query, answer_len, mode, vocab, custom_instructions)
    prompt, doc_sections = construct_prompt(query, answer_len, custom_instructions, mode=mode)
    
    response = openai.Completion.create(prompt=prompt, stream = False, **COMPLETIONS_API_PARAMS)
    #for response in openai.Completion.create(prompt=prompt, stream = False, **COMPLETIONS_API_PARAMS):
     # sys.stdout.write(response.choices[0].text)
      #sys.stdout.flush()
    

    add_sources(doc_sections) #The urls will always be printed, but should be hidden, user can specify if they want it to show
    if vocab: add_vocab(response)
    return response.choices[0].text





@csrf_exempt
def generate_response(request):
    print(request.method) #prints statement to detect POST or GET requests
    if request.method == 'POST':
        data = request.POST.dict()
        
        user_query = data['query']
        response_length = int(data['length'])
        define_vocab = bool(data['vocab'])
        custom_instructions = data['instructions']
        mode = data['mode']

        # replace with resposnse generations
        GPTresponse = ask(user_query, response_length, mode, define_vocab, custom_instructions)
        test = 'yo!'
        print("HEYYYYYYYYYYYYY")
        return render(request, 'monolith.html', {'response': test})    
    return render(request, 'monolith.html')