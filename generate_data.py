from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
import json
import pandas as pd

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = 'gpt-4o'
temperature = 0

def chunk_document(filename): 
    print("start partitioning")
    
    # Returns a List[Element] present in the pages of the parsed pdf document
    elements = partition_pdf(filename)
    print("done partitioning pdf into elements")

    # chunking the elements
    #specify max characters for each chunk (2**8)*x where x = 4
    print("start chunking")
    chunks = chunk_by_title(elements,max_characters=1024)
    print("done chunking")
    
    return chunks


# create qa pairs using openAI without JSON output parser #

def create_qa_pairs(content):
    
    # Create a prompt template
    prompt = PromptTemplate(
        template= '''Create 5 questions for a short-answer quiz in JSON format with keys “question“,”answer” 
        solely from the following information —-{content}—-''',
        input_variables=["content"]
    )

    # Initialize the OpenAI LLM
    model = ChatOpenAI(temperature=temperature)

    # Generate the response
    chain = prompt | model
    response = chain.invoke({"content": content})

    return response

# create qa pairs using openAI with JSON output parser#
def create_qa_pairs_JSON_parser(content):
    
    # Define the output structure
    class QAOutput(BaseModel):
            question: str = Field(description="The question asked")
            answer: str = Field(description="The answer to the question")

    # Set up the output parser
    parser = JsonOutputParser(pydantic_object=QAOutput)

    # Create a prompt template
    prompt = PromptTemplate(
        template= '''Create 5 questions for a short-answer quiz in JSON format with keys “question“,”answer” 
        solely from the following information —-{content}—-\n{format_instructions}\n''',
        input_variables=["content"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    # Initialize the OpenAI LLM
    model = ChatOpenAI(temperature=temperature)

    # Generate the response
    chain = prompt | model | parser
    response = chain.invoke({"content": content})

    return response


def create_qa_pairs_JSON_llama(content):

    # get question-answer pairs from the provided content using llama 3.2
    llm = ChatOllama(
        model = "llama3.2",
        temperature = 0,
    )

    messages = [
        ("human", f"Create 5 question-answer pairs for a short-answer quiz in JSON format with keys “question“,”answer” solely from the following information —-{content}---"),
    ]
    print("creating question-answer pair")
    response = llm.invoke(messages).content
    print(response)

    print("extracting question-answer from LLM response")
    # split response string to get only the question and answer parts of the response
    test = response.split("```")
    #print(test[1])

    # convert string to  object
    if len(test) > 1:
        json_object = json.loads(test[1])
    else:
        json_object = []

    # print the output
    print(type(json_object))
    print(json_object)
    print("done")

    return json_object


# partitioning pdf document and create chunks
#filename = "pyATS_p1_30.pdf"
#print(filename)
#chunks = chunk_document(filename)
#print(f"{len(chunks)}  chunks of text are created")

# convert chunks to dataframe and save to document_chunks.csv
#print("saving document chunks to document_chunks.csv")
#df_chunks = pd.DataFrame(chunks)
#df_chunks.to_csv('document_chunks.csv', index=False, header=["text"])


# generate question-answer pairs to be the data for model finetuning 
doc_chunks = pd.read_csv("document_chunks.csv")
output = []
output_filename = 'training_data.csv'
# start creating questions and answers for chunk j to last chunk 
j = 25 
n = len(doc_chunks)
print(f"working on chunk {j} to {n-1}")
for i in range(j,n):
     print(f"*****************chunk {i}******************")
     print("--------------------------------")
     print(doc_chunks["text"][i])
     print("--------------------------------")
     #qa_output = create_qa_pairs_JSON_parser(chunks[i].text)
     #qa_output = create_qa_pairs(chunks[i].text)
     qa_output = create_qa_pairs_JSON_llama(doc_chunks["text"][i])

     print(qa_output)
     
     if len(qa_output) > 0:
        # convert output to dataframe
        df = pd.DataFrame(qa_output)
        
        # saving output to training_data.csv file
        if os.path.exists(output_filename):
            # append the dataframe to training_data.csv file
            print("adding output to training_data.csv")
            df.to_csv(output_filename, mode='a', index=False, header=False)
        else:
            # create training_data.csv file and save the dataframe to the file 
            print("saving output to training_data.csv")
            df.to_csv(output_filename, index=False)
     else:
        print("no new questions-answers created.")
    

        #print("extending the final output list of questions-answers")
        #output.extend(qa_output)

#print("final output :")
#print(output)

