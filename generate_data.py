from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

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


# partitioning pdf document and create chunks
filename = "pyATS_p1_30.pdf"
print(filename)
chunks = chunk_document(filename)

print(f"{len(chunks)}  chunks of text are created")

# generate question-answer pairs to be the data for model finetuning 
n = 1
j = 50
for i in range(0,n):
     #qa_output = create_qa_pairs_JSON_parser(chunks[i].text)
     qa_output = create_qa_pairs(chunks[i].text)
     print("--------------------------------")
     print(chunks[i].text)
     print("--------------------------------")
     print(qa_output)
