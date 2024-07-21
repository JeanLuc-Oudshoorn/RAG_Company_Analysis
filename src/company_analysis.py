import os
import openai
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load the environment
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# Define company
company_name = "BrainChip"

# Load company specific vector store
vector_store = FAISS.load_local(f'../vector_stores/{company_name.lower()}.faiss', OpenAIEmbeddings(),
                                allow_dangerous_deserialization=True)

# Create retriever
retriever = vector_store.as_retriever(search_type='mmr',
                                      search_kwargs={'k': 5})

# Create the model
model = ChatOpenAI(model="gpt-4-turbo")

# Build a prompt template
template = """
You are an expert financial analyst that performs quantitative and qualitative analysis on companies listed on 
the stock market to evaluate if a worthwhile investment opportunity exists. You excel at providing a nuanced view 
of what the future trajectory of the company may look like. You do this by suggesting several scenarios that may play 
out where appropriate, as well as assigning a rough probability estimate to each scenario (very unlikely - very likely). 
You highlight the company's strengths as well as its weaknesses in your scenario descriptions. 

An excellent financial analyst analyses how a company is positioned in relation to the competition, if the company 
receives any tailwinds from mega-trends (e.g. sustainability, AI, etc.) and if the company's unique selling point 
offers an attractive value proposition for its target audience.

Use fair and wise judgement, as the stakes are very high! Your answer will determine the size of the investment in 
each company! 

Do not give politically correct answers along the lines of it is hard to predict the future, just provide your best 
estimate. This is your job as an elite Wall Street financial analyst. The users expect usable information.

Use the context provided below in backticks, in addition to your own knowledge to answer the question to the best 
of your ability.

Let's think step by step.

QUESTION:
{question}

CONTEXT:
```{context}```
"""

prompt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()

setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)

# Instructions
instructions = [
    f"How viable is the technology from {company_name}? What are its strengths and weaknesses. "
    f"Is there a high likelihood that this technology will be adopted at an increasing pace in the next five years?",
    f"What are the major incentives to adopt neuromorphic computing hardware? What are the obstacles? "
    f"Are there any interested parties that want to integrate this into their products in the next five years?",
    f"What are the major risks for {company_name}? How likely are these risks to play out in the future? "
    f"How may these risks be mitigated and is {company_name} doing anything to manage the risks?",
    f"What caused the 95% YoY revenue drop in 2023? What is {company_name} doing to reverse this trend? "
    f"How likely is it that revenue will grow in the future?",
    f"Does {company_name} have a well-defined path towards profitability? What could hinder or accelerate this path? "
    f"What are the catalysts that would help the company take off?",
    f"The {company_name} share price is A$0.20 on 21 July 2024, is this a reasonable evaluation all things considered? "
    f"Provide scenarios of what the stock price may be at the end of 2025, 2030 and 2050.",
 ]

# Set up section writing chain with LCEL
question_answering_chain = setup_and_retrieval | prompt | model | output_parser

# Create an empty list for written sections
answers = []

# Iteratively call the LLM to the sections
for instruction in instructions:
    response = question_answering_chain.invoke(instruction)
    print(f"INSTRUCTION: {instruction}\n {response}")
    answers.append(f"\n\n ##NEW INSTRUCTIONS: {instruction}")
    answers.append(response)

# Full company report
full_company_report = ''.join(answers)

# Save the string 'full_case_study' to a .txt file
with open(f'../results/{company_name}_full_report.txt', 'w') as f:
    f.write(full_company_report)
