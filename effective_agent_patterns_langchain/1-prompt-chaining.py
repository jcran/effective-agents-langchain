from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

outline_pt = PromptTemplate.from_template("Write a 5-bullet outline for an article on {topic}.")
validate_pt = PromptTemplate.from_template("Does this outline cover key points? Answer YES/NO and list missing topics:\n{outline}")
draft_pt   = PromptTemplate.from_template("Write the full article based on this outline:\n{outline}")

outline_chain  = LLMChain(llm=llm, prompt=outline_pt,  output_key="outline")
validate_chain = LLMChain(llm=llm, prompt=validate_pt, output_key="validation")
draft_chain    = LLMChain(llm=llm, prompt=draft_pt,   output_key="article")

seq = SequentialChain(chains=[outline_chain, validate_chain, draft_chain],
                      input_variables=["topic"], output_variables=["article"], verbose=True)

topic = input("Enter article topic: ")
result = seq({"topic": topic})

print(result["article"])