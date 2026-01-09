# import required packages
from pydantic import BaseModel, Field
from .state import MathAgentState
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# prompt for explainer node
EXPLAINER_SYSTEM_PROMPT = """
You are an empathetic and brilliant JEE Math Tutor. 
Your job is to take a verified mathematical solution and rewrite it into a clear, step-by-step explanation for a high school student.

also clean the text .

### GUIDELINES:
- **Tone**: Encouraging, clear, and academic but accessible.
- **Structure**: 
    1. **Concept Overview**: Explain the main mathematical principle used.
    2. **Step-by-Step Breakdown**: Use clear headings. Explain *why* a step is taken.
    3. **Key Pitfall**: Mention one common mistake students make in this type of problem.
    4. **Final Answer**: State the result clearly.
- **Formatting**: Use LaTeX for math expressions (e.g., $x^2$). Use bolding for emphasis.
"""

def explainer_node(state: MathAgentState):
    # prompt template for explainer node
    explainer_prompt = ChatPromptTemplate.from_messages([
        ("system", EXPLAINER_SYSTEM_PROMPT),
        ("human", (
            "PROBLEM: {problem}\n\n"
            "VERIFIED SOLUTION : {solution}\n\n"
            
        ))
    ])

    # load .env variables
    load_dotenv()
    groq_key = os.getenv("GROQ_API_KEY") # load api key
    llm = ChatGroq(model='llama-3.3-70b-versatile', api_key=groq_key, temperature=0.7)
    
    problem = state["parsed_output"]["cleaned_text"]
    solution = state["solver_output"]
    
    # Generate the tutorial
    response = (explainer_prompt | llm).invoke({
        "problem": problem,
        "solution": solution,
       
    })

    return {
        "explainer_output":response.content,
        "agent_logs": ["Explainer: Generated clean student tutorial."]
    }