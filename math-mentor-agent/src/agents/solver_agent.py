# import required packages
from pathlib import Path
from typing import Dict, Any, List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from .state import MathAgentState
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# function to retrive documents to model
def retrive_docs(query):

    # query to search in vector database
    query_to_search = query

    vectorstore_foldername = "vectorstore"
    current_dir = Path(__file__).parent
    vector_db_path = current_dir.parent.parent / vectorstore_foldername / "chroma_db"
    #vectordb folder path
    persist_dir = str(vector_db_path.resolve())
    # loading embedding model to retrive documents
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Step 4: Load the Chroma Vector Database
    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model
    )
    # retriving documents from db
    docs = vectordb.similarity_search(
            query=query_to_search,
            k=3,
            
        )
    # retriving past solutions data using key : type and value : past_solved_problem
    docs_previous_solutions= vectordb.similarity_search(
            query=query_to_search,
            k=1,
            filter={"type": "past_solved_problem"}
            
        )
        
    
    # combining all documents content.
    combined_content = "THEORY:\n"+"\n\n".join([doc.page_content for doc in docs])
    if docs_previous_solutions:
        # combining past solutions data to retrived theory content
        combined_content += "\n\nSIMILAR PAST SOLUTION:\n" + docs_previous_solutions[0].page_content


    return combined_content

def solver_node(state: MathAgentState):
    # query from parser node output
    query = state['parsed_output']['cleaned_text']
    # topic from parser node output
    topic = state['parsed_output']['topic']
    # retrived content
    retrieved_context = retrive_docs(query)
    # verifier output if avaiable
    verifier_feedback = state.get("verifier_output", {})
    # critique from verifier output (feedback from user)
    critique = verifier_feedback.get("critique", "")
    # base prompt for llm
    base_solver_instructions = """
    You are an Expert JEE {topic} Mentor. Your goal is to provide a rigorous yet accessible solution.
    A student of intermediate level will read this, so maintain a balance between technical depth and clarity.

    ### OPERATIONAL PROTOCOL:
    1. **RAG Integration**: Carefully review the 'Retrieved Context'. 
    - If the context contains relevant JEE shortcuts, theorems, or formulas, PRIORITIZE them.
    - If the context is irrelevant, rely on your internal expert knowledge only .
    2. **Mathematical Rigor**: Use LaTeX for ALL mathematical expressions ($...$ for inline, $$...$$ for blocks).
    3. **Pedagogical Structure**:
    - **Phase 1: Problem Analysis**: Briefly restate the core challenge and identify knowns/unknowns.
    - **Phase 2: Strategy & Tools**: Mention the specific JEE concepts and formulae (e.g., L'Hopital's Rule, Section Formula, etc.) you will use.
    - **Phase 3: Step-by-Step Execution**: Show every logical transition. Do not skip intermediate algebraic steps.
    - **Phase 4: Final Verification**: State the final answer clearly in a **boxed format**.

    ### CONTEXTUAL DATA:
    - **Retrieved JEE Resources**: {retrieved_doc}
    - **Problem to Solve**: {query}
    """
    # using critique avaiable from verifier output for self learning 
    if critique:
        solver_system_prompt = f"""
        {base_solver_instructions}

        ### ⚠️ CRITICAL REVISION REQUIRED:
        Your previous solution was flagged as INCORRECT. 
        **EXAMINER FEEDBACK**: "{critique}"

        **INSTRUCTIONS FOR RETRY**:
        1. Acknowledge the error mentioned in the feedback.
        2. Re-evaluate your previous logic specifically around the feedback point.
        3. Ensure this new solution addresses the critique while following the standard Pedagogical Structure.
        """
    else:
        solver_system_prompt = f"""
        {base_solver_instructions}
        
        **INSTRUCTIONS**: Provide a fresh, detailed solution following the Pedagogical Structure mentioned above.
        """
    # prompt template for llm
    solver_prompt_template = ChatPromptTemplate.from_messages([
    ("system", solver_system_prompt),
    ("human", "Solve this: {query}")
    ])
    # load .env variables
    load_dotenv()
    groq_key = os.getenv("GROQ_API_KEY") # load api key
    # llm initialization
    llm = ChatGroq(model='openai/gpt-oss-120b', api_key=groq_key)
    # connecting prompt template to llm 
    planner_chain = solver_prompt_template | llm

    try:
        # Invoke the LLM to generate the plan
        solver_output = planner_chain.invoke({
            "query": query,
            "topic": topic,
            "retrieved_doc": retrieved_context,
            "critique":critique
        })
        
        return {
            "solver_output": solver_output.content ,
            "agent_logs": [
                f"Solver Agent: Successfully solved the problem.\n"
                f"Output: {solver_output.content}" # Fixed the dot and the call
            ]    
        }
        
    except Exception as e:

         
        error_msg = str(e)
        if "rate_limit_exceeded" in error_msg.lower() or "429" in error_msg:
            friendly_error = "⚠️ API Rate Limit Exhausted. Please wait a moment before retrying."
        else:
            friendly_error = f"❌ Solver Error: {error_msg}"
        
        # Return the error in the logs so the UI can see it
        return {
            "agent_logs": [friendly_error],
         
        }         

        

    
