from typing import TypedDict, List, Optional,Annotated
import operator

class MathAgentState(TypedDict):
    # Inputs
    raw_input: str

    learning_signals: Annotated[List[str], operator.add]
    # Processed Data
    parsed_output: dict  

    #Retrived Content
    retrieved_doc: str
    
    # Outputs
    solver_output: str

    verifier_output:Annotated[dict, operator.ior]

    explainer_output : str

    review_feedback: str

    similar_past_solutions: List[dict] #Will hold the problems the agent previously solved.
    
    # Agent History (The Trace)
    agent_logs: Annotated[List[str], operator.add]