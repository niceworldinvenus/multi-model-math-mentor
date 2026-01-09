from typing import Literal
from .state import MathAgentState

def intent_router(state: MathAgentState) -> Literal[
    "human_intervention", 
    "algebra_solver", 
    "calculus_solver", 
    "probability_solver", 
    "linear_algebra_solver",
    "general_solver"
]:
    """
    Analyzes the parsed problem to route the workflow to the correct specialist.
    """
    parsed_output = state.get("parsed_output", {})
    
    # 1. Check for Ambiguity first - this is our HITL trigger 🛑
    if parsed_output.get("is_ambiguous"):
        print("--- ROUTER: Ambiguity detected. Routing to HITL ---")
        return "human_intervention"

    # 2. Routing based on the 'topic' field from our Parser 🏎️
    topic = parsed_output.get("topic", "").lower()
    
    if "algebra" in topic:
        return "algebra_solver"
    elif "calculus" in topic:
        return "calculus_solver"
    elif "probability" in topic:
        return "probability_solver"
    elif any(t in topic for t in ["Linear algebra", "quadratic", "polynomial"]):
        return "linear_algebra_solver"
    
    # 3. Fallback if the topic doesn't match our specialists 
    return "general_solver"