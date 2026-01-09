from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from .state import MathAgentState
from .parser_agent import parser_node
from .router_agent import intent_router
from .solver_agent import solver_node
from .verifier_agent import verifier_node
from .explainer_agent import explainer_node

# 1. Initialize Memory (Checkpointer)
# This handles the "Memory" requirement by saving the state of every thread.
memory = MemorySaver()

# 2. Initialize the Graph
workflow = StateGraph(MathAgentState)

workflow.add_node("parser_node", parser_node)
workflow.add_node("solver_node", solver_node)
workflow.add_node("verifier_node", verifier_node)
workflow.add_node("explainer_node", explainer_node)

workflow.add_node("human_intervention", lambda state: state)
workflow.add_node("human_intervention_2", lambda state: state)

workflow.set_entry_point("parser_node")

# Edge 1: Parser -> Router (Conditional)
workflow.add_conditional_edges(
    "parser_node",
    intent_router,
    {
        "human_intervention": "human_intervention",
        "algebra_solver": "solver_node",
        "calculus_solver": "solver_node",
        "probability_solver": "solver_node",
        "linear_algebra_solver": "solver_node",
        "general_solver": "solver_node"
    }
)

# Edge 2: Solver -> Verifier (Fixed)
workflow.add_edge("solver_node", "verifier_node")

# Edge 3: Verifier -> Loop/HITL/Explainer (Conditional)
def verifier_routing(state: MathAgentState):
    v_out = state.get("verifier_output") or {}
    
    # 1. Trigger 2: Verifier is unsure (HITL)
    if v_out.get("needs_hitl"):
        return "human_intervention_2"
    
    # 2. Self-Correction Loop
    if v_out.get("is_correct") is False: # Explicitly check for False
        return "solver_node" 
    
    # 3. Proceed to Explainer
    return "explainer_node"

def human_review_router(state: MathAgentState):
    """
    Decides where to go after human review in human_intervention_2.
    """
    v_out = state.get("verifier_output") or {}
    
    # If human approved (is_correct set to True in main.py), skip to Explainer
    if v_out.get("is_correct") is True:
        return "explainer_node"
    
    # If human edited or rejected, go back to Solver to re-process
    return "solver_node"

workflow.add_conditional_edges(
    "verifier_node",
    verifier_routing,
    {
        "human_intervention_2": "human_intervention_2",
        "solver_node": "solver_node",
        "explainer_node": "explainer_node"
    }
)

# Edge 4: Explainer -> End
workflow.add_edge("explainer_node", END)

# Edge 5: Human Intervention -> Parser (To re-process after user fix)
workflow.add_edge("human_intervention", "parser_node")
workflow.add_conditional_edges(
    "human_intervention_2",
    human_review_router,
    {
        "explainer_node": "explainer_node",
        "solver_node": "solver_node"
    }
)

# 5. Compile the Graph
# We use interrupt_before to stop the execution specifically at the human node.
app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["human_intervention","human_intervention_2"]
)