# import required packages
from pydantic import BaseModel, Field
from .state import MathAgentState
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# output format for verifier node 
class VerificationResult(BaseModel):

    is_correct: bool = Field(description="True if the solution is mathematically sound and follows the logic.")

    score: float = Field(description="A confidence score between 0 and 1.")

    critique: str = Field(description="Detailed explanation of errors or confirmation of correctness.")

    needs_hitl: bool = Field(description="True if the agent is unsure and needs a human to check.")


def verifier_node(state: MathAgentState):
    # prompt foor verifier node
    verifier_system_prompt = """
### INPUT DATA:
1. **Original Problem**: The cleaned question the student asked.
2. **Solver output**: The logical progression used to reach the answer.


You are a Senior JEE Math Examiner. Your job is to verify the Solver's work.
CRITERIA:
1. Correctness: Is the final answer mathematically true?
2. Units/Domain: Are constraints respected? (e.g., probability between 0-1, no division by zero).
3. Logic: Does every step follow logically from the previous one?
4. Edge Cases: Did the solver miss any special conditions?
If the solution is '100%' correct, set is_correct=True. 
If you are even '1%' unsure or the problem is highly complex, set needs_hitl=True.

### FIELD INSTRUCTIONS:
- **is_correct**: Set to False if there is ANY mathematical error.
- **score**: 1.0 for perfect logic, 0.0 for complete failure. Use 0.5-0.7 if the logic is mostly sound but the final calculation is slightly off.
- **critique**: Be specific. Instead of "wrong," say "The solver failed to apply the chain rule correctly in step 3."
- **needs_hitl**: If the math is wrong but fixable, set is_correct = False and needs_hitl = False. This triggers the loop back to the Solver.
    If the math is so confusing that the AI can't even critique it, then set needs_hitl = True.

CRITICAL: You are the last line of defense. Be extremely skeptical.    
"""

    # prompt template for verifier node
    verifier_prompt = ChatPromptTemplate.from_messages([
        ("system", verifier_system_prompt),
        ("human", (
            "ORIGINAL PROBLEM: {problem}\n\n"
            "Solver's output: {solver_output}\n\n"
            
        ))
    ])
    # load .env variables
    load_dotenv()
    groq_key = os.getenv("GROQ_API_KEY") # load api key
    # llm initialization
    llm = ChatGroq(model='llama-3.3-70b-versatile', api_key=groq_key, temperature=0.7)
    
    structured_llm = llm.with_structured_output(VerificationResult)
    # connecting prompt template to llm
    chain = verifier_prompt | structured_llm
    try:
        verifier_output = chain.invoke({
        "problem": state["parsed_output"]['cleaned_text'],
        "solver_output": state["solver_output"],
        
        })
    
        return {
        "verifier_output": verifier_output.model_dump(),
        "agent_logs": [
                f"Verifier Agent: Successfully verified the problem.\n"
                f"Output: {verifier_output.model_dump()}" 
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
            "agent_logs": [friendly_error]
           
        }

    