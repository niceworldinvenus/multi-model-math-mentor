# import required packages
from pydantic import BaseModel, Field
from typing import List, Optional
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from .state import MathAgentState

# output format of parser node
class ParsedMathProblem(BaseModel):

    cleaned_text: str = Field(description="The cleaned version of the math problem")

    topic: str = Field(description="The math domain (Algebra, Probability, Calculus (limits, derivatives, simple optimization),Linear algebra)")
    
    constraints: Optional[List[str]] = Field(description="List of constraints found in question Ex. ['x>0','y > 8']")

    variables: List[str] = Field(description="List of variables found in the problem")

    is_ambiguous: bool = Field(description="True if information is missing or unclear")

    clarification_needed: Optional[str] = Field(description="The specific question to ask the human if is_ambiguous is True")



parser_system_prompt = """
You are a Precision-Engineered Math Parser Agent. Your role is to act as a bridge between noisy OCR/ASR input and a formal Mathematical Solver. You specialize in JEE-level competitive math (Algebra, Calculus, Vectors, etc.).

### CORE LOGIC: THE "CLEANUP" PROTOCOL
When processing 'raw_input', apply these rules in order:
1. **Visual Correction (OCR Fixes):**
   
   - Identify 'x' as a variable vs '*' as multiplication based on context.

2. **ASR Phonetic Correction:**
   - "four" -> "4", "plus" -> "+", "equals" -> "=", "into" -> "*".
   - "x squared" -> "x^2", "root" -> "sqrt()".

3. **Notation Standardization:**
   - Use `^` for exponents, `*` for multiplication, `/` for division.
   - Ensure parentheses are balanced. If the raw text has `(x+5`, fix it to `(x+5)`.

### STRUCTURED FIELD DEFINITIONS:
1. **cleaned_text**: The reconstructed, readable problem statement. If the problem is a "Find" or "Solve" task, ensure the instruction is clear. Use standard ASCII math symbols.

2. **topic**: Classify into EXACTLY one of these JEE categories:
   - **Calculus**: Limits, Derivatives, Integrals, Differential Equations.
   - **Algebra**: Quadratic equations, Complex numbers, Sequences, Matrices, Determinants.
   - **Linear Algebra**: Vectors and 3D Geometry.
   - **Probability**: Probability, Statistics, Permutations & Combinations.
   - **Trigonometry**: Inverse trig, Trigonometric identities.

3. **constraints**: Extract all domain/boundary conditions.
   - Example: `["x > 0", "theta in [0, 2pi]", "n is an integer"]`. Return `[]` if none.

4. **variables**: Identify all unknowns and their given values if provided.
   - Example: `["x", "y=10", "A (matrix)"]`.

5. **is_ambiguous (Boolean)**: Set to `True` if:
   - Critical operators are missing (e.g., "2x 5 = 10" — is it + or -?).
   - The OCR is too garbled to form a valid JEE-level question.
   - The question asks for a variable that doesn't exist in the text.
   - if you have doubt in terms of query set is_ambigous to True

6. **clarification_needed**: 
   - If `is_ambiguous` is True, describe exactly what is missing or what is reason for setting is_ambigous to True. 
   - *Bad:* "The text is messy." 
   - *Good:* "The operator between '2x' and '5' is missing. Did you mean 2x + 5 or 2x - 5?"

### CRITICAL CONSTRAINTS:
- **NO SOLVING:** Do not provide the answer or steps to the solution.
- **NO ASSUMPTIONS:** If a number is truly illegible, do not guess; set `is_ambiguous` to True.
- **JSON ONLY:** Return the result in a valid JSON-parseable format.

### EXAMPLE 1:
      Raw query: "find the area under y=xS from x=O to 2"
      Output:
      "cleaned_text": "Find the area under the curve y = x^5 from x = 0 to x = 2.",
      "topic": "Calculus",
      "constraints": ["x is in [0, 2]"],
      "variables": ["y", "x"],
      "is_ambiguous": false,
      "clarification_needed": null

###Example-2 of what the "clean" version of 'ifx=5 and y= 6. thenfindthe value of  x + (y ^ 2)' would look like:
   "cleaned_text": "if x = 5 and y = 6 then find the value of x + (y ^ 2) ?",
   "topic": "Algebra",
   "constraints":None, 
   "variables": ["x = 5", "y = 6"],
   "is_ambiguous": False,
  "clarification_needed": "No clarification needed."

### Example-3 of what the "clean" version of 'if x = S and  y = M. then find the value of (x+y) ^ 2' would look like:
   "cleaned_text": "if x = S and  y = M. then find the value of (x+y) ^ 2 ? ",
   "topic": "Algebra",
   "constraints":None, 
   "variables": ["x = S", "y = M"],
   "is_ambiguous": True,
  "clarification_needed": " I need clarification!. what is the values of S and M . if S sand M are numerical digits tell me."  

"""

def parser_node(state: MathAgentState):
    
    
    # load .env variables
    load_dotenv()
    groq_key = os.getenv("GROQ_API_KEY") # load groq api key

   # propmt template for parser node
    parser_prompt_template = ChatPromptTemplate.from_messages([
    ("system", parser_system_prompt),
    ("human", "Raw Input: {raw_input}")
    ])

   # model initialization
    llm = ChatGroq(model='llama-3.3-70b-versatile', api_key=groq_key, temperature=0.7)
    structured_llm = llm.with_structured_output(ParsedMathProblem)

   # connecting prompt template to model
    planner_chain = parser_prompt_template | structured_llm

    try:
        # Invoke the LLM to generate the plan
        parser_output = planner_chain.invoke({
            "raw_input": state.get("raw_input"),
            
        })
       
        return {
            # passing parser output to graph state
            "parsed_output": parser_output.model_dump()  ,
            "agent_logs": [
                # logging to agent logs in graph state
                f"Parser Agent: Successfully structured the problem.\n"
                f"Output: {parser_output.model_dump()}" # Fixed the dot and the call
            ]       
        }
        
    except Exception as e:
        error_msg = str(e)
        if "rate_limit_exceeded" in error_msg.lower() or "429" in error_msg:
            friendly_error = "⚠️ API Rate Limit Exhausted. "
        else:
            friendly_error = f"❌ Solver Error: {error_msg}"
        
        # Return the error in the logs so the UI can see it
        return {
            "agent_logs": [friendly_error],
         
        }



   