import uuid
import os
from dotenv import load_dotenv
from src.agents.agent_graph import app
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
# Load environment variables
load_dotenv()

def run_math_mentor():
    # 1. Setup Session Memory
    # The thread_id allows the MemorySaver to track the state for HITL
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id},"recursion_limit": 5}

    print("\n" + "="*50)
    print("🎓 RELIABLE MULTIMODAL MATH MENTOR (JEE)")
    print("="*50)
    print("Type 'quit' or 'q' to exit.")

    # Get the initial problem
    initial_input = input("\n👤 Student: ")
    
    if initial_input.lower() in ['q', 'quit']:
        return

    # 2. Start Initial Execution
    # stream the values to see the 'agent_logs' in real-time
    print("\n⚙️  Processing...")
    for event in app.stream({"raw_input": initial_input}, config, stream_mode="values"):
        if "agent_logs" in event and event["agent_logs"]:
            # Print only the latest log entry
            print(f"🤖 {event['agent_logs'][-1]}")

    # 3. Enter the Interaction Loop (Handles HITL & Final Output)
    while True:
        state = app.get_state(config)
        
        # --- SCENARIO A: GRAPH IS WAITING FOR HUMAN ---
        if state.next and state.next[0] in ["human_intervention", "human_intervention_2"] :
            parsed_data = state.values.get("parsed_output", {})
           

            verifier_data = state.values.get("verifier_output", {})
            
            if parsed_data.get("is_ambiguous"):
                question = parsed_data.get("clarification_needed", "I'm stuck. Could you provide more details?")
                print(f"\n🚨 PARSER NEEDS CLARIFICATION: {question}")

                human_reply = input("👤 Your response: ")

                if human_reply.lower() in ['q', 'quit']:
                    break

                # Inject the human response back into the state
                #  update 'raw_input' so the Parser can try again with fresh info
                old_input = state.values.get("raw_input", "")
                combined_input = f"{old_input} (Correction/Clarification: {human_reply})"
                app.update_state(
                    config, 
                    {"raw_input": combined_input}, 
                    as_node="human_intervention"
                )



            elif verifier_data.get("needs_hitl"):
                print("\n📢 VERIFIER REQUESTS REVIEW")
                print(f"Critique: {verifier_data.get('critique')}")
                print("The AI has solved the problem but wants a human to confirm the logic.")
                print("-" * 30)
                print("OPTIONS: [A]pprove | [E]dit | [R]eject")
        
                choice = input("👤 Your choice: ").lower()

                if choice in ['a', 'approve']:
                    # APPROVE: Manually set is_correct to True and bypass back to Explainer
                    app.update_state(config, {"verifier_output": {"is_correct": True, "needs_hitl": False}}, as_node="human_intervention_2")
                    #  bypass Parser and go straight to Explainer by changing the next node logic
                    # OR  let it flow back to parser but with a "Verified" flag.
                    
                elif choice in ['e', 'edit']:
                    # EDIT: Human provides the correct logic
                    correction = input("👤 Enter the correct steps/logic: ")
                    old_input = state.values.get("raw_input", "")
                    combined = f"OLD INPUT: {old_input} and HUMAN CORRECTION: {correction}"
                    app.update_state(config, {"raw_input": combined}, as_node="human_intervention_2")
                    
                elif choice in ['r', 'reject']:
                    # REJECT: Mark as incorrect and send back to Solver
                    reason = input("👤 Why are you rejecting this? ")
                    app.update_state(config, {
                        "verifier_output": {"is_correct": False, "critique": f"Human Rejected: {reason}", "needs_hitl": False},
                    }, as_node="human_intervention")    
            
            print("\n⚙️  Resuming...")
            # Resume the graph from the interruption point
            for event in app.stream(None, config, stream_mode="values"):
                if "agent_logs" in event and event["agent_logs"]:
                    print(f"🤖 {event['agent_logs'][-1]}")

        # --- SCENARIO B: GRAPH HAS FINISHED (Explainer Done) ---
        elif not state.next:
            final_explanation = state.values.get("explainer_output", "Calculation complete, but no explanation was generated.")
            print("\n" + "✨" + "-"*20 + " FINAL TUTORIAL " + "-"*20 + "✨")
            print(final_explanation)
            print("="*60)


            vectorstore_foldername = "vectorstore"
            current_dir = Path(__file__).parent
            vector_db_path = current_dir / vectorstore_foldername / "chroma_db"
            persist_dir = str(vector_db_path.resolve())

            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
            # Step 4: Load the Chroma Vector Database
            vectordb = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model
            )

            memory_content = f"""
                ### PAST PROBLEM CASE ###
            RAW INPUT: {state.values.get('raw_input','')}
            PARSED OUTPUT: {state.values.get('parsed_output', {}).get('cleaned_text', '')}
            CONTEXT USED: {state.values.get('retrieved_doc','')}
            SOLVER OUTPUT: {state.values.get('solver_output','')}
            FINAL TUTORIAL: {state.values.get('explainer_output','')}
            """
            metadata = {
             "type": "past_solved_problem",
             "topic": state.values.get('parsed_output', {}).get('topic',''),
             "verifier_outcome": "Success" if state.values.get('verifier_output',{}).get('is_correct') else "Corrected",
            "feedback": state.values.get("review_feedback", "Approved by Human")
             }
            
            vectordb.add_texts(
            texts=[memory_content],
            metadatas=[metadata]
            )
            print("🧠 Memory Layer: Interaction successfully embedded and stored.")
            break
        
        # --- SCENARIO C: CATCH-ALL (In case of unexpected pause) ---
        else:
            # If for some reason it's stuck elsewhere, we try to resume
            app.invoke(None, config)

if __name__ == "__main__":
    try:
        run_math_mentor()
    except KeyboardInterrupt:
        print("\n👋 Mentor session ended.")
    except Exception as e:
        print(f"\n❌ A critical error occurred: {e}")