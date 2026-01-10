# import required packages
import streamlit as st
import uuid
import os
from src.image_audio_input import img_to_text,audio_to_text
from streamlit_mic_recorder import mic_recorder
from src.agents.agent_graph import app
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# remove anonymized info from terminal
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "draft_text" not in st.session_state:
    st.session_state.draft_text = ""

if "pending_problem" not in st.session_state:
    st.session_state.pending_problem = ""

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

if "current_run_logs" not in st.session_state:
    st.session_state.current_run_logs = []

title_container = st.container()
chat_container = st.container(height=400,) 
footer_container = st.container()

# cache chromadb instance
@st.cache_resource
def get_vectorstore():
    # path for vector db
    persist_dir = str((Path(__file__).parent / "vectorstore" / "chroma_db").resolve())
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # This ensures connect once and reuse the same instance
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

def handle_send():
    # 1. Grab the text from the widget key
    user_text = st.session_state.bottom_input
    
    if user_text:
        # 2. Store it in a temporary "buffer" variable
        st.session_state.pending_problem = user_text
        # 3. safe to clear the widget state because 
        # the widget hasn't been "drawn" yet in the new run
        st.session_state.bottom_input = ""

# function to run langgraph agent
def run_agent_cycle(user_input, config):
    """
    function to run agent .
    """
    # set to ture if agent is running
    st.session_state.is_processing = True
   
    # inside chat container
    with chat_container:
        # ai chat messages
        with st.chat_message("assistant"):
            # final result from agent
            final_explanation = ""
            # thinking card
            with st.status("🧠 Math Mentor is working...", expanded=True) as status:
                try:
                    # running agent
                    for event in app.stream(user_input, config, stream_mode="values"):
                        if "agent_logs" in event and event["agent_logs"]:
                            new_log = event["agent_logs"][-1]

                            if any(err in new_log.lower() for err in ["api", "exhausted", "429", "rate limit","api rate limit","rate_limit_exceeded"]):
                                st.error(f"🛑 {new_log}")  # Red error box for visibility
                                st.toast("⚠️ API Limit Reached!") # Brief pop-up notification
                            else:
                                # Normal progress log
                                st.write(f"⚙️ {new_log}")
                            
                            # appending new log from each node to current run log
                            st.session_state.current_run_logs.append(new_log)
                        
                        if "explainer_output" in event:
                            # if explainer result avaiable 
                            final_explanation = event["explainer_output"]
                    
                    status.update(label="✅ Solution Complete", state="complete", expanded=False)
                except Exception as e:
                    # If it gets stuck or errors out, update the status box
                    status.update(label="❌ Agent Halted", state="error")
                    st.error(f"The agent encountered a hitch: {e}")
                    st.info("Check the sidebar to try resuming.")
                    return # Stop the function here

                finally:
                    # agent displayed result
                    st.session_state.is_processing = False
            # if explainer result avaiable
            if final_explanation:
                # join all logs from each node
                all_logs_text = "\n\n\n".join([f"- {l}" for l in st.session_state.current_run_logs])
                # display all log to chat card
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"**Thinking Process:**\n{all_logs_text}",
                    "is_thought": True # Custom flag
                })
                
                # 2. Save the FINAL ANSWER
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": final_explanation,
                    "is_thought": False
                })
                st.markdown(final_explanation)
                # ---  MEMORY LAYER  ---
                try:
                    vectordb = get_vectorstore()
                    state = app.get_state(config)
                    
                    # 4. Prepare Memory Content from previous result
                    memory_content = f"""
                    ### PAST PROBLEM CASE ###
                    RAW INPUT: {state.values.get('raw_input','')}
                    SOLVER OUTPUT: {state.values.get('solver_output','')}
                    FINAL TUTORIAL: {final_explanation}
                    """
                    
                    metadata = {
                        "type": "past_solved_problem",
                        "topic": state.values.get('parsed_output', {}).get('topic',''),
                        "verifier_outcome": "Success" if state.values.get('verifier_output',{}).get('is_correct') else "Corrected"
                    }
                    
                    # 5. Add to Memory
                    vectordb.add_texts(texts=[memory_content], metadatas=[metadata])
                    st.toast("🧠 Memory Layer: Interaction successfully stored!", icon="💾")
                    
                except Exception as e:
                    st.error(f"Error saving to memory: {e}")

st.set_page_config(page_title="Math Mentor", page_icon="🎓",layout="wide")

st.markdown("""
    <style>
        /* This makes the main block take up the full height */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 0rem;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
    </style>
""", unsafe_allow_html=True)

    

# sidebar tools
with st.sidebar:
    st.write(f"**Session ID:** {st.session_state.thread_id}")
    st.title("🛠️ Mentor Tools")
    
    # 1. Selection for input method
    choice = st.radio(
        "Select Input Method:",
        ["⌨️ Text", "🖼️ Image", "🎤 Audio"]
    )
    
  

    # --- Conditional Logic for "inputs" ---
    # text input
    if choice == "⌨️ Text":
        st.subheader("Text Input")
        st.info("Simply type your problem in the main text area.")

    # image input
    elif choice == "🖼️ Image":
        st.subheader("Image Upload")
        uploaded_image = st.file_uploader("Upload an image:", type=["jpg", "png", "jpeg"])
        if uploaded_image is not None:

            # 2. Save it to a temporary file
            temp_img_path = "temp_problem_img.jpg"
            with open(temp_img_path, "wb") as f:
                f.write(uploaded_image.getbuffer())

            # 3. Trigger the OCR (Optical Character Recognition)
            if st.button("🔍 Extract Text from Image"):
                with st.spinner("Reading the math"):
                    # Use your Pix2Text function
                    extracted_text = img_to_text(temp_img_path)
                    
                    # Pop it into the session state for the text area
                    st.session_state.bottom_input = extracted_text
                    st.success("Text extracted!")
                    if os.path.exists(temp_img_path):
                        os.remove(temp_img_path)

                    st.rerun()    

    # audio input
    elif choice == "🎤 Audio":
        st.subheader("Voice Input")                    

        uploaded_audio = st.file_uploader("Upload an audio file:", type=["mp3", "wav", "m4a"])
        if uploaded_audio is not None:
            if st.button("🔊 Transcribe Uploaded Audio"):
                with st.spinner("Transcribing file..."):
                    # Save uploaded file temporarily
                    with open("temp_upload.wav", "wb") as f:
                        f.write(uploaded_audio.getbuffer())
                    
                    # Process and update
                    transcribed_text = audio_to_text("temp_upload.wav")
                    st.session_state.bottom_input = transcribed_text
                    st.success("Transcription complete!")
                    if os.path.exists('temp_upload.wav'):
                        os.remove('temp_upload.wav')
                    st.rerun()    
                    
        # 2. Handle Live Recording
        st.write("Or record your question live:")
        audio = mic_recorder(
            start_prompt="⏺️ Start Recording",
            stop_prompt="⏹️ Stop Recording",
            key='recorder'
        )

        if audio:
            # Check if  a new recording to process
            current_id = audio.get('id')
            if "last_audio_id" not in st.session_state or st.session_state.last_audio_id != current_id:
                with st.spinner("Transcribing your voice... 🎤"):
                    # Save bytes to a temp wav file
                    with open("temp_mic.wav", "wb") as f:
                        f.write(audio['bytes'])
                    
                    # Run your Whisper function
                    result = audio_to_text("temp_mic.wav")
                    
                    # Update state and "remember" this ID so don't repeat
                    st.session_state.bottom_input = result
                    st.session_state.last_audio_id = current_id
                    st.success("Transcription complete!")
                    if os.path.exists('temp_mic.wav'):
                        os.remove('temp_mic.wav')
                    st.rerun()    

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        #  Reset the UI messages
        st.session_state.messages = []
        
        #  Generate a new Thread ID for LangGraph memory
        st.session_state.thread_id = str(uuid.uuid4())
        
        #  Clear any pending inputs or actions
        st.session_state.pending_problem = ""
        st.session_state.verifier_action = None
        
        #  Refresh the app
        st.rerun()     

    st.divider()
    st.subheader("🤖 Agent Status")
    
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    state = app.get_state(config)
    
    # Logic: Only show status/button if NOT currently processing
    if st.session_state.is_processing:
        st.info("⏳ Agent is processing...")
    elif state.next:
        st.warning(f"Paused at: **{state.next[0]}**")
        if st.button("▶️ Resume Agent", use_container_width=True):
            run_agent_cycle(None, config)
            st.rerun()
    else:
        st.success("Agent is Good")              





with title_container:
    st.title("🔢 JEE Math Mentor")

with chat_container:

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("is_thought"):
                with st.expander("🔍 View Thinking Process"):
                    st.markdown(message["content"])
            else:
                st.markdown(message["content"])


    config = {"configurable": {"thread_id": st.session_state.thread_id}, "recursion_limit": 10}
    state = app.get_state(config)

    # ---  GRAPH IS WAITING FOR HUMAN ---
    if state.next and state.next[0] in ["human_intervention", "human_intervention_2"]:
        
        parsed_data = state.values.get("parsed_output", {})
        verifier_data = state.values.get("verifier_output", {})

        with chat_container:
            # --- PARSER NEEDS CLARIFICATION ---
            if parsed_data.get("is_ambiguous"):
                question = parsed_data.get("clarification_needed", "I'm stuck. Could you provide more details?")
                
                with st.chat_message("assistant"):
                    st.warning(f"🚨 **PARSER NEEDS CLARIFICATION:** {question}")
                    
                    # use a form so the user can type and submit
                    with st.form("clarification_form"):
                        human_reply = st.text_input("👤 Your response:")
                        if st.form_submit_button("Send Clarification"):
                            # update the state
                            old_input = state.values.get("raw_input", "")
                            combined_input = f"{old_input} (Correction/Clarification: {human_reply})"
                            
                            app.update_state(
                                config, 
                                {"raw_input": combined_input}, 
                                as_node="human_intervention"
                            )
                            run_agent_cycle(None, config)
                            st.rerun() # Refresh to start processing again    

            elif verifier_data.get("needs_hitl"):
                        with st.chat_message("assistant"):
                            st.info("📢 **VERIFIER REQUESTS REVIEW**")
                            st.write(f"**Critique:** {verifier_data.get('critique', 'No critique provided.')}")
                            st.write("The AI has solved the problem but wants a human to confirm the logic.")
                            st.divider()

                            # Create three columns for the buttons
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                if st.button("✅ Approve", use_container_width=True):
                                    # Logic: Set is_correct to True and let the graph finish
                                    app.update_state(config, {"verifier_output": {"is_correct": True, "needs_hitl": False}}, as_node="human_intervention_2")
                                    # This 'None' input tells the graph to resume from where it paused
                                    run_agent_cycle(None, config) 
                                    st.rerun()

                            with col2:
                                if st.button("✏️ Edit Logic", use_container_width=True):
                                    st.session_state.verifier_action = "edit"

                            with col3:
                                if st.button("❌ Reject", use_container_width=True):
                                    st.session_state.verifier_action = "reject"

                            # --- Handle Secondary Inputs (Edit/Reject) ---
                            if st.session_state.get("verifier_action") == "edit":
                                with st.form("edit_logic_form"):
                                    correction = st.text_input("Enter the correct steps/logic:")
                                    if st.form_submit_button("Submit Correction"):
                                        old_input = state.values.get("raw_input", "")
                                        combined = f"{old_input} (Human Correction: {correction})"
                                        app.update_state(config, {"raw_input": combined}, as_node="human_intervention_2")
                                        st.session_state.verifier_action = None # Reset
                                        run_agent_cycle(None, config)
                                        st.rerun()

                            elif st.session_state.get("verifier_action") == "reject":
                                with st.form("reject_logic_form"):
                                    reason = st.text_input("Why are you rejecting this?")
                                    if st.form_submit_button("Confirm Rejection"):
                                        app.update_state(config, {
                                            "verifier_output": {"is_correct": False, "critique": f"Human Rejected: {reason}", "needs_hitl": False},
                                        }, as_node="human_intervention")
                                        st.session_state.verifier_action = None # Reset
                                        run_agent_cycle(None, config)
                                        st.rerun()

with footer_container:
 
    st.text_area(
        "Input:",
        
        height=120,
        key = 'bottom_input',
        label_visibility="collapsed", # Hides the label for a cleaner look
        placeholder="Edit your extracted math problem here..."
    )

    if st.button("🚀 Send to Math Mentor", type="primary", on_click=handle_send):
        
        # Check our "buffer"
        if st.session_state.pending_problem:
            user_problem = st.session_state.pending_problem
            
            # Add to chat history
            st.session_state.messages.append({"role": "user", "content": user_problem})
            
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_problem)
            # Setup config
            config = {"configurable": {"thread_id": st.session_state.thread_id}, "recursion_limit": 10}
            
            # Run the AI Agent
            run_agent_cycle({"raw_input": user_problem}, config)
            
            # Clear the buffer so we don't send the same problem twice on next rerun
            st.session_state.pending_problem = ""
            st.rerun()

    
