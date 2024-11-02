import streamlit as st
import os
import sys
from together import Together
from datetime import datetime

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Update the import to use the correct path
from utils.document_processor import DocumentProcessor

# Set your Together AI API key
os.environ["TOGETHER_API_KEY"] = st.secrets["TOGETHER_API_KEY"]

def init_together_client():
    return Together()

def get_assistant_response(messages):
    client = init_together_client()
    try:
        # Get the user's latest message
        user_message = messages[-1]["content"]
        
        # Get relevant document chunks
        doc_processor = DocumentProcessor()
        relevant_chunks = doc_processor.query_similar(user_message)
        
        # If we found relevant chunks, include them in the context
        if relevant_chunks:
            context = "\n\n".join(relevant_chunks)
            system_message = {
                "role": "system",
                "content": f"Use the following context to help answer the user's question:\n\n{context}\n\nIf the context is relevant, use it to provide a detailed response. If the context isn't relevant, you can answer based on your general knowledge."
            }
            # Add system message with context at the beginning
            augmented_messages = [system_message] + messages
        else:
            augmented_messages = messages
        
        # Get response from LLM with context
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-lora",
            messages=augmented_messages,
            max_tokens=1000,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def process_uploaded_file(uploaded_file):
    try:
        # Save the uploaded file temporarily
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Process the file using DocumentProcessor
        doc_processor = DocumentProcessor()
        doc_processor.process_file(temp_file_path, uploaded_file.name)
        
        # Clean up the temporary file
        os.remove(temp_file_path)
        return True
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return False

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def main():
    st.set_page_config(page_title="AI Chat Assistant", page_icon="🤖")
    st.title("AI Chat Assistant 🤖")
    
    init_session_state()
    
    # Initialize states
    if "uploaded_file_processed" not in st.session_state:
        st.session_state.uploaded_file_processed = False
    if "should_clear" not in st.session_state:
        st.session_state.should_clear = False

    def on_file_processed():
        st.session_state.should_clear = True
        st.session_state.uploaded_file_processed = True
    
    # Add file uploader in the sidebar
    st.sidebar.header("Upload Document")
    
    # If should_clear is True, create a new key for the file uploader
    key = f"file_uploader_{datetime.now().timestamp()}" if st.session_state.should_clear else "file_uploader"
    uploaded_file = st.sidebar.file_uploader(
        "",  # Remove the label text since we have the header
        type=["txt", "md"],
        help="Upload a text or markdown file to include in the conversation",
        key=key
    )
    
    # Create a sidebar container for processing status right after the uploader
    status_container = st.sidebar.container()
    
    # Reset should_clear after creating new uploader
    if st.session_state.should_clear:
        st.session_state.should_clear = False
    
    # Automatically process file when uploaded and not already processed
    if uploaded_file and not st.session_state.uploaded_file_processed:
        with status_container:
            progress_bar = st.progress(0)
            st.caption(f'Processing: {uploaded_file.name}')
            
            # Process the file
            success = process_uploaded_file(uploaded_file)
            progress_bar.progress(100)
            
            if success:
                st.success('File processed successfully!')
                on_file_processed()
                st.rerun()
    
    # Display uploaded files with delete buttons
    doc_processor = DocumentProcessor()
    uploaded_files = doc_processor.get_uploaded_files()
    if uploaded_files:
        st.sidebar.subheader("Uploaded Files")
        for file in uploaded_files:
            col1, col2 = st.sidebar.columns([4, 1])
            with col1:
                st.text(f"• {file}")
            with col2:
                if st.button("🗑️", key=f"delete_{file}"):
                    doc_processor.remove_file(file)
                    st.rerun()
    
    # Reset the processed flag when no file is uploaded
    if not uploaded_file:
        st.session_state.uploaded_file_processed = False
    
    # Display chat messages
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        with st.chat_message(role):
            st.write(content)

    # User input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get assistant response
        assistant_response = get_assistant_response(st.session_state.messages)
        
        if assistant_response:
            # Display assistant response
            with st.chat_message("assistant"):
                st.write(assistant_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})

if __name__ == "__main__":
    main()