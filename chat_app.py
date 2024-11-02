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

def check_api_key():
    if "TOGETHER_API_KEY" not in st.secrets:
        st.error("Please set the TOGETHER_API_KEY in your Streamlit secrets.")
        st.stop()

def init_together_client():
    return Together()

def get_assistant_response(messages):
    client = init_together_client()
    try:
        # Get the user's latest message
        user_message = messages[-1]["content"]
        
        with st.status("🤔 Processing your request...", expanded=True) as status:
            # Get relevant document chunks
            status.write("Searching relevant documents...")
            doc_processor = DocumentProcessor()
            relevant_chunks = doc_processor.query_similar(user_message)
            
            # If we found relevant chunks, include them in the context
            if relevant_chunks:
                status.write("Found relevant context...")
                # Limit context size by taking most relevant chunks up to ~2000 tokens
                # Assuming ~4 chars per token, limit to 8000 chars
                combined_chunks = ""
                for chunk in relevant_chunks:
                    if len(combined_chunks) + len(chunk) < 8000:
                        combined_chunks += chunk + "\n\n"
                    else:
                        break
                
                system_message = {
                    "role": "system",
                    "content": f"Use the following context to help answer the user's question:\n\n{combined_chunks}\n\nIf the context is relevant, use it to provide a detailed response. If the context isn't relevant, you can answer based on your general knowledge."
                }
                # Keep only last few messages to manage context window
                recent_messages = messages[-3:] if len(messages) > 3 else messages
                augmented_messages = [system_message] + recent_messages
            else:
                status.write("No relevant documents found, using general knowledge...")
                # Keep only last few messages when no context
                augmented_messages = messages[-4:] if len(messages) > 4 else messages
            
            # Get response from LLM with context
            status.write("Generating response...")
            response = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-70B-Instruct-lora",
                messages=augmented_messages,
                max_tokens=800,  # Reduced from 1000
                temperature=0.7,
            )
            status.update(label="Response ready!", state="complete")
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
    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False
    if "show_login" not in st.session_state:
        st.session_state.show_login = False

def admin_login():
    """Handle admin login"""
    if "ADMIN_PASSWORD" not in st.secrets:
        st.sidebar.error("Admin password not configured.")
        return False
    
    with st.sidebar:
        st.subheader("Admin Login")
        password = st.text_input("Password", type="password", key="admin_password")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Login"):
                if password == st.secrets["ADMIN_PASSWORD"]:
                    st.session_state.is_admin = True
                    st.session_state.show_login = False
                    st.rerun()
                else:
                    st.error("Incorrect password")
        with col2:
            if st.button("Cancel"):
                st.session_state.show_login = False
                st.rerun()

def admin_logout():
    """Handle admin logout"""
    if st.sidebar.button("Logout Admin"):
        st.session_state.is_admin = False
        st.rerun()

def show_document_management():
    """Display document management interface for admins"""
    st.sidebar.header("Document Management")
    
    # Add file uploader in the sidebar
    key = f"file_uploader_{datetime.now().timestamp()}" if st.session_state.should_clear else "file_uploader"
    uploaded_file = st.sidebar.file_uploader(
        label="Upload a document",
        type=["txt", "md"],
        help="Upload a text or markdown file to include in the conversation",
        key=key,
        label_visibility="collapsed"
    )
    
    # Create a sidebar container for processing status
    status_container = st.sidebar.container()
    
    # Reset should_clear after creating new uploader
    if st.session_state.should_clear:
        st.session_state.should_clear = False
    
    # Process uploaded file
    if uploaded_file and not st.session_state.uploaded_file_processed:
        with status_container:
            progress_bar = st.progress(0)
            st.caption(f'Processing: {uploaded_file.name}')
            
            success = process_uploaded_file(uploaded_file)
            progress_bar.progress(100)
            
            if success:
                st.success('File processed successfully!')
                st.session_state.should_clear = True
                st.session_state.uploaded_file_processed = True
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

def main():
    check_api_key()
    st.set_page_config(page_title="AI Chat Assistant", page_icon="🤖")
    st.title("AI Chat Assistant 🤖")
    
    init_session_state()
    
    # Initialize states for file processing
    if "uploaded_file_processed" not in st.session_state:
        st.session_state.uploaded_file_processed = False
    if "should_clear" not in st.session_state:
        st.session_state.should_clear = False

    # Sidebar admin section
    with st.sidebar:
        if not st.session_state.is_admin:
            if st.button("Admin Login") or st.session_state.show_login:
                st.session_state.show_login = True
                admin_login()
        else:
            admin_logout()
            show_document_management()

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
        
        # Create a placeholder for the assistant's response
        with st.chat_message("assistant"):
            # Get and display assistant response
            assistant_response = get_assistant_response(st.session_state.messages)
            
            if assistant_response:
                st.write(assistant_response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            else:
                st.error("Failed to get response. Please try again.")

if __name__ == "__main__":
    main()