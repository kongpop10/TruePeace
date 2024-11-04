import streamlit as st
import os
import sys
from together import Together
from datetime import datetime
import base64 
import tempfile

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
        
        # Create a single container for all status messages
        status_placeholder = st.empty()
        
        with status_placeholder:
            with st.status("✨ Processing your request...") as status:
                # Get relevant document chunks
                status.write("Searching relevant context...")
                doc_processor = DocumentProcessor()
                relevant_chunks = doc_processor.query_similar(user_message)
                
                # If we found relevant chunks, include them in the context
                if relevant_chunks:
                    status.write("Found relevant context...")
                    # More aggressive token management - limit to ~4000 chars (~1000 tokens)
                    combined_chunks = ""
                    for chunk in relevant_chunks:
                        if len(combined_chunks) + len(chunk) < 4000:
                            combined_chunks += chunk + "\n\n"
                        else:
                            break
                    
                    system_message = {
                        "role": "system",
                        "content": f"""ใช้ข้อมูลต่อไปนี้ในการตอบคำถาม:

{combined_chunks}

คำแนะนำในการตอบ:
1. ตอบโดยอ้างอิงจากข้อมูลในเอกสารเป็นหลัก
2. ให้คำตอบที่ชัดเจน ตรงประเด็น ช่วยคลายความสงสัย
3. ตอบในแนวทางที่ช่วยลดความคิดปรุงแต่ง และความทุกข์ของผู้ถาม
4. ใช้ประโยคสั้นๆ ตอบคำถาม ประโยคที่ล้างกันในประโยค คำถามที่ล้างกันในคำถาม
"""
                    }
                    # Keep only last 2 messages to reduce context
                    recent_messages = messages[-2:] if len(messages) > 2 else messages
                    augmented_messages = [system_message] + recent_messages
                else:
                    status.write("No relevant documents found, using general knowledge...")
                    # Keep only last 3 messages when no context
                    augmented_messages = messages[-3:] if len(messages) > 3 else messages
                
                # Get response from LLM with context
                status.write("Generating response...")
                response = client.chat.completions.create(
                    model="meta-llama/Meta-Llama-3.1-70B-Instruct-lora",
                    messages=augmented_messages,
                    max_tokens=500,  # Reduced from 800
                    temperature=0.7,
                )
        
        # Clear the status messages
        status_placeholder.empty()
        
        return response.choices[0].message.content
                
    except Exception as e:
        if 'status_placeholder' in locals():
            status_placeholder.empty()
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
    
    with st.sidebar.expander("Admin Login"):
        # Create a container for login elements
        password = st.text_input("Password", type="password", key="admin_password")
        
        # Check for Enter key press or button click
        if st.button("Login") or (password and st.session_state.get("admin_password", "") != ""):
            # Convert both passwords to strings and strip whitespace
            entered_pass = str(password).strip()
            stored_pass = str(st.secrets["ADMIN_PASSWORD"]).strip()
            
            if entered_pass == stored_pass:
                st.session_state.is_admin = True
                st.session_state.show_login = False
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Incorrect password")
                # Clear the password field after failed attempt
                st.session_state.admin_password = ""

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

def display_message(role: str, content: str):
    """Display a chat message with custom styled icons"""
    # Add custom CSS for avatar styling
    st.markdown("""
        <style>
        /* Style for AI assistant avatar */
        [data-testid="chat-message-avatar-assistant"] {
            background-color: #ffd70020 !important;
            padding: 5px !important;
            border-radius: 50% !important;
        }
        
        /* Style for user avatar */
        [data-testid="chat-message-avatar-user"] {
            background-color: #ff8c0020 !important;
            padding: 5px !important;
            border-radius: 50% !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Set icons with custom colors
    icon = "☀️" if role == "assistant" else "🕯️"  # Lotus for AI, sparkles for user
    with st.chat_message(role, avatar=icon):
        st.write(content)

def reindex_documents():
    """Reindex all documents in the Supabase database."""
    try:
        doc_processor = DocumentProcessor()
        uploaded_files = doc_processor.get_uploaded_files()
        
        # Initialize the progress bar
        progress_bar = st.progress(0)
        
        total_files = len(uploaded_files)
        
        # Create a placeholder for the current file message
        current_file_placeholder = st.empty()
        
        for index, file in enumerate(uploaded_files):
            # Display the current file being indexed
            current_file_placeholder.text(f"Currently indexing: {file}")  # Show the current file name
            
            # Fetch the document content from Supabase
            document_chunks = doc_processor.supabase.get_document_chunks()
            
            # Find the content for the current file
            document_content = ""
            for chunk in document_chunks:
                if chunk['id'].startswith(file):
                    document_content += chunk['content'] + "\n"
            
            if document_content:
                # Create a temporary file to process the content
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
                    temp_file.write(document_content.encode('utf-8'))
                    temp_file_path = temp_file.name
                
                # Process the temporary file to reindex
                doc_processor.process_file(temp_file_path, file)
                
                # Update the progress bar
                progress = (index + 1) / total_files
                progress_bar.progress(progress)
                
                # Optionally delete the temporary file after processing
                os.remove(temp_file_path)
            else:
                st.warning(f"No content found for {file}. Skipping.")
        
        # Complete the progress bar
        progress_bar.progress(1.0)  # Set to 100%
        st.toast("Reindexing completed successfully!", icon="✅")
        
        # Clear the current file message
        current_file_placeholder.empty()  # Hide the current file message
        
        # Clear the progress bar
        progress_bar.empty()  # Hide the progress bar
    except Exception as e:
        st.toast(f"Error during reindexing: {str(e)}", icon="⚠️")

def main():
    check_api_key()
    st.set_page_config(
        page_title="Beyond Path",
        page_icon="☀️",
        initial_sidebar_state="collapsed",
        layout="centered"
    )
    
    # Add custom CSS to style the menu button and center content
    st.markdown("""
        <style>
        #MainMenu {visibility: visible;}
        [data-testid="collapsedControl"] {
            display: none;
        }
        .stDeployButton {
            display: none;
        }
        header {visibility: hidden;}
        
        /* Center the title container */
        .title-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 1rem 0 2rem 0;
            text-align: center;
        }
        
        /* Style the lotus emoji */
        .lotus-emoji {
            font-size: 4rem;
            line-height: 1.2;
            margin-bottom: 0.5rem;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 5rem;
        }
        
        /* Style the title text */
        .title-text {
            font-size: 2rem;
            font-weight: 500;
            margin: 0;
            background: linear-gradient(120deg, #ff9a9e 0%, #fad0c4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        /* Dark mode support */
        @media (prefers-color-scheme: dark) {
            .title-text {
                background: linear-gradient(120deg, #fad0c4 0%, #ff9a9e 100%);
                -webkit-background-clip: text;
                background-clip: text;
            }
        }

        /* Favicon */
        link[rel="shortcut icon"] {
            content: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E☀️%3C/text%3E%3C/svg%3E");
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Custom title with lotus image
    def get_base64_image(image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return encoded_string

    # Convert the image and embed it in HTML
    image_base64 = get_base64_image("assets/lotus.png")
    st.markdown(f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{image_base64}" alt="Lotus" style="width: 120px; height: auto; margin-left: -30px;"/>
            <h1>Beyond Path</h1>
    </div>
    """, unsafe_allow_html=True)
    
    init_session_state()
    
    # Initialize states for file processing
    if "uploaded_file_processed" not in st.session_state:
        st.session_state.uploaded_file_processed = False
    if "should_clear" not in st.session_state:
        st.session_state.should_clear = False

    # Sidebar admin section
    with st.sidebar:
        if not st.session_state.is_admin:
            admin_login()
        else:
            admin_logout()
            show_document_management()
            
            # Add a button for reindexing documents
            if st.button("Reindex Documents"):
                reindex_documents()

    # Display chat messages
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        display_message(role, content)

    # User input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Display user message
        display_message("user", user_input)
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Create a placeholder for the assistant's response
        # Get and display assistant response
        assistant_response = get_assistant_response(st.session_state.messages)
        
        if assistant_response:
            display_message("assistant", assistant_response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        else:
            st.error("Failed to get response. Please try again.")

if __name__ == "__main__":
    main()