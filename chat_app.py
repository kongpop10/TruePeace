import streamlit as st
import os
import sys
from groq import Groq
from datetime import datetime
import base64 
import tempfile
import time
import requests

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Update the import to use the correct path
from utils.document_processor import DocumentProcessor

def check_api_key():
    if "GROQ_API_KEY" not in st.secrets:
        st.error("Please set the GROQ_API_KEY in your Streamlit secrets.")
        st.stop()
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

def init_groq_client():
    return Groq()

def handle_rate_limit_error(response_headers):
    """Handle rate limit information from Groq API headers"""
    retry_after = int(response_headers.get('retry-after', 0))
    remaining_requests = int(response_headers.get('x-ratelimit-remaining-requests', 0))
    remaining_tokens = int(response_headers.get('x-ratelimit-remaining-tokens', 0))
    
    if retry_after > 0:
        st.warning(f"Rate limit reached. Please wait {retry_after} seconds before trying again.")
        time.sleep(retry_after)
        return True
    
    if remaining_requests == 0 or remaining_tokens == 0:
        reset_requests = response_headers.get('x-ratelimit-reset-requests', '0s')
        reset_tokens = response_headers.get('x-ratelimit-reset-tokens', '0s')
        st.warning(f"Rate limit approaching. Requests reset in: {reset_requests}, Tokens reset in: {reset_tokens}")
    
    return False

def get_assistant_response(messages):
    client = init_groq_client()
    try:
        user_message = messages[-1]["content"]
        status_placeholder = st.empty()
        
        with status_placeholder:
            with st.status("✨ Processing your request...") as status:
                status.write("Searching relevant context...")
                doc_processor = DocumentProcessor()
                relevant_chunks = doc_processor.query_similar(user_message)
                
                if relevant_chunks:
                    status.write("Found relevant context...")
                    combined_chunks = ""
                    for chunk in relevant_chunks:
                        if len(combined_chunks) + len(chunk) < 4000:
                            combined_chunks += chunk + "\n\n"
                        else:
                            break
                    
                    # Create a more explicit system message with instructions to use the context
                    system_message = {
                        "role": "system",
                        "content": """You are a helpful assistant that answers questions based on the provided context. 
Always use the context below to formulate your responses. If the information isn't in the context, 
say you don't have enough information to answer accurately.

Context:
""" + combined_chunks + """

Instructions for responding:
1. ตอบตามข้อมูลในเอกสารเป็นหลัก
2. ให้คำตอบที่ชัดเจน ลึก ตรงประเด็น ช่วยคลายความสงสัย
3. ให้ตัด ให้ลดความคิดปรุงแต่ง และความทุกข์ของผู้ถาม
4. ตอบอย่าใช้คำว่า ตามข้อมูลที่ให้มา อย่าใช้คำว่าเอกสาร ให้ตอบไปตรงๆ 
5. ถ้าถามเกี่ยวกับ มหากรุณา ให้ตอบว่า ให้ฟังสัจธรรมวัดร่มโพธิธรรม จ.เลย"""
                    }
                    
                    # Format conversation history
                    conversation = [system_message]
                    
                    # Add a reminder of the context before the user's question
                    conversation.append({
                        "role": "system",
                        "content": "Remember to answer based on the context provided above."
                    })
                    
                    # Add recent conversation history
                    recent_messages = messages[-2:] if len(messages) > 2 else messages
                    conversation.extend(recent_messages)
                    
                else:
                    status.write("No relevant documents found, using general knowledge...")
                    conversation = messages[-3:] if len(messages) > 3 else messages
                
                status.write("Generating response...")
                try:
                    response = client.chat.completions.create(
                        model="llama-3.1-70b-versatile",
                        messages=conversation,
                        max_tokens=500,
                        temperature=0.7,
                        top_p=0.9,
                        stream=False
                    )
                    
                    # Check rate limit headers from response
                    if hasattr(response, 'headers'):
                        if handle_rate_limit_error(response.headers):
                            # If rate limited, retry after waiting
                            return get_assistant_response(messages)
                            
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:  # Rate limit exceeded
                        handle_rate_limit_error(e.response.headers)
                        return "Rate limit exceeded. Please try again in a few moments."
                    raise e
                
        status_placeholder.empty()
        return response.choices[0].message.content.replace("assistant", "")
                
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
        processed_chunks = doc_processor.process_file(temp_file_path, uploaded_file.name)
        
        if processed_chunks:
            st.success('File processed successfully!')
        else:
            st.warning('No chunks were processed from the uploaded file.')
        
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
        # Only display content for assistant messages without the role
        if role == "assistant":
            st.write(content)  # Display content without the role
        else:
            st.write(content)  # Display content for user

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
                st.toast(f"No content found for {file}. Skipping.", icon="⚠️")
        
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