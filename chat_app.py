import streamlit as st
import os
import sys
import google.generativeai as genai
from datetime import datetime
import base64
import tempfile
import time
import requests

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Update the import to use the simplified document processor
from utils.simple_document_processor import SimpleDocumentProcessor

def check_api_key():
    if "GEMINI_API_KEY" not in st.secrets:
        st.error("Please set the GEMINI_API_KEY in your Streamlit secrets.")
        st.stop()
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

def init_gemini_model():
    return genai.GenerativeModel("gemini-2.0-flash")

def handle_rate_limit_error(e):
    """Handle rate limit errors from Gemini API"""
    # Gemini API uses different rate limiting mechanism
    # This is a simplified handler for demonstration
    retry_after = 5  # Default retry after 5 seconds

    if hasattr(e, 'status_code') and e.status_code == 429:
        st.warning(f"Rate limit reached. Please wait {retry_after} seconds before trying again.")
        time.sleep(retry_after)
        return True

    return False

def get_assistant_response(messages):
    model = init_gemini_model()
    try:
        user_message = messages[-1]["content"]
        status_placeholder = st.empty()

        with status_placeholder:
            with st.status("✨ Processing your request...") as status:
                # Extract keywords from user message
                keywords = [word.strip() for word in user_message.split() if len(word.strip()) > 1]
                status.write(f"Keywords extracted: {', '.join(keywords)}")

                # Search for content using the keywords - don't show notifications for regular users
                doc_processor = SimpleDocumentProcessor(show_notifications=False)
                all_chunks = []

                for keyword in keywords:
                    chunks = doc_processor.query_similar(keyword)
                    if chunks:
                        all_chunks.extend(chunks)

                if all_chunks:
                    status.write("Found relevant content...")
                    # Remove duplicates while preserving order
                    seen = set()
                    unique_chunks = []
                    for chunk in all_chunks:
                        if chunk not in seen:
                            unique_chunks.append(chunk)
                            seen.add(chunk)

                    # Combine chunks (limit to 10 most relevant)
                    combined_chunks = "\n\n".join(unique_chunks[:10])

                    # Create system message with context
                    system_prompt = f"""You are a helpful assistant that answers questions based on the provided context.
Use the information from the context below to answer the user's question: "{user_message}"

Context:
{combined_chunks}

Instructions:

1. ตอบตามข้อมูลในเอกสารเป็นหลัก
2. ให้คำตอบที่ชัดเจน ลึก ตรงประเด็น ช่วยคลายความสงสัย
3. ให้ตัด ให้ลดความคิดปรุงแต่ง และความทุกข์ของผู้ถาม
4. ตอบอย่าใช้คำว่า ตามข้อมูลที่ให้มา ไม่ให้ใช้คำว่าเอกสาร ให้ตอบข้อมูลในเอกสารไปตรงๆ
5. ถ้าถามเกี่ยวกับ มหากรุณา ให้ตอบว่า ให้ฟังสัจธรรมวัดร่มโพธิธรรม จ.เลย"""

                    # Format conversation for Gemini
                    # Convert messages to Gemini format
                    gemini_messages = []

                    # Add system prompt as first user message
                    gemini_messages.append({"role": "user", "parts": [system_prompt]})
                    gemini_messages.append({"role": "model", "parts": ["I'll help answer based on the context provided."]})

                    # Add recent conversation history
                    recent_messages = messages[-2:] if len(messages) > 2 else messages
                    for msg in recent_messages:
                        role = "user" if msg["role"] == "user" else "model"
                        gemini_messages.append({"role": role, "parts": [msg["content"]]})

                else:
                    status.write("No relevant documents found, using general knowledge...")
                    # Convert messages to Gemini format
                    gemini_messages = []
                    recent_messages = messages[-3:] if len(messages) > 3 else messages
                    for msg in recent_messages:
                        role = "user" if msg["role"] == "user" else "model"
                        gemini_messages.append({"role": role, "parts": [msg["content"]]})

                status.write("Generating response...")
                try:
                    # Create a chat session
                    chat = model.start_chat(history=gemini_messages[:-1])

                    # Generate response
                    response = chat.send_message(
                        gemini_messages[-1]["parts"][0],
                        generation_config={
                            "max_output_tokens": 500,
                            "temperature": 0.7,
                            "top_p": 0.9
                        }
                    )

                except Exception as e:
                    if "429" in str(e):  # Rate limit exceeded
                        if handle_rate_limit_error(e):
                            # If rate limited, retry after waiting
                            return get_assistant_response(messages)
                        return "Rate limit exceeded. Please try again in a few moments."
                    raise e

        status_placeholder.empty()
        return response.text

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

        # Process the file using SimpleDocumentProcessor - show notifications for admin users
        doc_processor = SimpleDocumentProcessor(show_notifications=True)
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
        # Initialize session state variables if they don't exist
        if "login_error" not in st.session_state:
            st.session_state.login_error = False
        if "clear_password" not in st.session_state:
            st.session_state.clear_password = False
        if "pw_key" not in st.session_state:
            st.session_state.pw_key = "admin_password_0"

        # Show error message if login failed
        if st.session_state.login_error:
            st.error("Incorrect password")
            st.session_state.login_error = False

        # Create a container for login elements
        if st.session_state.clear_password:
            st.session_state.clear_password = False
            st.session_state.pw_key = f"admin_password_{datetime.now().timestamp()}"

        password = st.text_input("Password", type="password", key=st.session_state.pw_key)

        # Check for Enter key press or button click
        if st.button("Login") or (password and password.strip() != ""):
            # Convert both passwords to strings and strip whitespace
            entered_pass = str(password).strip()
            stored_pass = str(st.secrets["ADMIN_PASSWORD"]).strip()

            if entered_pass == stored_pass:
                if "is_admin" not in st.session_state:
                    st.session_state.is_admin = False
                st.session_state.is_admin = True
                st.session_state.show_login = False
                st.success("Login successful!")
                st.rerun()
            else:
                # Set flag for error message and clear password
                st.session_state.login_error = True
                st.session_state.clear_password = True
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
    doc_processor = SimpleDocumentProcessor()
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
    """Reindex all documents in the database."""
    try:
        doc_processor = SimpleDocumentProcessor(show_notifications=True)
        uploaded_files = doc_processor.get_uploaded_files()

        # Initialize the progress bar
        progress_bar = st.progress(0)

        total_files = len(uploaded_files)

        # Create a placeholder for the current file message
        current_file_placeholder = st.empty()

        for index, file in enumerate(uploaded_files):
            # Display the current file being indexed
            current_file_placeholder.text(f"Currently indexing: {file}")  # Show the current file name

            # Fetch the document content from database
            document_chunks = doc_processor.db.get_document_chunks()

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

def sync_database():
    """Sync local database with Supabase."""
    try:
        doc_processor = SimpleDocumentProcessor(show_notifications=True)

        # Check if Supabase is available
        if not doc_processor.db.supabase_available:
            st.error("Supabase is not available. Cannot sync.")
            return

        # Create progress indicators
        status_placeholder = st.empty()
        progress_bar = st.progress(0)

        with status_placeholder.container():
            st.write("Syncing with Supabase...")

            # First sync from Supabase to local
            st.write("Step 1: Importing data from Supabase...")
            progress_bar.progress(25)
            doc_processor.import_from_supabase()

            # Then sync from local to Supabase
            st.write("Step 2: Exporting local changes to Supabase...")
            progress_bar.progress(75)
            doc_processor.sync_database()

            # Complete
            progress_bar.progress(100)
            st.success("Sync completed!")

            # Get current sync status
            status = doc_processor.db.get_sync_status()
            st.info(f"Last sync: {datetime.fromtimestamp(status['last_sync']).strftime('%Y-%m-%d %H:%M:%S')}")
            st.info(f"Status: {status['status']}")

    except Exception as e:
        st.error(f"Error during sync: {str(e)}")

def main():
    check_api_key()  # Now checks for GEMINI_API_KEY
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

            # Add database status and sync options
            st.sidebar.header("Database Management")

            # Create a document processor to check database status - show notifications for admin
            doc_processor = SimpleDocumentProcessor(show_notifications=True)
            status = doc_processor.db.get_sync_status()

            # Display database status
            if status['supabase_available']:
                st.sidebar.success("Supabase: Connected")
            else:
                st.sidebar.warning("Supabase: Not available (using local database)")

            # Display last sync time if available
            if status['last_sync'] > 0:
                last_sync_time = datetime.fromtimestamp(status['last_sync']).strftime('%Y-%m-%d %H:%M:%S')
                st.sidebar.info(f"Last sync: {last_sync_time}")
            else:
                st.sidebar.info("Never synced with Supabase")

            # Add sync and reindex buttons
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("Sync Database"):
                    sync_database()
            with col2:
                if st.button("Reindex Docs"):
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