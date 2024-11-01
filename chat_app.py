import streamlit as st
import os
from together import Together
from datetime import datetime

# Set your Together AI API key
os.environ["TOGETHER_API_KEY"] = st.secrets["TOGETHER_API_KEY"]

def init_together_client():
    return Together()

def get_assistant_response(messages):
    client = init_together_client()
    try:
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-lora",
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def main():
    st.set_page_config(page_title="AI Chat Assistant", page_icon="🤖")
    st.title("AI Chat Assistant 🤖")
    
    init_session_state()
    
    # Display chat messages
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        # Use Streamlit's chat_message component
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