
from __future__ import annotations

import uuid

import requests
import streamlit as st

st.logo('./logo/logo.png', size='large')
# Configure page settings
st.set_page_config(
    page_title='Chat Application',
    page_icon='💬',
    layout='centered',
)

# Apply custom CSS for better styling
# Inside your st.markdown for custom CSS
st.markdown(
    """
<style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b313e;
        color: white;
    }
    .chat-message.assistant {
        background-color: #475063;
        color: white;
    }
    .chat-message .message-content {
        display: flex;
        margin-top: 0;
    }
    .message-content p {
        margin-bottom: 0;
    }
    .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
    }
    /* Add this to style citation links */
    .chat-message a {
        color: #1E90FF; /* Blue color for links, change to your desired color */
        text-decoration: none; /* Optional: remove underline */
    }
    .chat-message a:hover {
        color: #00BFFF; /* Lighter blue on hover, optional */
        text-decoration: underline; /* Optional: underline on hover */
    }
</style>
""", unsafe_allow_html=True,
)

# Set API endpoint
API_ENDPOINT = 'http://127.0.0.1:8000/chat/'
HEADERS = {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
}

# Initialize session state
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'user_input' not in st.session_state:
    st.session_state.user_input = ''

# Title for the app
st.title("💬 TYP's AI Assistant")



# Function to send message and get response


def send_message():
    """
    Handles user input, sends the message to the backend API, 
    and updates the chat history with both the user's message 
    and the assistant's response.

    Workflow:
    - Retrieves the user message from the session state.
    - Appends the user message to the chat history.
    - Sends the message along with a thread ID to the API endpoint.
    - Waits for the assistant's response from the API.
    - Adds the assistant's response to the chat history.
    - Handles and displays any errors that may occur during the request.
    """
    if st.session_state.user_input:
        user_message = st.session_state.user_input

        # Add user message to chat
        st.session_state.messages.append({'role': 'user', 'content': user_message})

        # Clear input field
        st.session_state.user_input = ''

        # Prepare data for API call
        data = {
            'message': user_message,
            'thread_id': st.session_state.thread_id,
        }

        try:
            response = requests.post(API_ENDPOINT, json=data, headers=HEADERS)

            if response.status_code == 200:
                assistant_response = response.json()
                # Add assistant response to chat
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': assistant_response['content'],
                })
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Failed to communicate with the server: {str(e)}")



# Add a sidebar with options
with st.sidebar:
    # st.title('Options')

    if st.button('New Conversation'):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()  # Thay st.experimental_rerun() bằng st.rerun() (phiên bản mới hơn)

    st.markdown('---')
    st.write('Current Thread ID:', st.session_state.thread_id)

# Function to display chat messages


def display_messages():
    """
    Display all chat messages stored in the current session.

    This function loops through the list of chat messages saved in `st.session_state.messages`
    and renders each one with a styled HTML container. Each message includes:
        - A corresponding avatar based on the role (user or assistant).
        - The text content of the message.

    Avatars are fetched using the Dicebear API:
        - User: personas style
        - Assistant: bottts style
    """
    for message in st.session_state.messages:
        role = message['role']
        content = message['content']

        avatar_url = 'https://api.dicebear.com/7.x/bottts/svg?seed=assistant' if role == 'assistant' else 'https://api.dicebear.com/7.x/personas/svg?seed=user'

        with st.container():
            st.markdown(
                f"""
                <div class="chat-message {role}">
                    <div class="message-content">
                        <img class="avatar" src="{avatar_url}">
                        <div>{content}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True,
            )


# Display existing chat messages
display_messages()
# Create a form for user input
with st.form(key='chat_form', clear_on_submit=True):
    st.text_input(
        'Your message:',
        key='user_input',
        placeholder='Type your message here...',
    )
    submit_button = st.form_submit_button('Send', on_click=send_message)