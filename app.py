import streamlit as st
from PIL import Image
import google.generativeai as genai
from gemini_helper import GeminiInspector

# Page configuration
st.set_page_config(
    page_title="Construction Site Inspector",
    page_icon="🏗️",
    layout="wide"
)

# Initialize session state
if 'chat' not in st.session_state:
    st.session_state.chat = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'image_analyzed' not in st.session_state:
    st.session_state.image_analyzed = False
if 'current_image' not in st.session_state:
    st.session_state.current_image = None

# Sidebar for API key
with st.sidebar:
    st.title("Settings")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    if api_key:
        inspector = GeminiInspector(api_key)
    else:
        inspector = GeminiInspector()
    
    # Add clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat = inspector.start_chat()
        st.session_state.image_analyzed = False
        st.rerun()

# Main title
st.title("🏗️ Construction Site Inspector")

# Initialize chat if not already done
if st.session_state.chat is None:
    st.session_state.chat = inspector.start_chat()

# File uploader
uploaded_file = st.file_uploader("Upload a construction site image", type=['png', 'jpg', 'jpeg'])

# Main interface
if uploaded_file:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Construction Site Image", use_container_width=True)
    
    # Analyze button
    if not st.session_state.image_analyzed:
        if st.button("Analyze Image", type="primary"):
            with st.spinner("Analyzing construction site..."):
                # Store image for reference
                st.session_state.current_image = image
                
                # Get initial analysis
                report = inspector.analyze_image(image, st.session_state.chat)
                
                # Add the report to chat history
                st.session_state.messages.append({"role": "assistant", "content": report})
                st.session_state.image_analyzed = True
                st.rerun()

# Display chat history
st.markdown("### 💬 Construction Analysis Chat")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if st.session_state.image_analyzed:
    if prompt := st.chat_input("Ask questions about the construction site..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = inspector.send_message(st.session_state.chat, prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# Footer with instructions
st.markdown("---")
st.markdown("""
### How to Use This App
1. Upload a construction site image using the file uploader above
2. Click "Analyze Image" to get an initial inspection report
3. Ask questions about specific aspects of the construction site
4. The AI will maintain context of the previous analysis while answering your questions
5. Use the "Clear Chat History" button in the sidebar to start fresh

Example questions you can ask:
- Can you explain more about the safety concerns you noticed?
- What stage of construction is this project in?
- What types of equipment do you see on site?
- Are there any quality control issues I should be aware of?
""")

# Download button for chat history
if st.session_state.messages:
    chat_history = "\n\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state.messages])
    st.download_button(
        "Download Conversation",
        chat_history,
        file_name="construction_analysis.txt",
        mime="text/plain"
    )