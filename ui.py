"""Simple Streamlit UI for RAG service

Allows users to ask questions via web interface.
Queries the RAG API and displays answers + sources.
"""
import streamlit as st
import requests

st.set_page_config(page_title="RAG Query")

st.title("Query Your Documents")
st.write("Ask questions about your documents using semantic search + LLM")

# Sidebar configuration
with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", value="http://localhost:8000")
    num_sources = st.slider("Top sources to retrieve", 1, 5, 3)

# Main area
st.write("Type your question below:")
question = st.text_area("Question:", height=100)

if st.button("Search"):
    if not question:
        st.warning("Please enter a question")
    else:
        with st.spinner("Searching..."):
            try:
                # Call API
                response = requests.post(
                    f"{api_url}/query",
                    json={"question": question, "top_k": num_sources},
                    timeout=30,
                )

                if response.status_code == 200:
                    data = response.json()

                    # Show answer
                    st.subheader("Answer")
                    st.info(data["answer"])

                    # Show sources
                    st.subheader("Sources Found")
                    for i, source in enumerate(data["sources"]):
                        with st.expander(f"Source {i+1}"):
                            st.write(source["text"])

                    # Show model info
                    st.subheader("Model Info")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Model", data["model"].get("family", "?"))
                    with col2:
                        st.metric("Device", data["model"].get("device", "?"))

                else:
                    st.error(f"API Error: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error(f"Cannot connect to {api_url}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

