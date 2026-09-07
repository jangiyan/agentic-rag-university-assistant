
import streamlit as st
from main import app


# Page Settings
st.set_page_config(
    page_title="University AI Assistant",
    page_icon="🎓"
)


# Title
st.title("🎓 University AI Assistant")

st.write(
    "Ask questions about university policies and documents."
)


# User Input
question = st.text_input(
    "Ask your question:"
)


# Button
if st.button("Ask"):

    if question:

        with st.spinner("Thinking..."):

            result = app.invoke(
                {
                    "question": question,
                    "decision": "",
                    "answer": ""
                }
            )

            st.subheader("Answer")

            st.write(
                result["answer"]
            )

    else:

        st.warning(
            "Please enter a question."
        )
