import streamlit as st
import hashlib

from main import process_cv, app

st.set_page_config(
    page_title="AI Job Assistant",
    page_icon="💼",
    layout="wide"
)

st.markdown("""
<style>

.main {
    padding: 2rem;
}

.title {
    font-size: 42px;
    font-weight: 700;
}

.subtitle {
    font-size: 18px;
    color: gray;
    margin-bottom: 30px;
}

.job-card {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e6e6e6;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)

if "cv_uploaded" not in st.session_state:
    st.session_state.cv_uploaded = False

if "cv_hash" not in st.session_state:
    st.session_state.cv_hash = None

with st.sidebar:

    st.title("💼 AI Job Assistant")

    st.divider()

    st.markdown("### How it works")

    st.write("""
1. Upload your CV

2. AI analyzes your skills

3. Ask for suitable jobs

4. AI searches available jobs

5. Get personalized recommendations
""")

    st.divider()

    if st.session_state.cv_uploaded:

        st.success("Current CV Processed Successfully")

    else:

        st.warning("No CV Uploaded")

st.markdown(
    '<div class="title">AI Career Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '''
    <div class="subtitle">
    Upload your CV and discover jobs that match your
    skills and experience.
    </div>
    ''',
    unsafe_allow_html=True
)

st.subheader("Upload Your CV")

uploaded_file = st.file_uploader(
    "Upload CV (PDF only)",
    type=["pdf"]
)

if uploaded_file is not None:

    # File content read karo
    file_bytes = uploaded_file.getvalue()

    # Har CV ka unique hash
    current_cv_hash = hashlib.md5(
        file_bytes
    ).hexdigest()


    # Check karo kya NEW CV upload hui hai
    if current_cv_hash != st.session_state.cv_hash:

        # New CV detected
        st.session_state.cv_uploaded = False

        with st.spinner("Analyzing new CV..."):

            # File pointer reset
            uploaded_file.seek(0)

            result = process_cv(
                uploaded_file
            )


        if result["success"]:

            # New CV successfully processed
            st.session_state.cv_uploaded = True

            st.session_state.cv_hash = current_cv_hash

            st.success(
                f'New CV processed successfully! '
                f'{result["chunks"]} chunks created.'
            )

        else:

            st.session_state.cv_uploaded = False

            st.error(
                result["message"]
            )


st.divider()

st.subheader("Ask Your Career Assistant")


user_request = st.chat_input(
    "How can I help you?"
)


if user_request:

    if not st.session_state.cv_uploaded:

        st.warning(
            "Please upload your CV first."
        )

    else:

        # USER MESSAGE
        with st.chat_message("user"):

            st.write(user_request)


        # AI MESSAGE
        with st.chat_message("assistant"):

            with st.spinner(
                "AI is analyzing your CV and searching for jobs..."
            ):

                result = app.invoke(
                    {
                        "user_request": user_request
                    }
                )

            answer = result["answer"]

            st.write(answer)