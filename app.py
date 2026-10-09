# AI Resume Analyzer Project

import streamlit as st
import pypdf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# page title
st.set_page_config(page_title="My Resume Analyzer")
st.title("AI Resume Analyzer")
st.write("This project checks resume and tells best job for you")
st.write("---")

# upload part
pdf_file = st.file_uploader("Upload your Resume PDF here", type=["pdf"])

# my job list - i added finance jobs
my_jobs = {
    "Finance Analyst": "finance excel financial analysis budgeting forecasting accounting tally",
    "Business Analyst": "business analyst excel sql power bi data analysis communication",
    "Accountant": "accounting tally gst tax audit excel financial statement",
    "Banking Job": "banking finance excel communication customer handling",
    "Data Analyst": "excel sql data analysis python power bi"
}

# what skills needed for each job
skills_for_job = {
    "Finance Analyst": ["excel", "finance", "budgeting"],
    "Business Analyst": ["excel", "sql", "power bi"],
    "Accountant": ["tally", "gst", "accounting"],
    "Banking Job": ["finance", "communication"],
    "Data Analyst": ["excel", "sql", "python"]
}

if pdf_file is not None:
    # reading pdf
    pdf_reader = pypdf.PdfReader(pdf_file)
    full_text = ""
    for page in pdf_reader.pages:
        txt = page.extract_text()
        if txt:
            full_text = full_text + txt

    full_text_small = full_text.lower() # for checking

    st.subheader("Your Resume Text:")
    st.write(full_text[:1500]) # showing only 1500 chars

    # matching logic
    all_docs = [full_text_small] + list(my_jobs.values())
    vector = TfidfVectorizer()
    tfidf_matrix = vector.fit_transform(all_docs)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    st.write("---")
    st.subheader("Job Matching Result:")

    # simple bar chart using streamlit
    import pandas as pd
    score_list = []
    for s in similarity_scores:
        score_list.append(round(s*100, 1))

    df = pd.DataFrame({"Job": list(my_jobs.keys()), "Score": score_list})
    st.bar_chart(df.set_index("Job"))

    # showing scores one by one
    for i in range(len(my_jobs)):
        job_name = list(my_jobs.keys())[i]
        score = similarity_scores[i]*100
        if score > 20:
            st.success(f"{job_name} - {score:.1f}% - Good Match for you")
        elif score > 5:
            st.info(f"{job_name} - {score:.1f}% - Average Match")
        else:
            st.write(f"{job_name} - {score:.1f}% - Low Match")

    # finding best job
    best_index = similarity_scores.argmax()
    best_job_name = list(my_jobs.keys())[best_index]
    st.write("---")
    st.subheader(f"Best Job for You: {best_job_name}")

    # skill gap
    st.write(f"Skills needed for {best_job_name}:")
    needed_skills = skills_for_job[best_job_name]
    for sk in needed_skills:
        if sk in full_text_small:
            st.write(f"✅ You have {sk}")
        else:
            st.write(f"❌ Missing {sk} - you should learn this")

    # my own suggestions
    st.subheader("My Suggestions:")
    if "excel" not in full_text_small:
        st.write("- Please add Excel skill, very important for finance")
    if "power bi" not in full_text_small:
        st.write("- Learn Power BI, it will help for analyst job")
    if len(full_text.split()) < 300:
        st.write("- Your resume is short, add more about internship")

    st.balloons()

else:
    st.info("Please upload PDF file to start")