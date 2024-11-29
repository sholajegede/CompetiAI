import streamlit as st
from scrape import (
    scrape_website,
    extract_body_content,
    clean_body_content,
    split_dom_content,
)
from parse import parse_with_ollama

st.title("Competitive Intelligence Tool (Demo)")
url = st.text_input("Enter Competitor's Website URL")

if st.button("Gather Insights"):
    if url:
        st.write("Analyzing the website...")

        dom_content = scrape_website(url)
        body_content = extract_body_content(dom_content)
        cleaned_content = clean_body_content(body_content)

        st.session_state.dom_content = cleaned_content

        with st.expander("Review Extracted Content"):
            st.text_area("Extracted Content", cleaned_content, height=300)


if "dom_content" in st.session_state:
    parse_description = st.text_area("Specify the data or insights you want to extract")

    if st.button("Extract Insights"):
        if parse_description:
            st.write("Processing the content...")

            dom_chunks = split_dom_content(st.session_state.dom_content)
            parsed_result = parse_with_ollama(dom_chunks, parse_description)
            st.write(parsed_result)