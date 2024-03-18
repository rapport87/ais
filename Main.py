import streamlit as st

st.set_page_config(
    page_title="AI Main page",
    page_icon="🤖",
)

st.title("AI (ChatGPT-3.5)")

st.markdown(
    """
AI 메인 페이지에 오신것을 환영합니다. 

AI를 사용하여 편하게 일상 생활, 혹은 업무에 활용하세요!


----------------------------------------

Terms AI
약관(문서)를 업로드하고 해당 문서에 대한 내용을 AI가 학습하여 사용자의 질문에 대한 답변을 제공합니다.

Teacher AI 
문서를 업로드하거나 검색한 단어로 문제를 만들어 제공합니다. AI가 제출한 문제를 사용자가 풀고 AI가 채점을 해주는 기능도 제공합니다.

Video AI 
영상을 업로드하고 해당 영상의 내용을 AI가 학습하여 사용자의 질문에 대한 답변을 제공합니다.

- [약관AI](/TermsAI)
- [선생님AI](/TermsAI)
- [비디오AI](/TermsAI)
"""
)
