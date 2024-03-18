import json
import streamlit as st
from langchain.schema import BaseOutputParser, output_parser
from langchain.retrievers import WikipediaRetriever
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()

st.set_page_config(page_title="Teacher AI", page_icon="❓")

st.title("Teacher AI")

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


question_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            당신은 문제를 제출하는 선생님 도우미 입니다.

            다음 context를 사용하여 10개의 문제를 만드세요

            문제와 답안은 한글로 작성되어야 합니다

            각 질문에는 4개의 답안이 있어야 하며, 그 중 세 개는 틀린 답이어야 하고 하나만 정답이어야 합니다.

            정답의 끝에는 '(o)'를 사용하세요.

            질문 예시:

            질문: 바다는 무슨 색 인가요?
            답변: 빨강|노랑|초록|파랑(o)

            질문: 한국의 수도는 어디인가요?
            답변: 바쿠|서울(o)|마닐라|베이루트

            질문: 영화 '세얼간이'는 언제 개봉되었나요?
            답변: 1998|2001|2007|2011(o)

            질문: '모차르트'의 직업은 무엇인가요?
            답변: 작곡가(o)|화가|배우|모델
                
            Context: {context}
            """,
        )
    ]
)

questions_chain = {"context": format_docs} | question_prompt | llm

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    당신은 문제를 JSON 형태로 형식화하는 도우미 입니다
     
    시험 문제를 JSON 형식으로 형식화합니다.
    (o)가 있는 답은 정답입니다.

    예제 입력:
    질문: 바다는 무슨색인가요?
    답변: 빨강|노랑|초록|파랑(o)

    질문: 한국의 수도는 어디인가요?
    답변: 바쿠|서울(o)|마닐라|베이루트

    질문: 영화 '세얼간이'는 언제 개봉되었나요?
    답변: 1998|2001|2007|2011(o)

    질문: '모차르트'의 직업은 무엇인가요?
    답변: 작곡가(o)|화가|배우|모델
    
     
    예제 출력:
     
    ```json
    {{ "questions": [
            {{
                "question": "바다는 무슨색인가요??",
                "answers": [
                        {{
                            "answer": "빨강",
                            "correct": false
                        }},
                        {{
                            "answer": "노랑",
                            "correct": false
                        }},
                        {{
                            "answer": "초록",
                            "correct": false
                        }},
                        {{
                            "answer": "파랑",
                            "correct": true
                        }}
                ]
            }},
                        {{
                "question": "한국의 수도는 어디인가요?",
                "answers": [
                        {{
                            "answer": "바쿠",
                            "correct": false
                        }},
                        {{
                            "answer": "서울",
                            "correct": true
                        }},
                        {{
                            "answer": "마닐라",
                            "correct": false
                        }},
                        {{
                            "answer": "베이루트",
                            "correct": false
                        }}
                ]
            }},
                        {{
                "question": "아바타는 언제 개봉되었나요?",
                "answers": [
                        {{
                            "answer": "1998",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2011",
                            "correct": true
                        }}
                ]
            }},
            {{
                "question": "'모차르트'의 직업은 무엇인가요?",
                "answers": [
                        {{
                            "answer": "작곡가",
                            "correct": true
                        }},
                        {{
                            "answer": "화가",
                            "correct": false
                        }},
                        {{
                            "answer": "배우",
                            "correct": false
                        }},
                        {{
                            "answer": "모델",
                            "correct": false
                        }}
                ]
            }}
        ]
     }}
    ```
    Questions: {context}
""",
        )
    ]
)

formatting_chain = formatting_prompt | llm


@st.cache_data(show_spinner="파일을 읽는중입니다...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/teacherai/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="문제를 생성중입니다...")
def run_quiz_chain(_docs, topic):
    chain = {"context": questions_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)


@st.cache_data(show_spinner="위키피디아를 검색중입니다...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs


with st.sidebar:
    docs = None
    topic = None
    choice = st.selectbox(
        "사용할 방식을 선택하세요",
        (
            "파일",
            "위키피디아 검색",
        ),
    )
    if choice == "파일":
        file = st.file_uploader(
            "파일을 선택하세요(pdf, txt, docx)",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("검색어를 입력하세요")
        if topic:
            docs = wiki_search(topic)

if not docs:
    st.markdown(
        """
    안녕하세요!
                
    파일을 업로드 하거나 위키피디아를 검색해서,

    문제를 생성해주는 TeacherAI(선생님AI) 입니다

    사이드바에서 파일을 선택하여 파일을 업로드 하거나,

    위키피디아를 선택하여 검색어를 입력하세요
    """
    )
else:
    response = run_quiz_chain(docs, topic if topic else file.name)
    with st.form("questions_form"):
        for idx, question in enumerate(response["questions"]):
            st.write(question["question"])
            value = st.radio(
                "보기를 선택하세요",
                [answer["answer"] for answer in question["answers"]],
                key=f"{idx}_radio",
                index=None,
            )
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("정답입니다!")
            elif value is not None:
                st.error("오답입니다")
        button = st.form_submit_button()
