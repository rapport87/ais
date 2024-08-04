# 🤖AI's
Langchain을 이용하여 
약관등의 길고 원하는 내용을 찾기 힘든것을 보완하기 위해 문서의 내용을 검색하여 요약해주는 Term AI,
문서를 읽어 4가지 선택지를 가진 문제를 제출해주는 Teacher AI,
영상을 읽어들여 텍스트 파일로 변환한 뒤 질문받은 내용을 답변하는 Video AI를 구현해본 프로젝트 입니다

- Service URL : http://gpt.learninglab.co.kr:40100/ (가정에서 소규모 서버로 가동중이라 많이 느립니다)

## 🗓️개발기간
**2023년 7월 ~ 9월**

## 👨‍💻개발 인원(1인)
- **이 한결**


## 📜Term AI(약관AI)
**설명**
- 영상파일을 업로드 받아 해당 영상의 내용을 학습해 사용자의 질문에 답변해주는 비디오AI 프로젝트

**주요기능**
- 문서 파일 읽기
- ChatGPT 모델을 이용한 답변 생성 및 제공

**Detail**
- URL : https://ikvi.notion.site/AI-7eca101b8c5542e7880493b1e9ad70fd


## 👨‍🏫Teacher AI(선생님AI)
**설명**
- 문서 파일을 업로드 하거나 검색어를 입력받아 해당 내용에 관련된 문제를 출제해주는 선생님AI 프로젝트

**주요기능**
- 문서파일 업로드
- 위키피디아 검색 및 검색 내용 수집
- 문제 생성, 생성된 문제를 JSON형태로 변환
- JSON 형태로 변환된 문제를 Form으로 변환하여 사용자에게 제공

**Detail**
- URL : https://ikvi.notion.site/AI-a0cefbc3e6fc4a708d585ecc9289deff?pvs=4


## 📼Video AI(비디오AI)
**설명**
- 영상파일을 업로드 받아 해당 영상의 내용을 학습해 사용자의 질문에 답변해주는 비디오AI 프로젝트

**주요기능**
- 영상파일 업로드
- 영상파일에서 음성만 분할하여 추출
- 분할하여 추출된 음성파일을 하나의 텍스트파일로 변환
- ChatGPT 모델을 이용한 답변 생성 및 제공

**Detail**
- URL : https://ikvi.notion.site/AI-85021f223b28445b9c20d27b2e7bc7b1


## ⚙️개발환경
- Backend : Python
- Framework : Langchain
- Library : pydub
- Depolyment : Docker
