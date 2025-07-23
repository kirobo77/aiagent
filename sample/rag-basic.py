from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI

# OpenAI LLM 설정
llm = AzureChatOpenAI(azure_endpoint="https://mcp-openai-ict.openai.azure.com/", api_key="EIVEP6wb2yltGSy968l9oMfcZZprvps2MuEaTPc8UHfkRS9DuTVfJQQJ99BFACNns7RXJ3w3AAABACOGdf6H", api_version="2024-12-01-preview", deployment_name="gpt-4o", temperature=0.7, max_tokens=1000)


# 1. 문서 로드
loader = TextLoader('document.md', encoding='utf-8')
documents = loader.load()

# 2. 문서 분할
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# 3. 임베딩 및 벡터 저장소 생성
# Azure OpenAI 임베딩 모델 초기화
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",  # Azure에서 배포한 임베딩 모델의 배포 이름
    model="text-embedding-ada-002",  # 또는 "text-embedding-3-small", "text-embedding-3-large" 등
    azure_endpoint="https://mcp-openai-ict.openai.azure.com/",  # Azure OpenAI 리소스 엔드포인트
    api_key="EIVEP6wb2yltGSy968l9oMfcZZprvps2MuEaTPc8UHfkRS9DuTVfJQQJ99BFACNns7RXJ3w3AAABACOGdf6H",  # Azure OpenAI API 키
    api_version="2023-05-15"  # 사용할 API 버전
)

# https://mcp-openai-ict.openai.azure.com/openai/deployments/text-embedding-ada-002/embeddings?api-version=2023-05-15


vectorstore = Chroma.from_documents(texts, embeddings)

# 4. 검색기 설정
retriever = vectorstore.as_retriever()

# 5. QA 체인 생성
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# 6. 질문 실행
question = "문서에서 주요 내용은 무엇인가요?"
answer = qa_chain.run(question)
print(answer)