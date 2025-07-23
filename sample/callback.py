#from langchain.llms import OpenAI
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Azure OpenAI LLM 초기화
llm = AzureChatOpenAI(
    azure_endpoint="https://mcp-openai-ict.openai.azure.com/",
    api_key="EIVEP6wb2yltGSy968l9oMfcZZprvps2MuEaTPc8UHfkRS9DuTVfJQQJ99BFACNns7RXJ3w3AAABACOGdf6H",
    api_version="2024-12-01-preview",
    deployment_name="gpt-4o",
    temperature=0.7,
    max_tokens=1000,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],   
)

# 프롬프트 템플릿 생성
template = PromptTemplate(
    input_variables=["topic"],
    template="다음 주제에 대해 자세히 설명해주세요: {topic}"
)

# LLM 체인 생성
chain = LLMChain(llm=llm, prompt=template)

# 실행
result = chain.run("인공지능의 미래")
print(result)


