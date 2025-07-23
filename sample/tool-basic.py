from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_openai import AzureChatOpenAI
import requests

def search_web(query):
    """웹 검색 도구 예시"""
    # 실제로는 검색 API 호출
    return f"'{query}'에 대한 검색 결과입니다."

def calculate(expression):
    """계산 도구 예시"""
    try:
        result = eval(expression)
        return f"계산 결과: {result}"
    except:
        return "계산할 수 없는 식입니다."

# 도구 정의
tools = [
    Tool(
        name="웹검색",
        func=search_web,
        description="인터넷에서 정보를 검색할 때 사용하세요"
    ),
    Tool(
        name="계산기",
        func=calculate,
        description="수학 계산이 필요할 때 사용하세요"
    )
]

# 에이전트 초기화
# OpenAI LLM 설정
llm = AzureChatOpenAI(azure_endpoint="https://mcp-openai-ict.openai.azure.com/", api_key="EIVEP6wb2yltGSy968l9oMfcZZprvps2MuEaTPc8UHfkRS9DuTVfJQQJ99BFACNns7RXJ3w3AAABACOGdf6H", api_version="2024-12-01-preview", deployment_name="gpt-4o", temperature=0.7, max_tokens=1000)

agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 에이전트 실행
response = agent.run("2023년 AI 발전 현황을 검색하고, 10의 3제곱을 계산해주세요")
print(response)