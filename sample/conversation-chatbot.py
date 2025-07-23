from langchain_openai import AzureChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# OpenAI LLM 설정
llm = AzureChatOpenAI(azure_endpoint="https://mcp-openai-ict.openai.azure.com/", api_key="EIVEP6wb2yltGSy968l9oMfcZZprvps2MuEaTPc8UHfkRS9DuTVfJQQJ99BFACNns7RXJ3w3AAABACOGdf6H", api_version="2024-12-01-preview", deployment_name="gpt-4o", temperature=0.7, max_tokens=1000)

# 메모리 설정 (대화 기록 유지)
memory = ConversationBufferMemory()

# 대화형 체인 생성
chatbot_chain = ConversationChain(llm=llm, memory=memory)

def chatbot():
    print("챗봇에 질문을 입력하세요. 종료하려면 'quit'을 입력하세요.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("챗봇을 종료합니다.")
            break
        response = chatbot_chain.run(user_input)
        print(f"AI: {response}")

# 챗봇 실행
chatbot()