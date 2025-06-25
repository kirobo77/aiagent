MCP (Model Context Protocol)

Model Context Protocol(MCP)는 LLM 애플리케이션과 외부 데이터 소스 및 도구 간의 원활한 연동을 가능하게 해주는 오픈 프로토콜입니다. AI 기반 IDE를 만들거나, 채팅 인터페이스를 개선하거나, 맞춤형 AI 워크플로우를 구축하든 MCP는 LLM이 필요한 컨텍스트에 표준화된 방식으로 접근할 수 있도록 도와줍니다.

LLM이 내가 사용하는 어플리케이션을 하나의 도구로 활용할 수 있게 만들어주는 방법이라고 볼 수 있습니다.



Why MCP?

공식 홈페이지에서는 아래와 같은 장점이 있다고 소개하고 있네요.

MCP는 LLM(대형 언어 모델)을 기반으로 에이전트와 복잡한 워크플로우를 구축할 수 있도록 도와주는 플랫폼입니다. LLM은 종종 다양한 데이터나 도구와의 통합이 필요한데, MCP는 다음과 같은 기능을 제공합니다:

- LLM이 바로 연결해 사용할 수 있는 다양한 사전 구축된 통합 기능 제공
- LLM 제공업체 간 유연한 전환 가능
- 자체 인프라 내에서 데이터를 안전하게 보호할 수 있는 베스트 프랙티스 제공



주요 구성 요소

![img](https://resources-public-prd.modulabs.co.kr/post/attachment/7c0eb15f-6a43-4c17-af83-54e844b31343.png)

- MCP 호스트: Claude Desktop, IDE, 기타 AI 도구처럼 MCP를 통해 데이터에 접근하고자 하는 프로그램들
- MCP 클라이언트: 서버와 1:1 연결을 유지하는 프로토콜 클라이언트
- MCP 서버: 특화된 컨텍스트와 기능을 제공
- 로컬 데이터 소스: 사용자의 컴퓨터에 있는 파일, 데이터베이스, 서비스 등 MCP 서버가 안전하게 접근할 수 있는 로컬 자원
- 원격 서비스: API 등을 통해 인터넷 상에서 접근 가능한 외부 시스템으로, MCP 서버가 연결할 수 있음



MCP 작동 방식



연결 설정 (connection establishment)

- 호스트가 MCP 클라이언트를 생성
- 클라이언트는 MCP 서버와 연결 설정
- 연결 설정 과정에서 프로토콜 버전, 기능, 권한 등 설정

![img](https://resources-public-prd.modulabs.co.kr/post/attachment/52a65667-8013-44b2-9802-52a94471fc06.png)



컨텍스트 교환 (Context Exchange)

- 서버는 클라이언트에게 데이터 소스의 컨텍스트 정보를 제공
- 클라이언트는 이 정보를 호스트 프로세스에 전달
- 호스트 프로세스트는 여러 클라이언트로부터 받은 컨텍스트를 집계하여 AI 모델에 제공



도구 호출 (Tool Invocation)

- AI 모델은 특정 작업을 수행하기 위해 도구 호출 요청
- 호스트 프로세스는 이 요청을 적절한 클라이언트에 전달
- 클라이언트는 서버에 도구 호출 요청 전송
- 서버는 요청된 작업을 수행하고 결과를 클라이언트에 반환



결과 처리

- 클라이언트는 서버로부터 받은 결과를 호스트 프로세스에 전달
- 호스트 프로세스는 이 결과를 AI 모델에 제공
- AI 모델은 이 정보를 바탕으로 응답 생성



프로그래밍 언어

MCP 는 다양한 언어의 SDK를 제공하고 있습니다.

- [TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk)
- [Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Java SDK](https://github.com/modelcontextprotocol/java-sdk)
- [Kotlin SDK](https://github.com/modelcontextprotocol/kotlin-sdk)



MCP 서버 구현

저는 Python SDK를 이용해봤습니다. Python SDK는 MCP 사양을 완전히 구현하고 있어 다음과 같은 작업을 쉽게 수행할 수 있습니다.

- 어떤 MCP 서버와도 연결할 수 있는 MCP 클라이언트 구축
- 리소스, 프롬프트, 도구 등을 노출하는 MCP 서버 생성
- stdio, SSE 같은 표준 전송 방식 사용
- 모든 MCP 프로토콜 메시지와 생명주기 이벤트 처리



Install

MCP는 [uv](https://docs.astral.sh/uv/#highlights)를 통해서 설치하는 것을 권장합니다. 우선 uv를 설치해줍니다.

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```



MCP를 설치하기 전에 pyproject.toml을 만듭니다.

```
nano pyproject.toml
# 위 내용 붙여넣기 후 저장
```



```
# pyproject.toml
[project]
requires-python = ">=3.10"
name = "mcp-project"  # project 이름
version = "0.1.0"
dependencies = [
    "mcp[cli]>=1.6.0",
]
```



MCP를 설치해줍니다.

```
uv add "mcp[cli]"
```



대체 수단으로 pip를 통해서 설치도 가능합니다.

```
# pip 통한 설치 
pip install mcp
```



간단 서버 구현

덧셈 계산기와 인사하는 MCP 서버를 만들어봅니다.

```
from mcp.server.fastmcp import FastMCP


# Create an MCP server
mcp = FastMCP("mcp_project")


# Add an additional tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


# Add a dynamic greeing resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"


if __name__ == "__main__":
    mcp.run()
```



mcp 서버를 Dev 모드로 실행시킵니다.

```
mcp dev server.py
```



이후에 localhost:5173에 접속하면 다음과 같이 연결된 화면을 보실 수 있습니다.

![img](https://resources-public-prd.modulabs.co.kr/post/attachment/13a71f1b-fd67-4784-a7fb-a2ce60dc2027.png)

Resource Templates에서 get_greeting을 Tools에서 add를 확인 가능합니다.

![img](https://resources-public-prd.modulabs.co.kr/post/attachment/2f79a68f-43d1-4b1a-9cac-9c289615e33f.png)

![img](https://resources-public-prd.modulabs.co.kr/post/attachment/f3c235a1-ebca-4d87-9ede-176061e556c7.png)



이 서버는 [Claude Desktop](https://claude.ai/download)에 설치할 수 있으며, 아래 명령어를 실행하면 즉시 서버와 상호작용할 수 있습니다.

```
mcp install server.py
```



이제 Claude Desktop을 실행시키면 도구와 mcp 서버가 연동된 것을 볼 수 있습니다.

![img](https://resources-public-prd.modulabs.co.kr/post/attachment/edb9c8fc-0240-456d-bf0b-3e7b37cc07a2.png)

1+2를 해봅니다.

![img](https://resources-public-prd.modulabs.co.kr/post/attachment/d9cef71d-4973-4133-b58f-fa5786501023.png)

로컬 도구를 사용할지 여부를 묻는데 허용을 하면 로컬 도구를 이용해 답을 하는 것을 볼 수 있습니다.

![img](https://resources-public-prd.modulabs.co.kr/post/attachment/d2cbadf5-de94-4347-add0-8a6a7b1e6929.png)



T