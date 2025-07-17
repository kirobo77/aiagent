# macOS용 AutoGen 실습 환경 구성 가이드

## 📋 목차
1. [시스템 요구사항](#시스템-요구사항)
2. [개발 도구 설치](#개발-도구-설치)
3. [Python 환경 설정](#python-환경-설정)
4. [AutoGen 설치 및 설정](#autogen-설치-및-설정)
5. [IDE 설정](#ide-설정)
6. [프로젝트 구조 생성](#프로젝트-구조-생성)
7. [환경 검증](#환경-검증)

---

## 1. 시스템 요구사항

### 최소 요구사항
- **OS**: macOS 10.15 (Catalina) 이상
- **메모리**: 8GB RAM 이상 (16GB 권장)
- **저장공간**: 5GB 이상 여유 공간
- **Python**: 3.9 이상 (3.11 권장)

### 권장 사양
- **OS**: macOS 12 (Monterey) 이상
- **프로세서**: Apple M1/M2 또는 Intel Core i5 이상
- **메모리**: 16GB RAM 이상
- **저장공간**: 10GB 이상 여유 공간

---

## 2. 개발 도구 설치

### 2.1 Homebrew 설치 (패키지 매니저)

```bash
# Homebrew 설치 (이미 설치된 경우 건너뛰기)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 설치 확인
brew --version

# Homebrew 업데이트
brew update
```

### 2.2 Git 설치

```bash
# Git 설치
brew install git

# Git 버전 확인
git --version

# Git 사용자 정보 설정 (필요시)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### 2.3 Python 설치

```bash
# Python 3.11 설치 (pyenv 사용)
brew install pyenv

# .zshrc 또는 .bash_profile에 pyenv 설정 추가
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc

# 터미널 재시작 또는 설정 적용
source ~/.zshrc

# Python 3.11 설치
pyenv install 3.11.6
pyenv global 3.11.6

# Python 버전 확인
python --version
pip --version
```

---

## 3. Python 환경 설정

### 3.1 프로젝트 디렉토리 생성

```bash
# 홈 디렉토리에 프로젝트 폴더 생성
mkdir ~/autogen-project
cd ~/autogen-project

# 프로젝트 하위 디렉토리 생성
mkdir -p {agents,templates,static,workspace,logs,config}
```

### 3.2 가상환경 생성 및 활성화

```bash
# 가상환경 생성
python -m venv autogen-env

# 가상환경 활성화
source autogen-env/bin/activate

# 가상환경 활성화 확인 (프롬프트에 (autogen-env) 표시)
which python
```

### 3.3 pip 업그레이드

```bash
# pip 업그레이드
python -m pip install --upgrade pip

# pip 버전 확인
pip --version
```

---

## 4. AutoGen 설치 및 설정

### 4.1 핵심 라이브러리 설치

```bash
# AutoGen 설치 (AG2 버전)
pip install ag2==0.3.0

# 또는 기존 AutoGen 사용 시
# pip install pyautogen==0.9.0

# 웹 프레임워크 설치
pip install fastapi==0.104.1
pip install uvicorn[standard]==0.24.0

# 환경 관리
pip install python-dotenv==1.0.0

# 데이터 처리
pip install pydantic==2.5.0
pip install aiofiles==23.2.1
pip install python-multipart==0.0.6

# OpenAI API
pip install openai==1.3.0

# HTTP 클라이언트
pip install httpx==0.25.2
pip install requests==2.31.0

# 유틸리티
pip install jinja2==3.1.2
```

### 4.2 의존성 파일 생성

```bash
# requirements.txt 생성
pip freeze > requirements.txt

# requirements.txt 내용 확인
cat requirements.txt
```

### 4.3 환경 변수 설정

```bash
# .env 파일 생성
cat > .env << 'EOF'
# OpenAI API 설정
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1

# Azure OpenAI 설정 (사용 시)
AZURE_OPENAI_API_KEY=your_azure_key_here
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_API_VERSION=2023-12-01-preview

# 서버 설정
HOST=0.0.0.0
PORT=8000
DEBUG=True

# 로그 설정
LOG_LEVEL=INFO
LOG_FILE=logs/autogen.log
EOF

echo ".env 파일이 생성되었습니다."
```

---

## 5. IDE 설정

### 5.1 Visual Studio Code 설치

```bash
# VS Code 설치 (Homebrew Cask 사용)
brew install --cask visual-studio-code

# 또는 공식 웹사이트에서 다운로드
# https://code.visualstudio.com/
```

### 5.2 VS Code 확장 프로그램 설치

```bash
# VS Code에서 프로젝트 열기
cd ~/autogen-project
code .
```

**필수 확장 프로그램:**
- Python (ms-python.python)
- Python Debugger (ms-python.debugpy)
- Pylance (ms-python.vscode-pylance)
- autoDocstring (njpwerner.autodocstring)
- GitLens (eamodio.gitlens)

### 5.3 VS Code 설정 파일 생성

```bash
# .vscode 디렉토리 생성
mkdir -p .vscode

# settings.json 생성
cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "./autogen-env/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "88"],
    "editor.formatOnSave": true,
    "files.autoSave": "afterDelay",
    "files.autoSaveDelay": 1000
}
EOF

# extensions.json 생성 (권장 확장 프로그램)
cat > .vscode/extensions.json << 'EOF'
{
    "recommendations": [
        "ms-python.python",
        "ms-python.debugpy",
        "ms-python.vscode-pylance",
        "njpwerner.autodocstring",
        "eamodio.gitlens"
    ]
}
EOF
```

---

## 6. 프로젝트 구조 생성

### 6.1 디렉토리 구조

```bash
# 프로젝트 구조 생성
mkdir -p {
    agents,
    templates,
    static/{css,js,images},
    workspace,
    logs,
    config,
    tests/{unit,integration},
    docs
}

# 프로젝트 구조 확인
tree -L 2 .
```

### 6.2 기본 파일 생성

```bash
# main.py 생성 (FastAPI 엔트리포인트)
cat > main.py << 'EOF'
"""
AutoGen FastAPI 메인 애플리케이션
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

app = FastAPI(
    title="AutoGen AI Agent System",
    description="macOS에서 실행되는 AutoGen 기반 AI Agent 시스템",
    version="1.0.0"
)

# 정적 파일 및 템플릿 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def root():
    return {"message": "AutoGen AI Agent System이 정상 실행 중입니다!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "platform": "macOS"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
EOF

# __init__.py 파일들 생성
touch agents/__init__.py
touch config/__init__.py
touch tests/__init__.py
```

### 6.3 agents 모듈 기본 파일 생성

```bash
# agents/base.py 생성
cat > agents/base.py << 'EOF'
"""
AutoGen 기본 Agent 클래스
"""
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

class BaseAgentConfig:
    """기본 Agent 설정 클래스"""
    
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        
    def get_llm_config(self) -> Dict[str, Any]:
        """LLM 설정 반환"""
        if self.azure_api_key and self.azure_endpoint:
            return {
                "config_list": [{
                    "model": "gpt-4",
                    "api_type": "azure",
                    "api_key": self.azure_api_key,
                    "base_url": self.azure_endpoint,
                    "api_version": "2023-12-01-preview"
                }],
                "temperature": 0.7,
                "cache_seed": 42
            }
        else:
            return {
                "config_list": [{
                    "model": "gpt-4",
                    "api_key": self.openai_api_key
                }],
                "temperature": 0.7,
                "cache_seed": 42
            }

# 전역 설정 인스턴스
agent_config = BaseAgentConfig()
EOF
```

---

## 7. 환경 검증

### 7.1 Python 패키지 설치 확인

```bash
# 가상환경이 활성화되어 있는지 확인
which python
# 출력: /Users/your-username/autogen-project/autogen-env/bin/python

# 핵심 패키지 설치 확인
python -c "import ag2; print('AG2 버전:', ag2.__version__)"
# 또는 pyautogen 사용 시:
# python -c "import autogen; print('AutoGen 설치 확인 완료')"

python -c "import fastapi; print('FastAPI 버전:', fastapi.__version__)"
python -c "import uvicorn; print('Uvicorn 설치 확인 완료')"
```

### 7.2 기본 애플리케이션 실행 테스트

```bash
# FastAPI 개발 서버 실행
python main.py

# 또는
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**브라우저에서 확인:**
- 메인 페이지: http://localhost:8000
- API 문서: http://localhost:8000/docs
- 상태 확인: http://localhost:8000/health

### 7.3 AutoGen 기본 테스트

```bash
# 테스트 스크립트 생성
cat > test_autogen.py << 'EOF'
"""
AutoGen 기본 동작 테스트
"""
import os
import sys
sys.path.append('.')

from agents.base import agent_config

def test_autogen_basic():
    """AutoGen 기본 테스트"""
    try:
        # AG2 사용 시
        from ag2 import ConversableAgent
        
        # 또는 pyautogen 사용 시:
        # from autogen import ConversableAgent
        
        llm_config = agent_config.get_llm_config()
        
        # 테스트용 Agent 생성
        test_agent = ConversableAgent(
            name="test_agent",
            system_message="당신은 테스트용 AI 어시스턴트입니다.",
            llm_config=llm_config,
            human_input_mode="NEVER"
        )
        
        print("✅ AutoGen Agent 생성 성공!")
        print(f"✅ Agent 이름: {test_agent.name}")
        return True
        
    except Exception as e:
        print(f"❌ AutoGen 테스트 실패: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧪 AutoGen 기본 테스트 시작...")
    if test_autogen_basic():
        print("🎉 모든 테스트 통과!")
    else:
        print("⚠️  일부 테스트 실패")
EOF

# 테스트 실행
python test_autogen.py
```

---

## 8. 추가 설정 (선택사항)

### 8.1 Docker 설정 (선택사항)

```bash
# Docker Desktop for Mac 설치
brew install --cask docker

# Dockerfile 생성
cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

# docker-compose.yml 생성
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  autogen-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./workspace:/app/workspace
      - ./logs:/app/logs
EOF
```

### 8.2 Git 설정

```bash
# .gitignore 생성
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
autogen-env/
venv/
env/

# Environment variables
.env
.env.local
.env.production

# IDE
.vscode/settings.json
.idea/

# macOS
.DS_Store
.AppleDouble
.LSOverride

# Logs
logs/
*.log

# Workspace
workspace/temp/
workspace/*.tmp

# API Keys
config/secrets.json
EOF

# Git 저장소 초기화
git init
git add .
git commit -m "Initial commit: macOS AutoGen 프로젝트 설정"
```

---

## 9. 문제 해결

### 9.1 일반적인 문제들

**1. Python 경로 문제**
```bash
# pyenv 설정 확인
pyenv which python
pyenv versions

# PATH 확인
echo $PATH
```

**2. 가상환경 활성화 문제**
```bash
# 가상환경 재생성
rm -rf autogen-env
python -m venv autogen-env
source autogen-env/bin/activate
```

**3. 패키지 설치 오류**
```bash
# pip 캐시 정리
pip cache purge

# 의존성 재설치
pip install --force-reinstall -r requirements.txt
```

**4. macOS 권한 문제**
```bash
# Xcode 명령행 도구 설치
xcode-select --install

# 권한 확인
ls -la ~/autogen-project
```