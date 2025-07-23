bash
# VS Code Server를 수동으로 다운로드하고 설치
VSCODE_COMMIT="ddc367ed5c8936efe395cffeec279b04ffd7db78"
VSCODE_SERVER_DIR="$HOME/.vscode-server/bin/$VSCODE_COMMIT"

# 디렉토리 생성
mkdir -p "$VSCODE_SERVER_DIR"

# 직접 다운로드 (다른 미러 사용)
wget --no-check-certificate -O vscode-server.tar.gz \
  "https://github.com/microsoft/vscode/releases/download/1.95.3/vscode-server-linux-x64.tar.gz" || \
wget --no-check-certificate -O vscode-server.tar.gz \
 "https://vscode.download.prss.microsoft.com/dbazure/download/stable/${VSCODE_COMMIT}/vscode-server-linux-x64.tar.gz"
# 압축 해제
tar -xzf vscode-server.tar.gz -C "$VSCODE_SERVER_DIR" --strip-components=1

# 파일 확인
ls -la "$VSCODE_SERVER_DIR"