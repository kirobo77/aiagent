bash#!/bin/bash
# kt-snap-fix.sh

echo "=== KT 사내망 Snap SSL 문제 해결 ==="

# 1. 현재 인증서 상태 확인
echo "1. 현재 SSL 연결 상태 확인..."
openssl s_client -connect api.snapcraft.io:443 -verify_return_error 2>&1 | head -20

# 2. KT 프록시 환경에서 인증서 가져오기
echo "2. 인증서 체인 추출 중..."
echo | openssl s_client -connect api.snapcraft.io:443 -showcerts 2>/dev/null > /tmp/full-chain.pem

# 3. 루트 인증서 추출 (마지막 인증서가 보통 루트)
echo "3. 루트 인증서 추출..."
awk '/-----BEGIN CERTIFICATE-----/,/-----END CERTIFICATE-----/{cert=cert$0"\n"} /-----END CERTIFICATE-----/{print cert > "/tmp/cert"++i".pem"; cert=""}' /tmp/full-chain.pem

# 4. 각 인증서 설치
echo "4. 인증서 설치 중..."
for cert in /tmp/cert*.pem; do
    if [ -s "$cert" ]; then
        # 인증서 정보 확인
        subject=$(openssl x509 -in "$cert" -noout -subject 2>/dev/null | cut -d= -f2- | tr '/' '-' | tr ' ' '_')
        if [ ! -z "$subject" ]; then
            filename="kt-$(echo $subject | head -c 20).crt"
            sudo cp "$cert" "/usr/local/share/ca-certificates/$filename"
            echo "설치: $filename"
        fi
    fi
done

# 5. 인증서 업데이트
echo "5. CA 인증서 업데이트..."
sudo update-ca-certificates

# 6. snapd 재시작
echo "6. snapd 서비스 재시작..."
sudo systemctl restart snapd
sleep 3

# 7. 테스트
echo "7. 연결 테스트..."
if timeout 10 snap find hello >/dev/null 2>&1; then
    echo "✅ 성공! Snap이 정상적으로 작동합니다."
    snap find hello | head -5
else
    echo "❌ 여전히 문제가 있습니다."
    echo "IT팀에 다음 정보를 전달하세요:"
    echo "- api.snapcraft.io SSL 인증서 신뢰 필요"
    echo "- 회사 루트 인증서를 시스템에 설치 필요"
fi

# 8. 정리
rm -f /tmp/cert*.pem /tmp/full-chain.pem
