bash#!/bin/bash
# kt-snap-fix.sh

echo "=== KT 사내망 Snap SSL 문제 해결 ==="
sudo mkdir /usr/share/ca-certificates/extra
sudo cp -rf myssl.crt /usr/share/ca-certificates/extra/
sudo chmod 644 /usr/share/ca-certificates/extra/*
sudo dpkg-reconfigure ca-certificates