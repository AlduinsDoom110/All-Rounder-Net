#!/usr/bin/env bash
set -euo pipefail
mkdir -p certs
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout certs/localhost-key.pem \
  -out certs/localhost.pem \
  -subj "/C=US/ST=NA/L=Local/O=All-Rounder-Net/CN=localhost"
echo "Created certs/localhost.pem and certs/localhost-key.pem"
