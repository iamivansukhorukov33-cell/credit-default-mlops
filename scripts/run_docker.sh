#!/bin/sh
set -eu

# Build from the project root. / Собираем образ из корня проекта.
cd "$(dirname "$0")/.."
docker build -t credit-default:local .
exec docker run --rm -p "127.0.0.1:${PORT:-8000}:8000" credit-default:local
