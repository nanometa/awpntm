#!/bin/bash
set -e

AGENT_ID="${AWP_AGENT_ID:-predict-worker}"
PERSONA="${AWP_PERSONA:-degen}"
echo "[${AGENT_ID}] Starting — wallet: ${AWP_ADDRESS}"

mkdir -p /app/data

# Set proxy at OS level so every child process uses it
if [ -n "$PROXY_HOST" ] && [ -n "$PROXY_PORT" ]; then
    PROXY_URL="http://${PROXY_USER}:${PROXY_PASS}@${PROXY_HOST}:${PROXY_PORT}"
    export HTTP_PROXY="$PROXY_URL"
    export HTTPS_PROXY="$PROXY_URL"
    export http_proxy="$PROXY_URL"
    export https_proxy="$PROXY_URL"
    echo "[${AGENT_ID}] Proxy: ${PROXY_HOST}:${PROXY_PORT}"
fi

echo "[${AGENT_ID}] Using AWP_PRIVATE_KEY direct signing"

# Preflight (non-fatal)
predict-agent preflight 2>&1 | head -30 || true

# Set persona if not already set (required before predictions)
echo "[${AGENT_ID}] Setting persona: ${PERSONA}"
predict-agent set-persona "${PERSONA}" 2>&1 || echo "[${AGENT_ID}] Persona already set or cooldown active"

exec python3 /app/copy_trader.py
