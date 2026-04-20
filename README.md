# AWP Autonomous Prediction Bot (0 DH Edition) 🤖

This is a hardened, fully autonomous AWP Predict WorkNet validator designed for 24/7 operation with **Zero Cost (0 DH)**. 

## Features
- **Multi-Target Copy Trading**: Follows top-ranked wallets automatically.
- **Independent Fallback**: If targets are quiet, the bot uses its own technical analysis.
- **Momentum Engine**: Smart direction picking (UP/DOWN) based on real market trends.
- **Local AI Support**: Integration with Ollama/LM Studio via Docker `host-gateway`.
- **Hardened Parser**: Handles complex SMHL challenges (dots, symbols, obfuscation).

## Hardware & Software Requirements
- Windows (WSL2) or Linux.
- Docker + Docker Compose.
- (Optional but Recommended) A local LLM server (Ollama) running on port 18789.

## Setup Instructions

1. **Clone & Configure**:
   - Copy `.env.example` to `.env`.
   - Add your `WALLET3_ADDRESS` and `WALLET3_PRIVATE_KEY`.

2. **Local AI (Ollama)**:
   - Ensure Ollama is listening on `0.0.0.0` (not just localhost).
   - In `.env`, ensure `LLM_BASE_URL` points to `http://host.docker.internal:18789/v1`.

3. **Deploy**:
   ```bash
   docker-compose up -d --build
   ```

4. **Monitor**:
   ```bash
   docker logs -f predict-wallet-3
   ```

## Disclaimer
Running this bot involves financial risk. Use with caution.
