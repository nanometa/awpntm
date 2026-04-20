FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PREDICT_SERVER_URL=https://api.agentpredict.work

# Base tools + Python
RUN apt-get update && apt-get install -y \
    curl wget ca-certificates \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip3 install requests

# predict-agent binary — try install.sh first, fall back to direct release download
RUN curl -sSL https://raw.githubusercontent.com/awp-worknet/prediction-skill/main/install.sh \
        -o /tmp/install.sh && sh /tmp/install.sh \
    && (cp /root/.local/bin/predict-agent /usr/local/bin/predict-agent 2>/dev/null || \
        cp ~/.local/bin/predict-agent /usr/local/bin/predict-agent 2>/dev/null || true) \
    && (which predict-agent || \
        curl -fL "https://github.com/awp-worknet/prediction-skill/releases/latest/download/predict-agent-x86_64-unknown-linux-musl" \
             -o /usr/local/bin/predict-agent) \
    && chmod +x /usr/local/bin/predict-agent \
    && predict-agent --version || echo "predict-agent installed"

WORKDIR /app
COPY copy_trader.py .
COPY entrypoint.sh .
RUN sed -i 's/\r$//' entrypoint.sh && chmod +x entrypoint.sh

VOLUME ["/app/data"]

ENTRYPOINT ["/app/entrypoint.sh"]
