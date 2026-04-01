FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY eco_env.py eco_obs_encoder.py eco_ppo.py eco_vec_env.py server.py app.py ./
COPY static/ static/
COPY model/eco_latest.pkt model/eco_latest.pkt

EXPOSE 7860

CMD ["python", "app.py"]
