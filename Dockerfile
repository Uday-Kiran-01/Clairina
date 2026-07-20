FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -U pip
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi
EXPOSE 8501
ENV STREAMLIT_SERVER_HEADLESS=true
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.headless=true"]
