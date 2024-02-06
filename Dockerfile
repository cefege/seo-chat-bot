FROM python:3.12.1

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501


CMD ["python", "streamlit_app.py"]






