FROM python:3.7
RUN mkdir -p /usr/src/app/
WORKDIR /usr/src/app/
COPY . usr/src/app/
RUN pip freeze > requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "./start_service.py"]