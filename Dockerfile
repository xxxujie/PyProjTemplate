FROM python:3.9
WORKDIR /workspace/PyProjTemplate
COPY . ./
RUN pip install -r requirements.txt
CMD ["python", "main.py"]