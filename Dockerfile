FROM python:3.10-slim
ADD ./cnn_net.pt /app/cnn_net.pt
ADD ./requirements.txt /app/requirements.txt
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD python ./app.py
