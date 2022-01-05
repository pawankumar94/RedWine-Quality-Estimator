FROM python:slim
RUN echo $HOME
RUN mkdir -p /usr/home/wine_test
WORKDIR /usr/home/wine_test
COPY * .
RUN echo pwd
RUN pip install -r requirements.txt
CMD python train.py