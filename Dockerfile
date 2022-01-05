FROM python:slim
RUN mkdir -p /var/www/html/wine_test
WORKDIR /var/www/html/wine_test
RUN echo ls -lrt /var/www/html/wine_test
COPY * .
RUN pip install -r requirements.txt
CMD python train.py