
FROM python:3.7.6

LABEL name="Jonathan Sun <jonathansun456@gmail.com>"

# install dependencies and then copy the rest of source files
COPY requirements.txt /app/
WORKDIR /app
RUN pip install -r requirements.txt
COPY . /app

# starts CubeTea application
ENTRYPOINT python
CMD python app.py

