FROM ubuntu:20.04

ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY . /app
WORKDIR /app

RUN apt update && apt install git -y \
    python3.8 python3-dev python3-pip gcc g++ \
    libgl1-mesa-glx libglib2.0-0

RUN git clone git://github.com/Christophe-Foyer/maui63_postprocessing.git

WORKDIR /app/maui63_postprocessing
RUN pip3 install --upgrade pip
RUN pip3 install opencv-python
RUN pip3 install .

WORKDIR /app
EXPOSE 80
ENTRYPOINT [ "python3" ]
CMD [ "start_webserver.py" ]

