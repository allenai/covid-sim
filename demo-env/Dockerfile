FROM ubuntu
#WORKDIR /usr/src/app
RUN apt-get update
RUN apt-get -y install zip
RUN apt-get -y install unzip
RUN apt-get install -y git
RUN apt-get install python3.7
COPY . .
#RUN pip install -r -y requirements.txt

# INTALL CONDA

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
    


RUN sh ./install.sh
