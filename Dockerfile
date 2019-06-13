# hadolint ignore=DL3007
FROM sixsq/opencv-python:latest

RUN pip3 install flask

RUN mkdir -p /root/video_analysis
COPY *.py      /root/video_analysis/
COPY static    /root/video_analysis/static
COPY templates /root/video_analysis/templates

WORKDIR /root/video_analysis
ENTRYPOINT ["python3", "app.py"]
