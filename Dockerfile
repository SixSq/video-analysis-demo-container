# hadolint ignore=DL3007
FROM sixsq/opencv-python:latest

RUN pip3 install flask paho-mqtt requests

RUN mkdir -p /root/video_analysis
COPY *.py      /root/video_analysis/
COPY static    /root/video_analysis/static
COPY templates /root/video_analysis/templates
RUN mknod /dev/video0 c 81 0; mknod /dev/video1 c 81 1

WORKDIR /root/video_analysis
ENTRYPOINT ["python3", "app.py"]
