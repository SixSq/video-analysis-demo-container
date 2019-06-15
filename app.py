#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse

from flask import Flask, render_template, Response
from video_analysis import VideoAnalysis


app = Flask(__name__)
parameters = {}


@app.route('/')
def index():
    """Video streaming home page which makes use of /mjpeg."""
    return render_template('index.html')

@app.route('/video')
def video():
    """Video streaming home page which makes use of /jpeg."""
    return render_template('video.html')

@app.route('/mjpeg')
def mjpeg():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(VideoAnalysis(**parameters).mjpeg_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame',
                    direct_passthrough=True)

@app.route('/jpeg')
def jpeg():
    return Response(VideoAnalysis(**parameters).request_image(),
                    mimetype='image/jpeg',
                    direct_passthrough=True)

def get_argument_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--help', action='help', help='show this help message and exit')
    parser.add_argument('-i', '--input_source', default=os.environ.get('INPUT_SOURCE', 0),       help='Video input source')
    parser.add_argument('-q', '--quality',      default=os.environ.get('QUALITY', 80), type=int, help='Quality of the output stream [0-100]')
    parser.add_argument('-w', '--width',        default=os.environ.get('WIDTH', 1280), type=int, help='Quality of the output stream')
    parser.add_argument('-h', '--height',       default=os.environ.get('HEIGHT', 720), type=int, help='Quality of the output stream')
    parser.add_argument('-t', '--threads',      default=os.environ.get('THREADS',  2), type=int, help='Number of thread to run analysis')
    parser.add_argument('--n_frames',           default=os.environ.get('N_FRAMES', 1), type=int, help='Run analysis every n frames')
    # TODO
    #parser.add_argument('-d', '--debug', dest='debug', help='Show debug log level (all log messages).', action='store_true', default=False)
    #parser.add_argument('-q', '--quiet', dest='quiet', help='Show less log messages. Add more to get less details.', action='count', default=0)
    return parser

def main():
    global parameters

    argument_parser = get_argument_parser()
    args = argument_parser.parse_args()
    parameters = dict(args._get_kwargs())
    for k,v in parameters.items():
        print('{}: {}'.format(k, v))

    app.run(host='0.0.0.0', debug=False, threaded=True)

if __name__ == '__main__':
    main()
