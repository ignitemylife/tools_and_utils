import os
import datetime
import logging
import flask
import tornado.wsgi
import tornado.httpserver
import urllib.request
import base64
import numpy as np
import cv2
import sys
from pdb import set_trace as st

REPO_DIRNAME = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(REPO_DIRNAME)

import sys
sys.path.insert(0, ROOT_DIR)

from algo.dummy import Dummy

max_size = 1024
task_name = 'your task'
UPLOAD_FOLDER = os.path.join(REPO_DIRNAME, 'demos_uploads')

# Obtain the flask app object
app = flask.Flask(__name__)

def predict(filename):
    query_img_rect, topk, grid_img = app.model(filename)
    if isinstance(query_img_rect, cv2.UMat):
        query_img_rect = cv2.UMat.get(query_img_rect)
    result = embed_image_html(grid_img)

    print("========================================")
    return flask.render_template(
        'index.html', has_result=True, result=[result], task_name=task_name,
        origin_src=embed_image_html(query_img_rect)
    )

def call_and_render(filename):
    ret = app.model(filename)
    rgb = ret[..., ::-1]
    if isinstance(ret, np.ndarray):
        return flask.render_template(
            'index.html', has_result=True, result=[True, (1, 2, 3)], task_name=task_name,
            origin_src=embed_image_html(ret), pred_src=embed_image_html(rgb)
        )
    else:
        return flask.render_template(
            'index.html', has_result=True, result=[ret], task_name=task_name
        )


@app.route('/')
def index():
    return flask.render_template('index.html', has_result=False, task_name=task_name,)


@app.route('/task_url', methods=['GET'])
def task_url():
    imageurl = flask.request.args.get('imageurl', '')
    resp = urllib.request.urlopen(imageurl)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    cv2.imwrite(UPLOAD_FOLDER + "temp.jpg", image)
    # predict_result = predict(UPLOAD_FOLDER + "temp.jpg")
    predict_result = call_and_render(UPLOAD_FOLDER + "temp.jpg")
    return predict_result

@app.route('/img_upload', methods=['POST'])
def img_upload():
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['origin_src']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + '_' + imagefile.filename
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        imagefile.save(filename)
    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )

    logging.info('Image save as %s', filename)

    # predict_result = predict(filename)
    predict_result = call_and_render(filename)
    return predict_result


def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    a = (max_size * 1.0) / (image.shape[0] * 1.0)
    b = (max_size * 1.0) / (image.shape[1] * 1.0)
    scale = min(a, b)
    image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
    _, cv2_encode = cv2.imencode('.png', image)
    data = base64.b64encode(cv2_encode.tostring())
    return 'data:image/png;base64,' + bytes.decode(data)


def start_tornado(app, port):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app, port, cfg_file='./config.yaml'):
    # Initialize classifier + warm start by forward for allocation
    print("loading model parameters...")
    app.model = Dummy()
    print("model has been loaded..")
    start_tornado(app, port)


if __name__ == '__main__':
    port = sys.argv[1] if len(sys.argv) > 1 else 10086
    cfg_file = sys.argv[2] if len(sys.argv) > 2 else './config.yaml'

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(module)s :%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S %p')
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    start_from_terminal(app, port)
