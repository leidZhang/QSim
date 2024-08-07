import logging
from flask import Flask

app = Flask(__name__)

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)