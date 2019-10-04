import logging
import requests
import functools
import json
import sys
import os

from bottle import Bottle, request, response

LOGGER = logging.getLogger(__name__)

app = Bottle(__name__)


def jsonify(f):
    @functools.wraps(f)
    def wraps(*args, **kargs):
        rv = f(*args, **kargs)
        response.set_header('Content-Type', 'application/json')
        return json.dumps(rv)

    return wraps


@app.post('/event_notification')
@jsonify
def event_notification():
    LOGGER.info('Received event notification')
    return []


@app.post('/run_completer_command')
@jsonify
def run_completer_command():
    LOGGER.info('Received command request')

    return []


@app.post('/completions')
@jsonify
def get_completions():
    data = request.json
    basename = os.path.basename(data['filepath'])
    if basename.startswith('bash-fc'):
        data['history'] = open(os.path.expanduser('~/.bash_history')).read()
    r = requests.post('http://localhost:10086/completions', json=data)
    return r.json()


@app.post('/filter_and_sort_candidates')
def filter_and_sort_candidates():
    LOGGER.info('Received filter & sort request')
    return '[]'


@app.get('/healthy')
def get_healthy():
    LOGGER.info('Received health request')
    return 'true'


@app.get('/ready')
def get_ready():
    LOGGER.info('Received ready request')
    return 'true'


@app.post('/semantic_completion_available')
def filetype_completion_available():
    LOGGER.info('Received filetype completion available request')
    return 'false'


@app.post('/defined_subcommands')
def defined_subcommands():
    LOGGER.info('Received defined subcommands request')
    return '[]'


@app.post('/detailed_diagnostic')
def get_detailed_diagnostic():
    LOGGER.info('Received detailed diagnostic request')
    return '[]'


@app.post('/load_extra_conf_file')
def load_extra_conf_file():
    LOGGER.info('Received extra conf load request')
    return 'true'


@app.post('/ignore_extra_conf_file')
def ignore_extra_conf_file():
    return 'true'


@app.post('/debug_info')
@jsonify
def debug_info():
    LOGGER.info('Received debug info request')
    return {
        'python': {
            'executable': sys.executable,
        },
        'completer': None
    }


@app.post('/shutdown')
def shutdown():
    LOGGER.info('Received shutdown request')
    # TODO: shutdown gracefully
    os._exit(0)


@app.post('/receive_messages')
def receive_messages():
    # Receive messages is a "long-poll" handler.
    # The client makes the request with a long timeout (1 hour).
    # When we have data to send, we send it and close the socket.
    # The client then sends a new request.
    return 'false'

