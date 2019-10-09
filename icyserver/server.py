import time
import json
import logging

from collections import OrderedDict
from bottle import Bottle, request, response
from icyserver.icy import new_icy


logger = logging.getLogger(__name__)


class App:
    def __init__(self):
        self.model_name = './hub1000'
        self.bottle = Bottle()

    def load_model(self):
        self.icy = new_icy(self.model_name)


app = App()


def filter_items(items, probs):
    counter = OrderedDict()
    for item, p in zip(items, probs):
        item = item.splitlines(keepends=False)[0].rstrip()
        if not item.strip():
            continue
        if item in counter:
            counter[item] += p
        else:
            counter[item] = p
    return sorted(counter.items(), key=lambda x: x[1], reverse=True)


@app.bottle.post('/completions')
def completions():
    data = request.body.read()
    data = json.loads(data.decode('utf8'))
    filepath = data['filepath']
    line_num = data['line_num']
    column_num = data['column_num']
    contents = data['file_data'].get(filepath, {}).get('contents')
    if contents is None:
        return '[]'
    if column_num <= 1:
        return '[]'
    contents = contents.splitlines(keepends=False)
    front_half = contents[:line_num]
    front_half[-1] = front_half[-1][:column_num-1]
    context = '\n'.join(front_half)
    history = data.get('history')
    if history:  # interactive bash
        context = history + context
    t0 = time.time()
    logger.info(f'context: \n----\n[{context}]\n')
    result = app.icy.predict(context, filepath)
    if result is None:
        return '{}'
    n, prefix, items, probs = result
    logger.info(f'cost {time.time()-t0} seconds, candidates: {items!r}')
    items = filter_items(items, probs)
    logger.info(f'final candidates: {items}')
    completions = [
        {"insertion_text": item, "extra_menu_info": "{: >6.3f}".format(p*100)}
            for item, p in items
    ]
    if prefix.strip():
        prefix_item = {"insertion_text": prefix.rstrip()}
        if len(completions) == 0:
            completions.insert(0, prefix_item)
        else:
            completions.insert(1, prefix_item)

    result = {"completions": completions,
              "completion_start_column": max(data['column_num'] - n, 0),
              "errors": []}
    response.set_header('Content-Type', 'application/json')
    return json.dumps(result)


if __name__ == '__main__':
    import tensorflow as tf

    logging.basicConfig(level=logging.DEBUG)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    app.load_model()
    app.bottle.run(host='0.0.0.0', port=10086)
