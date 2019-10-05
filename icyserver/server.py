import time
import json
from collections import OrderedDict
from bottle import Bottle, request, response
from icyserver.icy import new_icy


class App:
    def __init__(self):
        self.icy = new_icy('./hub1000')
        self.bottle = Bottle()


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
        context = history + "#!bin/sh\n" + context
    t0 = time.time()
    print(f'context: {context}')
    n, items, probs = app.icy.predict(context)
    print(f'cost {time.time()-t0} seconds, candidates: {items!r}')
    items = filter_items(items, probs)
    print(f'final candidates: {items}')
    completions = [
        {"insertion_text": item, "extra_menu_info": "{: >6.3f}".format(p*100)}
            for item, p in items
    ]
    result = {"completions": completions,
              "completion_start_column": max(data['column_num'] - n, 0),
              "errors": []}
    response.set_header('Content-Type', 'application/json')
    return json.dumps(result)


if __name__ == '__main__':
    app.bottle.run(port=10086)
