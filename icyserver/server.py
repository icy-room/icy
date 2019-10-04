import time
import json
from bottle import Bottle, request, response
from icyserver.icy import Icy


class App:
    icy: Icy
    bottle: Bottle

    def __init__(self):
        self.icy = Icy('gpt2-medium')
        self.bottle = Bottle()


app = App()


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
    n, items = app.icy.predict(context)
    print(f'cost {time.time()-t0} seconds')
    completions = [
        {"insertion_text": item.splitlines(keepends=False)[0]}
            for item in items
    ]
    result = {"completions": completions,
              "completion_start_column": max(data['column_num'] - n, 0),
              "errors": []}
    response.set_header('Content-Type', 'application/json')
    return json.dumps(result)


if __name__ == '__main__':
    app.bottle.run(port=10086)
