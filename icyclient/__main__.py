import argparse
from icyclient import handlers


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='server hostname')
    parser.add_argument('--port', type=int, default=10000,
                        help='server port')
    parser.add_argument('--log', type=str, default='info',
                        help='log level, one of '
                             '[debug|info|warning|error|critical]')
    parser.add_argument('--idle_suicide_seconds', type=int, default=0,
                        help='num idle seconds before server shuts down')
    parser.add_argument('--check_interval_seconds', type=int, default=600,
                        help='interval in seconds to check server '
                             'inactivity and keep subservers alive')
    parser.add_argument('--options_file', type=str,
                        help='file with user options, in JSON format')
    parser.add_argument('--stdout', type=str, default=None,
                        help='optional file to use for stdout')
    parser.add_argument('--stderr', type=str, default=None,
                        help='optional file to use for stderr')
    parser.add_argument('--keep_logfiles', action='store_true', default=None,
                        help='retain logfiles after the server exits')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    handlers.app.run(
        host=args.host, port=args.port,
    )

