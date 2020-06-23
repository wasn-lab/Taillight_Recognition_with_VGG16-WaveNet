import json
import logging

def read_json_file(filename):
    if not os.path.isfile(filename):
        logging.error("File not found: %s", filename)
        return []
    with io.open(filename, encoding="utf-8") as _fp:
        jdata = json.load(_fp)
    return jdata
