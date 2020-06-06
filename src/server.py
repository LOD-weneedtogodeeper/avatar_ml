import json
import asyncio
import uvloop
import time
import datetime
import os


from base64 import b64decode, b64encode
from pprint import pprint
from signal import signal, SIGINT
from sanic import Sanic, response
from sanic_cors import CORS, cross_origin
from sanic_prometheus import monitor
from first_order_model import FirstOrderModel

app = Sanic(__name__)
CORS(app)

# Exposing port for prometheus metrics
#monitor(app, mmc_period_sec=None).expose_endpoint()

model = FirstOrderModel(config_path=os.environ['CONFIG_PATH'], checkpoint_path=os.environ['CP_PATH'], if_cpu=False)

@app.route("/inference/", methods=["POST"])
async def infer(request):
    print(request.json.keys())
    try:
        image = request.json['img']
        video = request.json['video']
    except AttributeError:
        err_code = {"status"}
        return response.json(
            {'message': 'Wrong POST, please upload image and video'},
            headers={'X-Served-By': 'sanic'},
            status=404
            )

    image, video = (b64decode(image), b64decode(video))
    preds = model.infer(image, video)
   

    return response.json(
        {'video': b64encode(preds).decode('utf-8')},
        headers={'X-Served-By': 'sanic'},
        status=200
        )



if __name__ == "__main__":
    sanic_config = {
        "host": "127.0.0.1",
        "port": 9999,
        "workers": 1,
        "debug": False,
        "access_log": False
    }

    app.run(**sanic_config)
