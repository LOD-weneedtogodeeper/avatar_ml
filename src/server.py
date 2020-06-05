import json
import asyncio
import uvloop
import time
import datetime
import os
import torch

torch.multiprocessing.set_start_method('spawn')
from base64 import b64decode
from pprint import pprint
from signal import signal, SIGINT
from sanic import Sanic, response
from sanic_cors import CORS, cross_origin
from dateutil import parser
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
    print(model.infer(image, video))
    return response.json(test)



if __name__ == "__main__":

    asyncio.set_event_loop(uvloop.new_event_loop())
    serv_coro = app.run(host="0.0.0.0", port=9999,workers=100, return_asyncio_server=True)
    loop = asyncio.get_event_loop()
    serv_task = asyncio.ensure_future(serv_coro, loop=loop)
    signal(SIGINT, lambda s, f: loop.stop())
    server = loop.run_until_complete(serv_task)
    try:
        loop.run_forever()
    except KeyboardInterrupt as e:
        loop.stop()
    finally:
        server.before_stop()

        # Wait for server to close
        close_task = server.close()
        loop.run_until_complete(close_task)

        # Complete all tasks on the loop
        for connection in server.connections:
            connection.close_if_idle()
        server.after_stop()
