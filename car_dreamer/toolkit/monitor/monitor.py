import atexit
import base64
import json
import queue
import threading

import cv2
import numpy as np
from flask import Flask, Response, render_template


class EnvMonitorBase:
    def __init__(self, config):
        self._config = config
        self._obs_queue = queue.Queue()
        self._info_queue = queue.Queue()
        self._thread = threading.Thread(target=self._run_server)
        self._thread.start()
        atexit.register(self.stop)

    def _run_server(self):
        app = Flask(__name__, template_folder="templates")

        @app.route("/")
        def index():
            return render_template("index.html")

        @app.route("/stream")
        def stream():
            def generate():
                while True:
                    obs = self._obs_queue.get()
                    info = self._info_queue.get()
                    frame = self._render(obs, info)
                    yield f"data: {json.dumps(frame)}\n\n"
                    self._obs_queue.task_done()
                    self._info_queue.task_done()

            return Response(generate(), mimetype="text/event-stream")

        app.run(
            host="0.0.0.0",
            port=self._config.world.carla_port + 7000,
            use_reloader=False,
        )

    def _render_info(self, info):
        rendered_info = {}
        for key, value in info.items():
            if isinstance(value, (float, int, bool)):
                rendered_info[key] = value
            elif isinstance(value, np.number):
                rendered_info[key] = value.item()
            elif isinstance(value, np.ndarray) and value.ndim == 1:
                rendered_info[key] = value.tolist()
            else:
                rendered_info[key] = str(value)
        return rendered_info

    def _render_images(self, obs):
        images = []
        display_config = self._config.display
        if display_config.enable and display_config.render_keys:
            for key in display_config.render_keys:
                if key in obs:
                    img = obs[key]
                    if len(img.shape) == 2:
                        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
                    else:
                        img = img[:, :, ::-1]
                    _, img_encoded = cv2.imencode(".webp", img)
                    img_base64 = base64.b64encode(img_encoded).decode("utf-8")
                    images.append({"key": key, "image": img_base64})
        return images

    def _render(self, obs, info):
        return {"images": self._render_images(obs), "info": self._render_info(info)}

    def stop(self):
        if self._thread and self._thread.is_alive():
            self._thread.join()

    def __del__(self):
        self.stop()


class EnvMonitorOpenCV(EnvMonitorBase):
    def render(self, obs, info):
        if not self._obs_queue.full():
            self._obs_queue.put(obs)
        if not self._info_queue.full():
            self._info_queue.put(info)
