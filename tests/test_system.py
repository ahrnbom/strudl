import json
import os
from io import BytesIO
from tempfile import TemporaryFile, NamedTemporaryFile
from time import sleep

import cv2

from server import make_app

client = make_app().app.test_client()

def post_json(url, data):
    return client.post(url, data=json.dumps(data), content_type='application/json')

def run_job(url, timeout=60):
    r = client.post(url)
    assert r.status_code == 202
    jobid = r.json

    res = None
    for i in range(timeout):
        r = client.get("/jobs?jobs_type=recent_with_status")
        assert r.status_code == 200
        for status in r.json:
            if status['id'] == jobid and status['result'] not in ('running', 'queued'):
                res = status['result']
                break
        if res is not None:
            break
        sleep(1)
    else:
        client.delete("/jobs/%s" % jobid)
        res = 'timeout'

    r = client.get("/jobs/%s" % jobid)
    assert r.status_code == 200
    return res, r.data.decode('utf8')


mydir = os.path.dirname(os.path.abspath(__file__))

class TestSystem:
    def test_ssd_download(self):
        r = client.get("/pretrained_weights")
        assert r.status_code == 200

    def test_create_dataset(self):
        r = client.post('/datasets?dataset_name=test&class_names=car%2Cpedestrian%2Cbicycle%2Cmotorbike%2Cbus%2Cother&class_heights=1.5%2C1.8%2C2%2C2%2C3%2C2')
        assert r.status_code == 200

    def test_config_dataset(self):
        config = {
          "annotation_train_split": 1.0,
          "images_to_annotate": 3,
          "images_to_annotate_per_video": 1,
          "point_track_resolution": "(320, 240)",
          "video_fps": 15,
          "video_resolution": "(640, 480, 3)"
        }
        r = post_json('/datasets/config?dataset_name=test', config)
        assert r.status_code == 200

        r = client.get('/datasets/config?dataset_name=test')
        assert r.json == config
        assert r.status_code == 200

    def test_masks(self):
        mask_file_name = mydir + "/../test_data/mask.png"
        data = {'mask_image_file': (open(mask_file_name, "rb"), "mask.png")}
        r = client.post('/datasets/masks?dataset_name=test', data=data)
        assert r.status_code == 200

        r = client.get('/datasets/masks?dataset_name=test')
        assert r.status_code == 200

        img1 = cv2.imread(mask_file_name)
        with NamedTemporaryFile(suffix='.png') as fd:
            fd.write(r.data)
            fd.flush()
            img2 = cv2.imread(fd.name)
        assert all((img1 == img2).flat)

    def test_import_videos(self):
        r = client.get("/videos?dataset_name=test")
        if r.status_code == 200:
            old_count = len(r.json)
        else:
            old_count = 0

        res, log = run_job('/jobs/import_videos?dataset_name=test&path=test_data%2F*.mp4&method=imageio')
        print(log)
        assert res == 'success'

        r = client.get("/videos?dataset_name=test")
        assert r.status_code == 200
        assert len(r.json) == old_count + 3


