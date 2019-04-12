import json
import os
from io import BytesIO
from tempfile import TemporaryFile, NamedTemporaryFile
from time import sleep
from zipfile import ZipFile

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
    print(r.data.decode('utf8'))
    return res


mydir = os.path.dirname(os.path.abspath(__file__))

class TestWorkflow:
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

        res = run_job('/jobs/import_videos?dataset_name=test&path=test_data%2F*.mp4&method=imageio')
        assert res == 'success'

        r = client.get("/videos?dataset_name=test")
        assert r.status_code == 200
        assert len(r.json) == old_count + 3

    def test_prep_anotation(self):
        res = run_job("/jobs/prepare_annotation?dataset_name=test")
        assert res == 'success'

    def test_point_tracks(self):
        res = run_job("/jobs/point_tracks?dataset_name=test&visualize=true&overwrite=true")
        assert res == 'success'

    def test_annotate(self):
        r = client.get("/annotate/images?dataset_name=test&annotation_set=train")
        assert r.status_code == 200
        names = [None, None, None]
        for name, n, state in r.json:
            i = ['170439', '170444', '170449'].index(name.split('_')[1])
            names[i] = name
        annotations=[
            b'6 0.28984 0.49062 0.04531 0.13125 px:0.31250,0.26719,0.28125,0.31250 py:0.55625,0.47292,0.42500,0.55000 pedestrian\n3 0.07422 0.68333 0.14844 0.22500 px:0.00156,0.00000,0.07500,0.14844 py:0.79583,0.70208,0.57083,0.63333 car\n3 0.13906 0.29063 0.08750 0.12708 px:0.10312,0.09531,0.15156,0.18281 py:0.35417,0.32500,0.22708,0.25208 car\n3 0.33125 0.19062 0.14063 0.08542 px:0.34531,0.26094,0.35156,0.40156 py:0.23333,0.19167,0.14792,0.20208 car\n',
            b'3 0.11957 0.25862 0.06936 0.08138 px:0.09731,0.08489,0.14079,0.15424 py:0.29931,0.25379,0.21793,0.27034 car\n3 0.07557 0.67862 0.14907 0.22069 px:0.00207,0.00104,0.07039,0.15010 py:0.78897,0.61931,0.56828,0.62345 car\n3 0.93219 0.44828 0.13561 0.15448 px:0.99379,0.86439,0.95135,1.00000 py:0.52552,0.41103,0.37103,0.41793 car\n6 0.67754 0.71862 0.05694 0.14345 px:0.69358,0.64907,0.69772,0.70600 py:0.79034,0.73655,0.64690,0.66207 pedestrian\n',
            b'3 0.07609 0.68345 0.14596 0.21931 px:0.00414,0.00311,0.07764,0.14907 py:0.79310,0.66483,0.57379,0.63310 car\n3 0.09731 0.27517 0.05797 0.08690 px:0.07557,0.06832,0.11905,0.12629 py:0.31862,0.25931,0.23172,0.30069 car\n3 0.60300 0.29862 0.16460 0.12552 px:0.65114,0.52070,0.60663,0.68530 py:0.36138,0.28966,0.23586,0.30897 car\n3 0.86232 0.42276 0.12629 0.13103 px:0.88509,0.79917,0.87474,0.92547 py:0.48828,0.40690,0.35724,0.41931 car\n3 0.90787 0.25793 0.06625 0.07448 px:0.92961,0.87474,0.88820,0.94099 py:0.29517,0.22207,0.22069,0.27172 car\n1 0.53416 0.91103 0.09524 0.17241 px:0.50621,0.48654,0.54037,0.58178 py:0.99724,0.99310,0.82483,0.89517 bicycle\n',
        ]

        get_anotate_url_template = '/annotate/annotation?annotation_set=train&image_number=1&video_name=%s&dataset_name=test&output_format=plain&accept_auto=false'
        set_anotate_url_template = '/annotate/annotation?annotation_set=train&image_number=1&video_name=%s&dataset_name=test'
        for i in range(3):
            r = client.post(set_anotate_url_template % names[i], data=annotations[i], content_type="text/plain")
            assert r.status_code == 200

            r = client.get(get_anotate_url_template % names[i])
            assert r.status_code == 200
            assert r.data == annotations[i]

    def test_config_run(self):
        config = {
            "confidence_threshold": 0.6,
            "detection_batch_size": 3,
            "detection_training_batch_size": 3,
            "detector_resolution": "(640, 480, 3)"
        }
        r = post_json("/runs/config?dataset_name=test&run_name=testrun", config)
        assert r.status_code == 200

        r = client.get("/runs/config?dataset_name=test&run_name=testrun")
        assert r.status_code == 200
        assert r.json == config

    def test_train(self):
        res = run_job("/jobs/train_detector?dataset_name=test&run_name=testrun&epochs=5")
        assert res == 'success'

    def test_detect_objects(self):
        res = run_job("/jobs/detect_objects?dataset_name=test&run_name=testrun", 600)
        assert res == 'success'

    def test_visualize_detections_pixels(self):
        res = run_job("/jobs/visualize_detections?dataset_name=test&run_name=testrun&confidence_threshold=0.6&coords=pixels")
        assert res == 'success'

    def test_world_calibration(self):
        calib_data = b"Cx: 284.440315296138\nCy: 295.577475064004\nSx: 0.977499718912692\nTx: 0.655851421157206\nTy: -5.413644708654\nTz: 22.6704367187891\ndx: 1.0\ndy: 1.0\nf: 655.929783301456\nk: 1.19677966687451e-06\nr1: 0.77782287558943\nr2: -0.628168580366148\nr3: 0.0198949453522387\nr4: 0.350254902950456\nr5: 0.459548568226311\nr6: 0.8161719282114\nr7: -0.521836255130819\nr8: -0.627868894023207\nr9: 0.577466513963466"
        r = client.post("/world/calibration?dataset_name=test", data=calib_data, content_type="text/plain")
        assert r.status_code == 200
        r = client.get("/world/calibration?dataset_name=test")
        assert r.status_code == 200
        assert r.data == calib_data

    def test_detections_to_world_coordinates(self):
        res = run_job("/jobs/detections_to_world_coordinates?dataset_name=test&run_name=testrun&make_videos=false")
        assert res == 'success'

    def test_visualize_detections_world(self):
        res = run_job("/jobs/visualize_detections?dataset_name=test&run_name=testrun&confidence_threshold=0.6&coords=world")
        assert res == 'success'

    def test_tracking_config(self):
        config = {'mask_margin': 6.536554210404338, 'incorrect_class_cost': {'person_bicycle': 21.412795914808466, 'default': 5.204337756031922e+16, 'bicycle_person': 26.12625336049012}, 'time_region_check_thresh': 1.0215532178387352, 'cost_thresh': {'bicycle': 5.587621442965462, 'default': 20.43681365287192}, 'is_too_close_thresh': {'bicycle_bicycle': 1.5646137043645143, 'default': 1.7305655543938163}, 'creation_too_close_thresh': 5.959437096597299, 'cost_dir_weight': 0.7685414019120581, 'time_drop_thresh': 6.542976812490805, 'cost_dist_weight': 0.6283587849948333}
        r = post_json("/world/tracking_config?dataset_name=test&run_name=testrun", config)
        assert r.status_code == 200
        r = client.get("/world/tracking_config?dataset_name=test&run_name=testrun")
        assert r.status_code == 200
        assert r.json == config

    def test_tracking_world_coordinates(self):
        res = run_job("/jobs/tracking_world_coordinates?dataset_name=test&run_name=testrun&confidence_threshold=0.6&make_videos=true")
        assert res == 'success'

    def test_all_tracks_as_zip(self):
        res = run_job("/jobs/all_tracks_as_zip?dataset_name=test&run_name=testrun&tracks_format=csv&coords=world")
        assert res == 'success'
        r = client.get("/tracks/all?dataset_name=test&run_name=testrun&tracks_format=csv&coords=world")
        assert r.status_code == 200
        zip = ZipFile(BytesIO(r.data))
        assert len(zip.namelist()) == 6

