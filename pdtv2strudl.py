import os
from glob import glob


os.system("rm -r cam/vid*")

for fcnt, fn in enumerate(sorted(glob("cam/*.jpg"))):
    v = fcnt // 100
    fd = open("cam/vid%d.log" % v, "a")
    ts = os.path.basename(fn).replace(".jpg", "")
    date, time = ts.split("-")
    print('%.5d' % (fcnt % 100), date[:4], date[4:6], date[6:8], time[:2] + ':' + time[2:4] + ':' + time[4:], file=fd)
    os.makedirs("cam/vid%d" % v, exist_ok=True)
    os.link(fn, 'cam/vid%d/%.5d.jpg' % (v, fcnt % 100))

for vid in glob("cam/vid?"):
    os.system("ffmpeg -i %s/%%05d.jpg -c:v libx264  %s.mp4" % (vid, vid))
