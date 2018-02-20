#!/usr/bin/env python3.6

from datetime import datetime

def get_path():

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    return "{}/run-{}/".format(root_logdir, now)
