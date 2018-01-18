#!/usr/bin/env python3.6

from datetime import datetime


def get_path(file_name):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_log_dir = "tf_logs"
    return "{}/{}-{}/".format(root_log_dir, file_name, now)

