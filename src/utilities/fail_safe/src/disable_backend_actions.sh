#!/bin/bash
set -x
rosparam set /fail_safe/should_notify_backend 0
rosparam set /fail_safe/should_post_issue 0
rosparam set /fail_safe/should_send_bags 0
