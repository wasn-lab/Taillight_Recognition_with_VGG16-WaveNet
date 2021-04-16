#!/bin/bash

sleep 30;
# Old setting, streamed images freeze after 8 seconds.
# firefox "https://service.itriadv.co:8784/Unit/DriverDashboard?URL=local&R=true"
chromium-browser "http://service.itriadv.co:8785/Unit/DriverDashboard?URL=local&R=true"
