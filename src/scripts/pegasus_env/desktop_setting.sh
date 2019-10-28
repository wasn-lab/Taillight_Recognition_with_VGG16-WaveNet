#!/bin/bash
gsettings set org.gnome.desktop.session idle-delay 0
gsettings set org.gnome.desktop.screensaver lock-enabled false
gsettings set org.gnome.desktop.media-handling automount-open false
gsettings set org.compiz.unityshell:/org/compiz/profiles/unity/plugins/unityshell/ shortcut-overlay false
chown -R nvidia:nvidia /home/nvidia/.config

