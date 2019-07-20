#!/usr/bin/env bash

sudo apt update && sudo apt install -y wget unzip cmake libjpeg8-dev
wget https://github.com/jacksonliam/mjpg-streamer/archive/master.zip && unzip master.zip
pushd mjpg-streamer-master/mjpg-streamer-experimental
make
sudo make install
popd
rm -Rf mjpg-streamer-master master.zip
cat <<- 'EOF' | sudo tee /etc/systemd/system/mjpg-streamer_rpi-camera.service
	[Unit]
	Description=MJPG-Streamer - Raspberry Pi Camera
	
	[Service]
	ExecStartPre=/bin/bash -c '! /usr/bin/vcgencmd get_camera | grep -F "detected=0"'
	ExecStart=/usr/local/bin/mjpg_streamer -i "input_raspicam.so -x 1280 -y 720" -o "output_http.so -w /usr/local/share/mjpg-streamer/www -p 8081"
	Restart=always
	RestartSec=15
	
	[Install]
	WantedBy=multi-user.target
EOF
sudo systemctl enable mjpg-streamer_rpi-camera.service
sudo systemctl start mjpg-streamer_rpi-camera.service
cat <<- 'EOF' | sudo tee /etc/systemd/system/mjpg-streamer_usb-camera-0.service
	[Unit]
	Description=MJPG-Streamer - USB camera 0
	BindsTo=dev-video0.device
	After=dev-video0.device
	
	[Service]
	ExecStart=/usr/local/bin/mjpg_streamer -i "input_uvc.so -n -r 1280x720 -d /dev/video0 -f 15" -o "output_http.so -w /usr/local/share/mjpg-streamer/www -p 8082"
	Restart=always
	RestartSec=5
EOF
cat <<- 'EOF' | sudo tee /etc/systemd/system/mjpg-streamer_usb-camera-1.service
	[Unit]
	Description=MJPG-Streamer - USB camera 1
	BindsTo=dev-video1.device
	After=dev-video1.device
	
	[Service]
	ExecStart=/usr/local/bin/mjpg_streamer -i "input_uvc.so -n -r 1280x720 -d /dev/video1 -f 15" -o "output_http.so -w /usr/local/share/mjpg-streamer/www -p 8083"
	Restart=always
	RestartSec=5
EOF
cat <<- 'EOF' | sudo tee /etc/udev/rules.d/99-video-systemd.rules
	KERNEL=="video0", SYMLINK="video0", TAG+="systemd", ENV{SYSTEMD_WANTS}="mjpg-streamer_usb-camera-0.service"
	KERNEL=="video1", SYMLINK="video1", TAG+="systemd", ENV{SYSTEMD_WANTS}="mjpg-streamer_usb-camera-1.service"
EOF
