#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "Starting Phishing Scraper API..."
"$DIR/app" &
sleep 3
echo "Starting DNS Monitor (requires password for port 53)..."
sudo "$DIR/dns_monitor_db"
echo "Shutting down services..."
killall app

echo "Done."