# SmartStop
a smarter stop sign using Raspberry Pi with camera module and a Pytorch-built Unet recognition model

works by first cd /d to directory of the Server scripts
run python ServerScriptPCRecieve.py
NOTE: RECALL CHANGE PUBLIC IP IN PI SCRIPTS (adjust IP name upon reconnecting to WIFI)
run both laptop scripts first (they wait for connection to rPI)
after laptop ready, 
run python ClientScriptPYSend.py (will stop automatically after 2 minutes)
run python ClientScriptPYrecieve.py
