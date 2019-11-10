# SmartStop
a smarter stop sign using Raspberry Pi with camera module and a Pytorch-built Unet recognition model

1. works by first cd /d to directory of the Server scripts
2. run python ServerScriptPCRecieve.py
..* NOTE: RECALL CHANGE PUBLIC IP IN PI SCRIPTS (adjust IP name upon reconnecting to WIFI)
3. run both laptop scripts first (they wait for connection to rPI)
4. after laptop ready, 
5. run python ClientScriptPYSend.py (will stop automatically after 2 minutes)
6. run python ClientScriptPYrecieve.py
