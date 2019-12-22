# SmartStop

### About
Created for the Decode Congestion hackathon hosted by the City of Vancouver!

SmartStop is a very cheap computer vision-based recognition software that controls 3 or 4 way stop intersections in a more intuitive way. SmartStop uses a Raspberry Pi with a camera module that controls a blinking red LED light. While there are no cars, the LED blinks continuously. It detects when a car arrives at the intersection, and stops blinking the light given that it is safe to cross. Simply, blinking means STOP, not blinking means GO. SmartStop is meant to reduce some of the ambiguity that can arise at 4-way intersections, especially when multiple cars come up at once. However, importantly, it does NOT change the rules of the road. Whoever arrives at the stop sign first is allowed to pass first, and pedestrians are given the right of way.

### Notes about running
1. works by first cd /d to directory of the Server scripts
2. run python ServerScriptPCRecieve.py
..* NOTE: RECALL CHANGE PUBLIC IP IN PI SCRIPTS (adjust IP name upon reconnecting to WIFI)
3. run both laptop scripts first (they wait for connection to rPI)
4. after laptop ready, 
5. run python ClientScriptPYSend.py (will stop automatically after 2 minutes)
6. run python ClientScriptPYrecieve.py

### Output images
These are some sample outputs from the segmentation model.

For an image with no cars: <img src="https://github.com/Tonyxu74/SmartStop/blob/master/noCar.png" width="200" height="200"><img src="https://github.com/Tonyxu74/SmartStop/blob/master/noCar_mask.png" width="200" height="200">

Segmenting an image with cars: <img src="https://github.com/Tonyxu74/SmartStop/blob/master/test_raw.png" width="200" height="200"><img src="https://github.com/Tonyxu74/SmartStop/blob/master/test_mask.png" width="200" height="200">
