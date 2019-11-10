import socket                   # Import socket module
import time
import pygame
from pygame.locals import*

####pygame globals######
WIDTH = 800
HEIGHT = 480
pygame.init()
windowSurface = pygame.display.set_mode((WIDTH, HEIGHT), 0, 0)
flash = pygame.Color(255,0,0)
noFlash = pygame.Color(0,0,0)

# flashing alternates to flash colors
flashing = 0
######################################

#s = socket.socket()             # Create a socket object
port = 60000                    # Reserve a port for your service.

# 0 flash       to stop
# 1 no flash    to go
def chooseIm(num):
    global flashing
    print(flashing)
    if (num == 0 and flashing == 0):
        windowSurface.fill(flash)
        flashing = 1
    elif (num == 0 and flashing == 1):
        windowSurface.fill(noFlash)
        flashing = 0
    else:
        windowSurface.fill(noFlash)
        flashing = 0
        
    pygame.display.update()


while True:
    s = socket.socket() 
    s.connect(('206.12.45.91', port))
    print('receiving data...')
    data = s.recv(1024)
    chooseIm(int(data))
    print('data=%s', (data))
    print('Successfully get the file')
    s.close()
    s=-1
    time.sleep(0.8)
    
            



