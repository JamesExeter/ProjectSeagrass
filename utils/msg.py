#!/usr/bin/env python
"""
Class dedicated to printing messages with the time of program executing prepending the message
Also used to log the details of execution to a logging file
"""

import time
import math
import logging

# Initialisation of the class with the start time and logging set up
def ini(path):
    global start_time
    start_time = time.time()
    logging.basicConfig(filename=path,level=logging.DEBUG, format='%(asctime)s %(message)s')

# prints the message with the time prepended with respect to start of program execution
def timemsg (msg):
    global start_time
    elapsed_time = time.time() - start_time # seconds

    hour = math.floor( elapsed_time / 3600 )
    minute = math.floor( (elapsed_time%3600) / 60 )
    second = math.floor( elapsed_time%60 )
    output = str(hour) +  ":" +  str(minute) +  ":" + str(second) + " " + msg
    print(output)
    logging.info(output)

# prints out a message in debug mode
def debug(msg):
    print(msg)
    logging.debug(msg)