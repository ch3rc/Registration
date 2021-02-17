"""
Author:     Cody Hawkins
Class:      CS5420
Date:       2/16/21
Desc:       Search through provided file path
            for input image and return result(s)
"""

import os
import sys


def find(filename, search):
    # search through files and return path of image
    result = []
    for root, dirs, files in os.walk(search):
        if filename in files:
            result.append(os.path.join(root, filename))
    if len(result) == 0:
        print("Could not find file!")
        sys.exit(0)

    return result[0]
