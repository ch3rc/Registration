"""
Author:     Cody Hawkins
Date:       2/16/21
Class:      CS6420
Desc:       Demonstration of manual or automatic
            registration on an image
"""

import cv2 as cv
import numpy as np
import getopt
import sys


def help_me():
    print("\n{:>10}{:->15}{}{:->15}".format("","-","HELP","-"))
    print("[-M or --manual]:    Perform manual registration by picking your location points")
    print("[-e or --epsilon]:   ECC's convergence epsilon [default: 0.0001]")
    print("[-m or --motion]:    Type of motion (translation:0/Euclidean:1/affine:2/homograpy:3) [default: affine:2]")
    print("[-o or --output]:    Output warp matrix filename [default: out_warp.ecc]")
    print("[-w or --warped]:    Warped image output [default: warped_ecc.jpg]")
    print("image_file:          Provide input image (to be warped/aligned)")
    print("template_file:       Provide template image for alignment")
    print("warp_file:           Provide input file containing warp matrix [optional]")


def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hMe:m:o:w:", ["help", "manual", "epsilon", "motion", "output",
                                                                "warped"])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(1)

    manual = False
    epsilon = 0.0001
    motion = 2
    output = "out_warp.ecc"
    warped = "warped_ecc.jpg"
    image_file = None
    template_file = None
    warp_file = None

    for o, a in opts:
        if o in ("-h", "--help"):
            help_me()
            sys.exit(1)
        elif o in ("-M", "--manual"):
            manual = True
        elif o in ("-e", "--epsilon"):
            epsilon = float(a)
            if epsilon > 0.1:
                epsilon = 0.1
            if epsilon < 0.0001:
                epsilon = 0.0001
        elif o in ("-m", "--motion"):
            motion = int(a)
            motions = [0, 1, 2, 3]
            if motion not in motions:
                print("Input entered is not a valid option")
                help_me()
                sys.exit(1)
        elif o in ("-o", "--output"):
            output = a
        elif o in ("-w", "--warped"):
            warped = a
        else:
            assert False, "Unhandled Option!"

    if len(args) < 2:
        print("Too few input files!")
        help_me()
        sys.exit(1)
    elif len(args) is 2:
        image_file = args[0]
        template_file = args[1]
    elif len(args) is 3:
        image_file = args[0]
        template_file = args[1]
        warp_file = args[2]
    elif len(args) > 3:
        print("Too many input files!")
        help_me()
        sys.exit(1)


if __name__ == "__main__":
    main()