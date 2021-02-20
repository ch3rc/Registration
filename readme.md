##How to run
___
python3 register.py [-h] [-M manual registration] [-e epsilon] [-m motion] [-o output] \
[-w warped] image_file template_file [warp_file]
___
- -h --help:        Brings up a help message
- -M --manual:      Perform manual image registration by picking 4 points on both input image 
                    and template image
- -e --epsilon:     cv2.findTransformECC()'s convergence epsilon [default: 0.0001]
- -m --motion:      Type of motion (translation/euclidean/affine/homography) [default: affine]
- -o --output:      Output warp matrix filename [default: out_warp.ecc] 
- -w --warped:      Warped Output image [default: warped_ecc.jpg]
- image_file:       Input image (to be warped/aligned)
- template_file:    Template image for alignment
- warp_file:        Input file containing warp matrix. warp matrix must match motion type
                    else an identity matrix of the correct size will be created in its place

___
##Known issues
___
Affine transformation tends to separate color channels for some reason.
___
##links
___
- [github](www.https://github.com/ch3rc/Registration "github account") for code and logs under Registration 
main branch
- contact me at my [UMSL email](ch3rc@umsystem.edu) if you have any questions