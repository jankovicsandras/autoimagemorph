#######################################################
#   
#   Automatic Image Morphing
#   András Jankovics andras@jankovics.net , 2020, adapted from
#   
#   https://github.com/ddowd97/Morphing
#   Author:     David Dowd
#   Email:      ddowd97@gmail.com
#   
#   Additional features :
#    - automatic triangle points selection by cv2.goodFeaturesToTrack()
#    - No GUI, single file program
#    - batch processing: transition between many images, not just 2
#    - optional subpixel processing to fix image artifacts
#    - automatic image dimensions safety (the dimensions of the first image defines the output)
#
#   Install dependencies:
#    pip install scipy numpy matplotlib opencv-python
#
#   Recommended postprocessing:
#    Install FFmpeg https://ffmpeg.org/
#    Example from command line:
#     ffmpeg -framerate 15 -i image%d.png output.avi
#     ffmpeg -framerate 15 -i image%d.png output.gif
#
#   TODOs:
#    - testing, error checks, sanity checks
#    - speed optimization in interpolatePoints()
#    - RGBA support, currently it's only RGB
#    - tuning the parameters of cv2.goodFeaturesToTrack() in autofeaturepoints() / giving user control
#    - built-in video output with cv2 ?
#    - image scaling uses cv2.INTER_CUBIC ; tuning / giving user control ?
#    - LinAlgError ? Image dimensions should be even numbers?
#       related: https://stackoverflow.com/questions/44305456/why-am-i-getting-linalgerror-singular-matrix-from-grangercausalitytests
#         File "batchautomorph.py", line 151, in interpolatePoints
#           righth = np.linalg.solve(tempRightMatrix, targetVertices)
#         File "<__array_function__ internals>", line 5, in solve
#         File "numpy\linalg\linalg.py",
#           line 403, in solve
#           r = gufunc(a, b, signature=signature, extobj=extobj)
#         File "numpy\linalg\linalg.py",
#           line 97, in _raise_linalgerror_singular
#           raise LinAlgError("Singular matrix")
#           numpy.linalg.LinAlgError: Singular matrix
#
#######################################################

import cv2, time, argparse, ast
from scipy.ndimage import median_filter
from scipy.spatial import Delaunay
from scipy.interpolate import RectBivariateSpline
from matplotlib.path import Path
import numpy as np

#######################################################
#   https://github.com/ddowd97/Morphing
#   Author:     David Dowd
#   Email:      ddowd97@gmail.com
#######################################################

def loadTriangles(limg, rimg, featuregridsize, showfeatures) -> tuple:
    leftTriList = []
    rightTriList = []

    lrlists = autofeaturepoints(limg,rimg,featuregridsize,showfeatures)

    leftArray = np.array(lrlists[0], np.float64)
    rightArray = np.array(lrlists[1], np.float64)
    delaunayTri = Delaunay(leftArray)

    leftNP = leftArray[delaunayTri.simplices]
    rightNP = rightArray[delaunayTri.simplices]

    for x, y in zip(leftNP, rightNP):
        leftTriList.append(Triangle(x))
        rightTriList.append(Triangle(y))

    return leftTriList, rightTriList


class Triangle:
    def __init__(self, vertices):
        if isinstance(vertices, np.ndarray) == 0:
            raise ValueError("Input argument is not of type np.array.")
        if vertices.shape != (3, 2):
            raise ValueError("Input argument does not have the expected dimensions.")
        if vertices.dtype != np.float64:
            raise ValueError("Input argument is not of type float64.")
        self.vertices = vertices
        self.minX = int(self.vertices[:, 0].min())
        self.maxX = int(self.vertices[:, 0].max())
        self.minY = int(self.vertices[:, 1].min())
        self.maxY = int(self.vertices[:, 1].max())

    def getPoints(self):
        xList = range(self.minX, self.maxX + 1)
        yList = range(self.minY, self.maxY + 1)
        emptyList = list((x,y) for x in xList for y in yList)

        points = np.array(emptyList, np.float64)
        p = Path(self.vertices)
        grid = p.contains_points(points)
        mask = grid.reshape(self.maxX - self.minX + 1, self.maxY - self.minY + 1)

        trueArray = np.where(np.array(mask) == True)
        coordArray = np.vstack((trueArray[0] + self.minX, trueArray[1] + self.minY, np.ones(trueArray[0].shape[0])))

        return coordArray

        
class Morpher:
    def __init__(self, leftImage, leftTriangles, rightImage, rightTriangles):
        if type(leftImage) != np.ndarray:
            raise TypeError('Input leftImage is not an np.ndarray')
        if leftImage.dtype != np.uint8:
            raise TypeError('Input leftImage is not of type np.uint8')
        if type(rightImage) != np.ndarray:
            raise TypeError('Input rightImage is not an np.ndarray')
        if rightImage.dtype != np.uint8:
            raise TypeError('Input rightImage is not of type np.uint8')
        if type(leftTriangles) != list:
            raise TypeError('Input leftTriangles is not of type List')
        for j in leftTriangles:
            if isinstance(j, Triangle) == 0:
                raise TypeError('Element of input leftTriangles is not of Class Triangle')
        if type(rightTriangles) != list:
            raise TypeError('Input leftTriangles is not of type List')
        for k in rightTriangles:
            if isinstance(k, Triangle) == 0:
                raise TypeError('Element of input rightTriangles is not of Class Triangle')
        self.leftImage =  np.ndarray.copy(leftImage)
        self.leftTriangles = leftTriangles  # Not of type np.uint8
        self.rightImage = np.ndarray.copy(rightImage)
        self.rightTriangles = rightTriangles  # Not of type np.uint8
        self.leftInterpolation = RectBivariateSpline(np.arange(self.leftImage.shape[0]), np.arange(self.leftImage.shape[1]), self.leftImage)#, kx=2, ky=2)
        self.rightInterpolation = RectBivariateSpline(np.arange(self.rightImage.shape[0]), np.arange(self.rightImage.shape[1]), self.rightImage)#, kx=2, ky=2)


    def getImageAtAlpha(self, alpha, smoothMode):
        for leftTriangle, rightTriangle in zip(self.leftTriangles, self.rightTriangles):
            self.interpolatePoints(leftTriangle, rightTriangle, alpha)
            # print(".", end="") # TODO: this doesn't work as intended

        blendARR = ((1 - alpha) * self.leftImage + alpha * self.rightImage)
        blendARR = blendARR.astype(np.uint8)
        return blendARR

    def interpolatePoints(self, leftTriangle, rightTriangle, alpha):
        targetTriangle = Triangle(leftTriangle.vertices + (rightTriangle.vertices - leftTriangle.vertices) * alpha)
        targetVertices = targetTriangle.vertices.reshape(6, 1)
        tempLeftMatrix = np.array([[leftTriangle.vertices[0][0], leftTriangle.vertices[0][1], 1, 0, 0, 0],
                                   [0, 0, 0, leftTriangle.vertices[0][0], leftTriangle.vertices[0][1], 1],
                                   [leftTriangle.vertices[1][0], leftTriangle.vertices[1][1], 1, 0, 0, 0],
                                   [0, 0, 0, leftTriangle.vertices[1][0], leftTriangle.vertices[1][1], 1],
                                   [leftTriangle.vertices[2][0], leftTriangle.vertices[2][1], 1, 0, 0, 0],
                                   [0, 0, 0, leftTriangle.vertices[2][0], leftTriangle.vertices[2][1], 1]])
        tempRightMatrix = np.array([[rightTriangle.vertices[0][0], rightTriangle.vertices[0][1], 1, 0, 0, 0],
                                    [0, 0, 0, rightTriangle.vertices[0][0], rightTriangle.vertices[0][1], 1],
                                    [rightTriangle.vertices[1][0], rightTriangle.vertices[1][1], 1, 0, 0, 0],
                                    [0, 0, 0, rightTriangle.vertices[1][0], rightTriangle.vertices[1][1], 1],
                                    [rightTriangle.vertices[2][0], rightTriangle.vertices[2][1], 1, 0, 0, 0],
                                    [0, 0, 0, rightTriangle.vertices[2][0], rightTriangle.vertices[2][1], 1]])
        lefth = np.linalg.solve(tempLeftMatrix, targetVertices)
        righth = np.linalg.solve(tempRightMatrix, targetVertices)
        leftH = np.array([[lefth[0][0], lefth[1][0], lefth[2][0]], [lefth[3][0], lefth[4][0], lefth[5][0]], [0, 0, 1]])
        rightH = np.array([[righth[0][0], righth[1][0], righth[2][0]], [righth[3][0], righth[4][0], righth[5][0]], [0, 0, 1]])
        leftinvH = np.linalg.inv(leftH)
        rightinvH = np.linalg.inv(rightH)
        targetPoints = targetTriangle.getPoints()  # TODO: ~ 17-18% of runtime

        leftSourcePoints = np.transpose(np.matmul(leftinvH, targetPoints))
        rightSourcePoints = np.transpose(np.matmul(rightinvH, targetPoints))
        targetPoints = np.transpose(targetPoints)

        for x, y, z in zip(targetPoints, leftSourcePoints, rightSourcePoints):  # TODO: ~ 53% of runtime
            self.leftImage[int(x[1])][int(x[0])] = self.leftInterpolation(y[1], y[0])
            self.rightImage[int(x[1])][int(x[0])] = self.rightInterpolation(z[1], z[0])

########################################################################################################


# Automatic feature points
def autofeaturepoints(leimg, riimg, featuregridsize, showfeatures):
    result = [[],[]]
    for idx, img in enumerate([leimg,riimg]) :
        try:

            if (showfeatures) : print(img.shape)
            
            # add the 4 corners to result
            result[idx] = [ [0, 0], [(img.shape[1]-1), 0], [0, (img.shape[0]-1)], [(img.shape[1]-1), (img.shape[0]-1)] ]

            h = int(img.shape[0] / featuregridsize)-1
            w = int(img.shape[1] / featuregridsize)-1
            
            for i in range(0,featuregridsize) :
                for j in range(0,featuregridsize) :
                    
                    # crop to a small part of the image and find 1 feature pont or middle point
                    crop_img = img[ (j*h):(j*h)+h, (i*w):(i*w)+w ]
                    gray = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
                    featurepoints = cv2.goodFeaturesToTrack(gray,1,0.1,10) # TODO: parameters can be tuned
                    if featurepoints is None:
                        featurepoints = [[[ h/2, w/2 ]]]
                    featurepoints = np.int0(featurepoints)
                    
                    # add feature point to result, optionally draw
                    for featurepoint in featurepoints:
                        x,y = featurepoint.ravel()
                        y = y + (j*h)
                        x = x + (i*w)
                        if (showfeatures) : cv2.circle(img,(x,y),3,255,-1)
                        result[idx].append([x,y])
            
            # optionally draw features
            if (showfeatures) : 
                cv2.imshow("",img)
                cv2.waitKey(0)

        except Exception as ex : print(ex)
    return result

#####


def initmorph(startimgpath,endimgpath,featuregridsize,subpixel,showfeatures,scale) :
    timerstart = time.time()
            
    # left image load
    leftImageRaw = cv2.imread(startimgpath)
    # scale image if custom scaling
    if scale != 1.0 :
        leftImageRaw = cv2.resize(leftImageRaw, (int(leftImageRaw.shape[1]*scale),int(leftImageRaw.shape[0]*scale)), interpolation = cv2.INTER_CUBIC)
    # upscale image if subpixel calculation is enabled
    if subpixel > 1 :
        leftImageRaw = cv2.resize(leftImageRaw, (leftImageRaw.shape[1]*subpixel,leftImageRaw.shape[0]*subpixel), interpolation = cv2.INTER_CUBIC)

    # right image load
    rightImageRaw = cv2.imread(endimgpath)
    # resize image
    rightImageRaw = cv2.resize(rightImageRaw, (leftImageRaw.shape[1],leftImageRaw.shape[0]), interpolation = cv2.INTER_CUBIC)

    leftImageARR = np.asarray(leftImageRaw)
    rightImageARR = np.asarray(rightImageRaw)

    # autofeaturepoints() is called in loadTriangles()
    triangleTuple = loadTriangles(leftImageRaw, rightImageRaw, featuregridsize, showfeatures)

    # Morpher objects for color layers BGR
    morphers = [
                Morpher(leftImageARR[:, :, 0], triangleTuple[0], rightImageARR[:, :, 0], triangleTuple[1]),
                Morpher(leftImageARR[:, :, 1], triangleTuple[0], rightImageARR[:, :, 1], triangleTuple[1]),
                Morpher(leftImageARR[:, :, 2], triangleTuple[0], rightImageARR[:, :, 2], triangleTuple[1])
               ]

    print("\r\nSubsequence init time: "+"{0:.2f}".format(time.time()-timerstart)+" s ")
    
    return morphers

####

def morphprocess(mphs,framerate,outimgprefix,subpixel,smoothing) :
    global framecnt
    
    # frame_0 is the starting image, so framecnt = _1
    framecnt = framecnt + 1
    
    # loop generate morph frames and save
    for i in range(1,framerate) :
        
        timerstart = time.time()
        alfa = i/framerate
        
        # image calculation and smoothing BGR
        if smoothing > 0 :
            outimage = np.dstack([
                np.array(median_filter(mphs[0].getImageAtAlpha(alfa, True),smoothing)),
                np.array(median_filter(mphs[1].getImageAtAlpha(alfa, True),smoothing)),
                np.array(median_filter(mphs[2].getImageAtAlpha(alfa, True),smoothing)),
            ])
        else :
            outimage = np.dstack([
                np.array(mphs[0].getImageAtAlpha(alfa, True)),
                np.array(mphs[1].getImageAtAlpha(alfa, True)),
                np.array(mphs[2].getImageAtAlpha(alfa, True)),
            ])
        
        # downscale image if subpixel calculation is enabled
        if subpixel > 1 :
            outimage = cv2.resize(outimage, ( int(outimage.shape[1]/subpixel), int(outimage.shape[0]/subpixel) ), interpolation = cv2.INTER_CUBIC)

        # write file
        filename = outimgprefix+str(framecnt)+".png"
        framecnt = framecnt + 1
        cv2.imwrite(filename,outimage)
        timerelapsed = time.time()-timerstart
        usppx = 1000000 * timerelapsed / (outimage.shape[0]*outimage.shape[1])
        print(filename+" saved, dimensions "+str(outimage.shape)+" time: "+"{0:.2f}".format(timerelapsed)+" s ; μs/pixel: "+"{0:.2f}".format(usppx) )

####

def batchmorph(imgs,featuregridsize,subpixel,showfeatures,framerate,outimgprefix,smoothing,scale) :
    global framecnt
    framecnt = 0
    totaltimerstart = time.time()
    for idx in range(len(imgs)-1) :
        morphprocess(
            initmorph(imgs[idx],imgs[idx+1],featuregridsize,subpixel,showfeatures,scale),
            framerate,outimgprefix,subpixel,smoothing
        )
    print("\r\nDone. Total time: "+str(time.time()-totaltimerstart)+" s ")

###############################################################################

mfeaturegridsize = 7 # number of image divisions on each axis, for example 5 creates 5x5 = 25 automatic feature points + 4 corners come automatically
mframerate = 30 # number of transition frames to render + 1 ; for example 30 renders transiton frames 1..29
moutprefix = "f" # output image name prefix
framecnt = 0 # frame counter
msubpixel = 1 # int, min: 1, max: no hard limit, but 4 should be enough
msmoothing = 0 # median_filter smoothing
mshowfeatures = False # render automatically detected features
mscale = 1.0 # image scale

# batch morph process
#batchmorph(["f0.png","f30.png","f60.png","f90.png","f120.png","f150.png"],mfeaturegridsize,msubpixel,mshowfeatures,mframerate,moutprefix,msmoothing)

# CLI arguments

margparser = argparse.ArgumentParser(description="Automatic Image Morphing https://github.com/jankovicsandras/autoimagemorph adapted from https://github.com/ddowd97/Morphing")
margparser.add_argument("-inframes", default="", required=True, help="REQUIRED input filenames in a list, for example: -inframes ['f0.png','f30.png','f60.png']")
margparser.add_argument("-outprefix", default="", required=True, help="REQUIRED output filename prefix, -outprefix f  will write/overwrite f1.png f2.png ...")
margparser.add_argument("-featuregridsize", type=int, default=mfeaturegridsize, help="Number of image divisions on each axis, for example -featuregridsize 5 creates 25 automatic feature points. (default: %(default)s)")
margparser.add_argument("-framerate", type=int, default=mframerate, help="Frames to render between each keyframe +1, for example -framerate 30 will render 29 frames between -inframes ['f0.png','f30.png'] (default: %(default)s)")
margparser.add_argument("-subpixel", type=int, default=msubpixel, help="Subpixel calculation to avoid image artifacts, for example -subpixel 4 is good quality, but 16 times slower processing. (default: %(default)s)")
margparser.add_argument("-smoothing", type=int, default=msmoothing, help="median_filter smoothing/blur to remove image artifacts, for example -smoothing 2 will blur lightly. (default: %(default)s)")
margparser.add_argument("-showfeatures", action="store_true", help="Flag to render feature points, for example -showfeatures")
margparser.add_argument("-scale", type=float, default=mscale, help="Input scaling for preview, for example -scale 0.5 will halve both width and height, processing will be approx. 4x faster. (default: %(default)s)")

args = vars(margparser.parse_args())

# arguments sanity check TODO

if( len( args['inframes'] ) < 2 ) :
    print("ERROR: command line argument -inframes must be a string array with minimum 2 elements.")
    print("Example\r\n > python autoimagemorph.py -inframes ['frame0.png','frame30.png','frame60.png'] -outprefix frame ")
    quit()

if( len(args['outprefix']) < 1 ) :
    print("ERROR: -outprefix (output filename prefix) must be specified.")
    print("Example\r\n > python autoimagemorph.py -inframes ['frame0.png','frame30.png','frame60.png'] -outprefix frame ")
    quit()
args['inframes'] = ast.literal_eval(args['inframes'])

args['featuregridsize'] = int(args['featuregridsize'])

args['subpixel'] = int(args['subpixel'])

args['framerate'] = int(args['framerate'])

args['smoothing'] = int(args['smoothing'])

args['scale'] = float(args['scale'])

print("User input: \r\n"+str(args))
    
# processing

batchmorph(args['inframes'],args['featuregridsize'],args['subpixel'],args['showfeatures'],args['framerate'],args['outprefix'],args['smoothing'],args['scale'])

