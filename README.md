
![alt Automatic Image Morphing logo](f.gif)

# Automatic Image Morphing

### Version 1.0.0
### This is a Python command line application.

## This is adapted from Python Image Morpher (PIM) by David Dowd. https://github.com/ddowd97/Morphing
David's Python Image Morpher (PIM) is a great tool with a nice GUI. You should use it when you want to select the morph-points manually. This leads to much better results, at the expense of manual work.
```autoimagemorph.py``` selects the morph-points automatically using OpenCV ```cv2.goodFeaturesToTrack()``` . No manual work required, starting the command will generate and save the image sequence (animation frames).

## Additional features:
- automatic triangle points selection using ```cv2.goodFeaturesToTrack()```
- No GUI, single file command line program
- batch processing: transition between many images, not just 2
- optional subpixel processing to fix image artifacts
- automatic image dimensions safety (the dimensions of the first image defines the output)

## Warning
Be careful when using the required ```-outprefix``` parameter.  The program overwrites ```<outprefix><sequencenumber>.png``` files without warning. Example: ```... -outprefix f ...``` can overwrite ```f1.png```, ```f2.png``` ... Backup your png files before running the program and avoid name conflicts.

## Examples
### Help:
```python autoimagemorph.py -h```

### This will create and save ```f1.png```, ```f2.png```, ... ```f29.png```, then ```f31.png```, ```f32.png```, ... ```f59.png``` creating a continuous image sequence between the keyframes. The keyframes ```f0.png```, ```f30.png``` and ```f60.png``` will not be modified (overwritten), but only if the framerate matches their filename.
```python autoimagemorph.py -inframes ['f0.png','f30.png','f60.png'] -frameprefix f -framerate 30```

### This is how the logo ```f.gif``` was created:

1. Got some [van Gogh self portraits from Wikipedia](https://en.wikipedia.org/wiki/Vincent_van_Gogh) , converted and renamed them to keyframes ```f0.png```, ```f30.png```, ...

2. Ran this and took a nap. :)  The process took more than an hour.
```python autoimagemorph.py -inframes "['f0.png','f30.png','f60.png','f90.png','f120.png','f150.png','f0.png']" -frameprefix f -framerate 30 -subpixel 4```

3. FFmpeg postprocessing:
```ffmpeg -framerate 15 -i f%d.png f.gif```

## Install dependencies:
```pip install scipy numpy matplotlib opencv-python```

## Recommended postprocessing:
Install FFmpeg https://ffmpeg.org/
Then, from command line:
```ffmpeg -framerate 15 -i frame%d.png output.avi```
```ffmpeg -framerate 15 -i frame%d.png output.gif```

## TODO:
- testing, error checks, sanity checks
- speed optimization in interpolatePoints()
- RGBA support, currently it's only RGB
- tuning the parameters of cv2.goodFeaturesToTrack() in autofeaturepoints() / giving user control
- built-in video output with cv2 ?
- image scaling uses cv2.INTER_CUBIC ; tuning / giving user control ?
- LinAlgError sometimes? Image dimensions should be even numbers?

## License
### The Unlicense / PUBLIC DOMAIN

This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to [http://unlicense.org](http://unlicense.org)
