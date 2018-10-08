import numpy as np
import cv2 as cv
import sys


# Notas - Andre Costa
# Este é apenas um arquivo exemplo de como usar funções básicas do OpenCV
# Não manjo quase nada de Python, então muito provavelmente farei coisas de uma forma burra
# Desculpem o vacilo e obrigado pela compreensão. Estou 100% aberto a sugestões e a aprender...

imagePath   = sys.argv[1]

# cv.imread() - Reads an image and store it in a numpy array
image = cv.imread(imagePath)

# .shape property gets image properties
rows,cols,channels = image.shape        
print("Image Shape", image.shape, "| Size", image.size, "| Dtype", image.dtype)        

# cv.imshow(windowName, image) - Shows image
cv.imshow("RawImage",image)   

# We can use numpy indexing to Get ROI (Region of Interest of an image)
imageROI = image[200:220,200:220]
cv.imshow("imageROI",imageROI)          

# This junk is necessary... otherwise the window will open and close and one won't be able to see the image T_T             
cv.waitKey(0)
cv.destroyAllWindows() 


