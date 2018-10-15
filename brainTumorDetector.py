# slides : https://www.overleaf.com/project/5bb8c845f83b4a4c4a710e13
# dataset : https://figshare.com/articles/brain_tumor_dataset/1512427
 
import numpy as np
import cv2 as cv
import sys
import math

# Usage: python3 brainTumorDetector.py filename K_value
# TODO - Implement argparse...

STOP_THRESHOLD = 0
USE_OPENCV_FILTERS = False
STEP_BY_STEP = False
SHOW_MORPHO = True

class Detector:

    def medianFilter(self, MRIImg, window_size):
        rows,cols = MRIImg.shape
        k = int(window_size/2)
        outImage = MRIImg.copy()

        for i in range(k,rows-k):
            for j in range(k,cols-k):
                tmp_list = []
                for x in range(i-k,i+k+1):
                    for y in range(j-k,j+k+1):
                        tmp_list.append(MRIImg[x][y])

                tmp_list.sort()
                outImage[i][j] = tmp_list[int(len(tmp_list)/2)]

        return outImage

    def doDilation(self, img, window_size, kernel):
        rows,cols = img.shape
        k = int((window_size)/2)
        outImage = img.copy()

        for i in range(k,rows-k):
            for j in range(k,cols-k):
                max_val = outImage[i][j]
                ok = True  
                for x in range(i-k,i+k+1):
                    for y in range(j-k,j+k+1):
                        if(kernel[x-(i-k)][y-(j-k)] == 0):
                            continue
                        max_val = max(max_val, img[x][y])
                if(ok == True):
                    outImage[i][j] = max_val
                
        return outImage        

    def doErosion(self, img, window_size, kernel):
        rows,cols = img.shape
        k = int((window_size)/2)
        outImage = img.copy()

        for i in range(k,rows-k):
            for j in range(k,cols-k):
                min_val = outImage[i][j]
                ok = True
                for x in range(i-k,i+k+1):
                    for y in range(j-k,j+k+1):
                        if(kernel[x-(i-k)][y-(j-k)] == 0):
                            continue
                        min_val = min(min_val, img[x][y])
                if(ok == True):
                    outImage[i][j] = min_val

        return outImage

    def doOpen(self, img, window_size, kernel):
        return self.doDilation(self.doErosion(img, window_size, kernel), window_size, kernel);

    def detect(self, imagePath, k, ksize):

        # 1) Pre-processing 
        # Opens MRI image and convert it to grayscale
        MRIImg = cv.imread(imagePath,cv.IMREAD_GRAYSCALE)
        rows,cols = MRIImg.shape 

        # Runs Median Filter over the image - TODO Test different kernel sizes
        print("Applying Median Filter")
        if(USE_OPENCV_FILTERS):
            MRIImgMedian = cv.medianBlur(MRIImg,3)  
        else:
            MRIImgMedian = self.medianFilter(MRIImg,3)          

        cv.imshow("Input / Median Filter",np.hstack((MRIImg,MRIImgMedian)))    
        if(STEP_BY_STEP):
            cv.waitKey(0)

        # 2) K-means clustering
         # Declares array to store k centers and uniformly distributes values from 0-255 range
        centers = np.zeros(k,'float') 
        for i in range(k):
            centers[i] = int((i*255)/(k))

        # Variable to store segmented outputs
        outputImages = np.zeros((k,rows,cols),'float')                  

        # Run till convergence is met
        iteration = 0
        temp2 = None
        while True:
            iteration = iteration+1   
            print("--- Running Iteration "+str(iteration))         
            print("Centers", centers) 

            # Variables used to update center values
            centerSumPixel  = np.zeros(k,'float')             
            centerCount     = np.zeros(k,'float')             
                         
            # For each pixel...          
            for row in range(rows):
                for col in range(cols):

                    minDist     = 1000 
                    minCenter   = None             
                    pixelValue  = MRIImg[row][col] 
                    
                    # Finds minimum distance from centers values
                    for i in range(k):
                        centerValue = centers[i]
                        dist = abs(float(pixelValue)-float(centerValue))
                        
                        if(dist<minDist):
                            minDist = dist
                            minCenter = i

                    # After finding minimum distance we know to which cluster the pixel belongs
                    outputImages[minCenter][row][col] = pixelValue

                    # Increments variables used to Update center values
                    centerSumPixel[minCenter]   = centerSumPixel[minCenter] + pixelValue             
                    centerCount[minCenter]      = centerCount[minCenter] + 1
    
            # Variable used to stablish the breakpoint
            centersDiffSum = 0

            # Update center values
            temp = MRIImg.copy()
            for i in range(k):
                newCenter =  float(centerSumPixel[i])/float(centerCount[i])
                centersDiffSum += abs(int(centers[i]) - int(newCenter)) 
                centers[i] = newCenter

                temp = np.hstack((temp,outputImages[i]))
                if(i == k and iteration>1):
                    temp2 = np.vstack((temp2,temp))
                else:   
                    temp2 = temp

            if(STEP_BY_STEP):    
                cv.imshow("Iteration "+str(iteration),temp)                    
                cv.waitKey(0)                
               
            print("Centers",centers)
            print("",centersDiffSum)               
        
            # When the sum of the errors from the the centers is less than STOP_THRESHOLD, converged...
            if (centersDiffSum <= STOP_THRESHOLD):
                break             

        # 3) Morphological Filtering

        # Creates structuring element used on the morphological filtering
        #ksize = int(math.ceil(1.5*rows/100))        
        strel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(ksize,ksize))
        print("ksize",ksize)

        # For each output image        
        for i in range(k):            
            
            # Opening filter
            if(USE_OPENCV_FILTERS):
                opening = cv.morphologyEx(outputImages[i], cv.MORPH_OPEN, strel)
            else:
                opening = self.doOpen(outputImages[i], ksize, strel)

            if(SHOW_MORPHO):
                out = np.hstack((outputImages[i],opening))
                cv.imshow("K-Means / Morphological ("+str(i)+")",out)
            else:
                cv.imshow("K-Means ("+str(i)+")",(outputImages[i]))

            if(STEP_BY_STEP):                   
                cv.waitKey(0)              

            if(save):
                cv.imwrite(str(i)+"_"+imagePath, opening)
            
            outputImages[i] = opening  
        
        cv.waitKey(0)
        cv.destroyAllWindows() 

if __name__ == "__main__":
    imagePath   = sys.argv[1]
    k           = sys.argv[2] 
    ksize       = sys.argv[3]     

    if(len(sys.argv)>4 and sys.argv[4] == "--save"):   
        save        = True   
    else:
        save        = False

    x = Detector()
    x.detect(imagePath,int(k),int(ksize))

  