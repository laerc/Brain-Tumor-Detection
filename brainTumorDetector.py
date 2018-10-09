import numpy as np
import cv2 as cv
import sys
import math

# Usage: python3 brainTumorDetector.py filename K_value
# TODO - Implement argparse...

STOP_THRESHOLD = 0

class Detector:
    def detect(self, imagePath,k):

        # 1) Pre-processing - The article states that
        #       "The preprocessing stage will convert the RGB input image to grey scale. 
        #       Noise present if any, will be removed using a median filter"
        
        # Opens MRI image and convert it to grayscale
        MRIImg = cv.imread(imagePath,cv.IMREAD_GRAYSCALE)
        rows,cols = MRIImg.shape       

        # Runs Median Filter over the image - TODO Test different kernel sizes
        MRIImgMedian = cv.medianBlur(MRIImg,3)

        # 2) K-means clustering
        #   2.1) Let x1,…, xM are N data points in the input image, let k be the number of clusters which is given by the user.      
        #   2.2) Choose c1,…, cK cluster centres.
        #   2.3) Distance between each pixel and each cluster centre is found.
        #   2.4) The distance function is given by 
        #        J=|xi - cj| for i=1,…,N and for j=1,…,k, where |xi-Cj|, the absolute difference of the distance  
        #        between a data point and the cluster centre indicates the distance of the N data points from their
        #        respective cluster centers.
        #   2.5) Distribute the datapoints x among the k clusters using the relation x∈Cj if |x-cj|<|x-ci| for i=1,2,…,k, i≠j,
        #         where Cj denotes the set of data points whose cluster centre is cj
        #   2.6) Updated cluster centre is given as, ci= 1/m*(SUM x∈Ci, for i=1,…,k, where mi is the number of objects in the 
        #        dataset Ci,where is the i-th cluster and ci is the centre of cluster Ci.
        #   2.7) Repeat from Step 5 to Step 8 till convergence is met.
        #   2.8) After segmentation and detection of the desired region, there are chances for misclustered regions to occur 
        #        after the segmentation algorithm, hence morphological filtering is performed for enhancement of the tumor 
        #        detected portion. Here structuring element used is disk shaped.           

        # Declares array to store k centers and uniformly distributes values from 0-255 range
        centers = np.zeros(k,'float') 
        for i in range(k):
            centers[i] = (1+i)*int(255/k)

        # Variable to store segmented outputs
        outputImages = np.zeros((k,rows,cols),'float')                     

        # Run till convergence is met
        iteration = 0
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
            for i in range(k):
                newCenter =  float(centerSumPixel[i])/float(centerCount[i])
                centersDiffSum += abs(centers[i] - newCenter) # TODO - REVER.... pode ser melhorado para convergir mais rapidamente
                centers[i] = newCenter
               
            print("Centers",centers)
            print("",centersDiffSum)               
        
            # When the sum of the errors from the the centers is less than STOP_THRESHOLD, converged...
            if (centersDiffSum <= STOP_THRESHOLD):
                break

        # 3) Morphological Filtering

        # Creates structuring element used on the morphological filtering
        ksize = math.ceil(1.5*rows/100)
        print("ksize",ksize)

        strel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(ksize,ksize))

        cv.imshow("Input / Median Filter",np.hstack((MRIImg,MRIImgMedian)))

        # For each output image        
        for i in range(k):            
            
            # Opening filter   
            opening = cv.morphologyEx(outputImages[i], cv.MORPH_OPEN, strel)

            out = np.hstack((outputImages[i],opening))
            cv.imshow("K-Means / Morphological ("+str(i)+")",out)

            outputImages[i] = opening  

            if(save):
                cv.imwrite(str(i)+"_"+imagePath, opening)
            
        cv.waitKey(0)
        cv.destroyAllWindows() 

if __name__ == "__main__":
    imagePath   = sys.argv[1]
    k           = sys.argv[2] 

    if(len(sys.argv)>3 and sys.argv[3] == "--save"):   
        save        = True   
    else:
        save        = False

    x = Detector()
    x.detect(imagePath,int(k))

  