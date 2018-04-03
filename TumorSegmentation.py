import dicom
import numpy 
import nrrd
import math
import os
from sklearn.cluster import KMeans
import morphsnakes
from scipy.misc import imread
from matplotlib import pyplot, cm
import pylab
	
def MIP_Return(IMGSeries):
    maxarray=numpy.zeros((int(IMGSeries.shape[0]), int(IMGSeries.shape[1]), int(IMGSeries.shape[2])))
    oneDimgarr=[]
    
    for i in range(IMGSeries.shape[2]):
        arr = IMGSeries[:,:,i].reshape(IMGSeries[:,:,i].shape[0]*IMGSeries[:,:,i].shape[1],1)
        oneDimgarr.append(arr)
    
    maxarr=numpy.maximum(oneDimgarr[0], oneDimgarr[1])
    for i in range(2,len(oneDimgarr)):
        maxarr= numpy.maximum(maxarr, oneDimgarr[i])
                   
    MipImage = maxarr.reshape(math.sqrt(len(maxarr)), math.sqrt(len(maxarr)))            
    return(MipImage)
	
def segment(image, tumorxcoord, tumorycoord, num_clusts):
    x1=0
    y1=0
    x2=image.shape[0]
    y2 = image.shape[1]
        
    #Add cluster atttribute distance, from center bottom?    
    
    clt = KMeans(n_clusters = num_clusts, random_state=0)
    
    flattenedImage=image[x1:x2, y1:y2].reshape(image[x1:x2,y1:y2].shape[0]*image[x1:x2,y1:y2].shape[1],1)
    avgsurrpixels = avgsurrvalues(image, x1,x2,y1,y2)
    
    arr=[]
    
    for i in range(0,len(avgsurrvalues(image, x1,x2,y1,y2))):
        arr.append([flattenedImage[i], avgsurrpixels[i]])

    clt.fit(arr) 
    
    return clt.labels_
	
def avgsurrvalues(img, x1, x2, y1, y2):
    
    avgsurrvaluesvec = []

    for i in range(x1,x2):
        for j in range(y1,y2):
            if(i==0):
                if(j==0):
                    avg =int(numpy.mean([img[x1,y1+1], img[x1+1,y1], img[x1+1,y1+1]]))
                    
                if(j==(img.shape[1]-1)):
                    avg = int(numpy.mean([img[x1,y2-2], img[x1+1,y2-1], img[x1+1,y2-2]]))
                
                else: 
                    avg=int(numpy.mean([img[i+1,j+1], img[i,j+1], img[i+1,j], img[i,j-1], img[i+1,j-1]]))
                        #bot right, right, below, above, top right
            elif(i==(img.shape[0]-1)):
                if(j==0):
                    avg=int(numpy.mean([img[x2-2,y1], img[x2-1, y1+1], img[x2-2, y1+1]]))
                if(j==(img.shape[1]-1)):
                    avg=int(numpy.mean([img[x2-1, y2-2], img[x2-2, y2-1], img[x2-2, y2-2]]))
                else:
                    avg=int(numpy.mean([img[i-1,j-1], img[i-1,j], img[i,j+1],img[i,j-1],img[i-1,j+1]]))  
            
            elif(j==0 and i!=0 and i!=img.shape[0]-1):
                avg=int(numpy.mean([img[i-1, j], img[i+1, j], img[i-1,j+1], img[i+1,j+1], img[i,j+1]]))
            elif(j==img.shape[1]-1 and i!= 0 and i != img.shape[0]-1):
                avg=int(numpy.mean([img[i-1, j], img[i+1, j], img[i-1,j-1], img[i+1,j-1], img[i,j-1]]))
            
            else:
                avg= int(numpy.mean([img[i+1,j], img[i-1,j],img[i,j-1],img[i,j+1],img[i-1,j-1],img[i+1,j+1], img[i-1,j+1],img[i+1,j-1]]))
            
            avgsurrvaluesvec.append(avg)
        
    return avgsurrvaluesvec
	
def circle_binary_set(shape, center, radius, scalerow=1.0):

    bin_grid = numpy.mgrid[list(map(slice, shape))].T - center
    rad = radius - numpy.sqrt(numpy.sum((bin_grid.T)**2, 0))
    result = numpy.float_(rad > 0)

    return result
	
def morph_snake(img, ROI, radius):
    img1 = img/255.0
    # g(I)
    gI = morphsnakes.gborders(img1, alpha=1000, sigma=11)

    mgac = morphsnakes.MorphACWE(img1, smoothing=3, lambda1=1, lambda2=1)
    mgac.levelset = circle_binary_set(img1.shape,ROI , radius)
    
    # Visual evolution.
    pylab.figure(figsize=(8, 6), dpi=400)
    return(morphsnakes.evolve_visual(mgac, num_iters=30, background=img1)) #110
    pylab.show()

def trigtimetumordetect(dicomImgArr):
    ##Finds trigger times that contain tumors using their MIPS **
    
    trigtimearr = []
    
    for filenameDCM in dicomImgArr:
        ds = dicom.read_file(filenameDCM)
        trigtimearr.append(ds.TriggerTime)
    
    unique, counts = numpy.unique(trigtimearr, return_counts=True)
    trigdict= dict(zip(unique, counts))
    
    RefImg = dicom.read_file(dicomImgArr[1])
    ConstPixelDims = (int(RefImg.Rows), int(RefImg.Columns),len(trigdict), len(trigtimearr)/len(trigdict))
    ImgArray =numpy.zeros(ConstPixelDims, dtype=RefImg.pixel_array.dtype)
    
    n=numpy.zeros((len(trigdict)))
    
    for filenameDCM in dicomImgArr:
        ds = dicom.read_file(filenameDCM)
        for i in unique: 
            if(ds.TriggerTime==i):
                ImgArray[:, :,numpy.where(unique==i)[0][0],n[numpy.where(unique==i)[0][0]]] = ds.pixel_array
                n[numpy.where(unique==i)[0][0]]+=1
    maxdiff=0
    maxdiffarg=0
    for i in range(1,len(unique)):
        diff = (numpy.mean(MIP_Return(ImgArray[:,:,i,:])))-(numpy.mean(MIP_Return(ImgArray[:,:,i-1,:])))
        if(diff > maxdiff):
            maxdiff= diff
            maxdiffarg = i
    
    returnimgs = []
    returnimgargs = []
    #for i in range(maxdiffarg,len(unique)):
        #returnimgs.append(ImgArray[:,:,i,:])
        #returnimgs.append(MIP_Return(ImgArray[:,:,i,:]))
        #returnimgargs.append(i) 
    
    returnimgs = ImgArray[:,:,maxdiffarg,:]
    returnimgargs = maxdiffarg
    
    returnimgs=numpy.array(returnimgs)
    
    MIPBeforePeakTT = MIP_Return(ImgArray[:,:,returnimgargs-1,:])
    PicDim = int(ImgArray.shape[0])
    
    #return imgs of peak TT, arg of peak TT in TT seq, and MIP of TT before
    return (returnimgs, returnimgargs, MIPBeforePeakTT, PicDim)
	
def LocateTumor(dicomimgarr):
    ImgarrTTPeak, TTarg, ImgarrMIPBeforeTTPeak,dim  = trigtimetumordetect(dicomimgarr)
    #ImgarrTTPeak are all images in dicomarr in peakTT
    #TTarg is the arg of TT in TT list
    #ImgarrMIPBEFORETTPEAK is mip of (TTPeak -1) images (as comparison)
    #dim is image dim (480 in 480*480 case)
    
    #NEED TO INPUT MORPHOLOGICAL SNAKE ON CHEST
    TTPeakMIPsegarr =segment(MIP_Return(ImgarrTTPeak)[:,:],0,0, 4)
    TTPeakMIPseg = TTPeakMIPsegarr.reshape(math.sqrt(len(TTPeakMIPsegarr)), math.sqrt(len(TTPeakMIPsegarr)))
    
    TTPeakMIPseg2=numpy.copy(TTPeakMIPseg)
    TTPeakMIPseg2[numpy.where(TTPeakMIPseg2==1)]=0
   
    #Getting MIP Segged Chest of PrevPeak
    TTPrevPeakMIPsegarr =segment(ImgarrMIPBeforeTTPeak,0,0, 4)
    TTPrevPeakMIPseg = TTPrevPeakMIPsegarr.reshape(math.sqrt(len(TTPrevPeakMIPsegarr)), math.sqrt(len(TTPrevPeakMIPsegarr)))
    
    TTPrevPeakMIPseg2=numpy.copy(TTPrevPeakMIPseg)
    TTPrevPeakMIPseg2[numpy.where(TTPrevPeakMIPseg2==1)]=0
    
    maxpixdiff=0
    imax=0
    jmax=0
    
    for i in range(dim-20):
        for j in range(dim-20):
            sum1=sum(TTPeakMIPseg2[i:i+20, j:j+20].ravel())
            sum2=sum(TTPrevPeakMIPseg2[i:i+20, j:j+20].ravel())
            
            if(all(l!=0 for l in (TTPrevPeakMIPseg2[i:i+20, j:j+20].ravel())) and all(l!=0 for l in (TTPeakMIPseg2[i:i+20, j:j+20].ravel()))):
                a= abs(sum1-sum2)
                if(a>maxpixdiff):
                    maxpixdiff=a
                    imax=i+10
                    jmax=j+10
            
    return imax,jmax
	
def findImgsWithMass(dicomimgarr): #174,329
    coords = LocateTumor(dicomimgarr)
    x=coords[0]
    y=coords[1]
    
    TTimages, TTimagearg, ImgarrMIPBeforeTTPeak, dim = trigtimetumordetect(dicomimgarr)
    TTmip=MIP_Return(TTimages)
    
    massimagearr=[]
    
    mean1= numpy.mean(TTmip[(x-10):(x+10), (y-10):(y+10)].ravel()) ##TTMip sum
    #sum2= sum(ImgarrMIPBeforeTTPeak[(x-10):(x+10), (y-10):(y+10)].ravel()) ##prev mip
    
    for i in range(int(TTimages.shape[2])):
        mean3= numpy.mean((TTimages[(x-10):(x+10), (y-10):(y+10), i]).ravel()) #sum of current image
        meanelsehwere = numpy.mean((TTimages[(x-40):(x+40), (y-40):(y+40), i]).ravel())
        
        #if(abs(sum1-sum3)<abs(sum2-sum3)):
        if(mean3>meanelsehwere):
            massimagearr.append(TTimages[:,:,i])
    
    massimagearr2=numpy.array(massimagearr)
    return massimagearr2
	
def displayTumorImageSnakes(dicomimgarr):
    morph_snake_patient2arr=numpy.array([])
    massloc= LocateTumor(dicomimgarr)
    imgs= findImgsWithMass(dicomimgarr)
    
    for i in range(0,2):#imgs.shape[0]):
        snake_seg = segment(imgs[i],0,0,2)
        snake_seg2D= snake_seg.reshape(math.sqrt(len(snake_seg)), math.sqrt(len(snake_seg)))
        numpy.append(morph_snake_patient2arr,morph_snake(snake_seg2D, massloc, 15))
        #morph_snake(snake_seg2D, massloc, 15)
