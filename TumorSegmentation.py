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

##To RUN: 

def MIP_Return(IMGSeries):
	#Input: Image Array in format: [Pixel Value, Pixel Value, Image # in Sequence]
	#Output: Singular MIP image
	
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
	#Input: Image as 2D Pixel Array - [Pixel Value, Pixel Value] 
	#Output: Clustered Image
	
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
    seg=numpy.array(clt.labels_)
    
    centers= numpy.array(clt.cluster_centers_[:,1])
    sortedcenters=numpy.array(numpy.sort(centers))
    
    for i in range(0,len(seg)):
        a=centers[seg[i]]
        b=numpy.where(sortedcenters==a)
        seg[i]=b[0]
    
    seg2D= seg.reshape(math.sqrt(len(seg)), math.sqrt(len(seg)))
    
    return seg2D

def avgsurrvalues(img, x1, x2, y1, y2):
    #Input: image as 2D pixel array
	#Output: (N*N)*1 vector for N*N input image that contains the average neighbor pixel values for every pixel in the input image
	
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

def circle_levelset(shape, center, sqradius, scalerow=1.0):
	#Input: image shape, region of interest, and radius
	#Output: binary circle levelset for use in the morph_snake algorithm
	
    grid = numpy.mgrid[list(map(slice, shape))].T - center
    phi = sqradius - numpy.sqrt(numpy.sum((grid.T)**2, 0))
    u = numpy.float_(phi > 0)

    return u

def morph_snake(img, ROI, radius):
	#Input: Image to apply snake to as well as a region of interest and radius for the snake to initialize from and extend to
	#Output: Image of applied snake (including pixel values for mass outline)
	
	
    img1 = img/255.0
    # g(I)
    gI = morphsnakes.gborders(img1, alpha=1000, sigma=11)

    mgac = morphsnakes.MorphACWE(img1, smoothing=3, lambda1=1, lambda2=1)
    mgac.levelset = circle_levelset(img1.shape,ROI , radius)
    
    pylab.figure(figsize=(8, 6), dpi=400)
    return(morphsnakes.evolve_visual(mgac, num_iters=30, background=img1)) #110
    pylab.show()

def trigtimetumordetect(dicomImgArr):
    #Input: dicom image array
    #Output: imgs of Peak Trigger Time, arg of Peak TT in TT seq, array of images of TT before Peak TT, and the image dimensions for each TT array
	
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
    lastimgs = ImgArray[:,:,len(unique)-1,:]
    
    returnimgs = ImgArray[:,:,maxdiffarg,:]
    returnimgargs = maxdiffarg
    
    returnimgs=numpy.array(returnimgs)
    
    returnimgsbeforeTT = ImgArray[:,:,returnimgargs-1,:]
    MIPBeforePeakTT = MIP_Return(ImgArray[:,:,returnimgargs-1,:])
    PicDim = int(ImgArray.shape[0])
    
    return (returnimgs, returnimgargs, returnimgsbeforeTT, PicDim, lastimgs)

def LocateTumor(dicomimgarr):
	#Input: dicom image array
	#Output: x,y coordinates of detected mass

    ImgarrTTPeak, TTarg, ImgarrBeforeTTPeak,dim,lastimgs  = trigtimetumordetect(dicomimgarr)
    #ImgarrTTPeak are all images in dicomarr in peakTT
    #TTarg is the arg of TT in TT list
    #ImgarrMIPBEFORETTPEAK is mip of (TTPeak -1) images (as comparison)
    #dim is image dim (480 in 480*480 case)
    
    
    TTPeakMIPseg =segment(MIP_Return(ImgarrTTPeak)[:,:],0,0, 4)
    
    TTPeakMIPseg2=numpy.copy(TTPeakMIPseg)
    TTPeakMIPseg2[numpy.where(TTPeakMIPseg2==3)]=0

    TTPrevPeakMIPseg =segment(MIP_Return(ImgarrBeforeTTPeak),0,0, 4)
    
    TTPrevPeakMIPseg2=numpy.copy(TTPrevPeakMIPseg)
    TTPrevPeakMIPseg2[numpy.where(TTPrevPeakMIPseg2==3)]=0
    
    maxpixdiff=0
    imax=0
    jmax=0
    
    for i in range(dim-20):
        for j in range(dim-20):
            sum1=sum(TTPeakMIPseg2[i:i+20, j:j+20].ravel())
            sum2=sum(TTPrevPeakMIPseg2[i:i+20, j:j+20].ravel())
            
            if(all(l!=0 for l in (TTPrevPeakMIPseg2[i:i+20, j:j+20].ravel())) and all(l!=0 for l in (TTPeakMIPseg2[i:i+20, j:j+20].ravel()))):
            #if((TTPrevPeakMIPseg2[i:i+20, j:j+20].ravel()==0).sum()<30 and (TTPeakMIPseg2[i:i+20, j:j+20].ravel()==0).sum()<30 and (TTPeakMIPseg2[i:i+20, j:j+20].ravel()==3).sum()>50 and (TTPrevPeakMIPseg2[i:i+20, j:j+20].ravel()==3).sum()>50):    
                a= sum1-sum2
                if(a>maxpixdiff):
                    maxpixdiff=a
                    imax=i+10
                    jmax=j+10
            
    return imax,jmax
#need to return tumor center + radius to input into morph snake function **Keep chest center + rad same??

def MorphSnakeChest(img):
    morph_snake(img, (220,240), 130) 

def findImgsWithMass(dicomimgarr):
	#Input: dicom image array
	#Output: images in inputted dicom array where masses were detected

    coords = LocateTumor(dicomimgarr)
    x=coords[0]
    y=coords[1]
    
    TTimages, TTimagearg, ImgarrMIPBeforeTTPeak, dim, lastimgs = trigtimetumordetect(dicomimgarr)
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

def dice(im1,im2):
	#Input: 2 images (binary)
	#Output: Dice coefficient between images
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
        
    im_sum = im1.sum() + im2.sum()

    if im_sum == 0:
        return 0

    intersection = numpy.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum

def mapBackToOrigSeq(dicomimgarr, masslist):
	#Inputs: dicom image array and arguments of detected masses 
	#The function maps the arguments of the images containing masses in the respective peak trigger time array to their arguements in the original dicom array

    a=trigtimetumordetect(dicomimgarr)
    origargs=[]
    
    RefImg = dicom.read_file(dicomimgarr[1])
    ConstPixelDims = (int(RefImg.Rows), int(RefImg.Columns),len(dicomimgarr))
    ImgArray =numpy.zeros(ConstPixelDims, dtype=RefImg.pixel_array.dtype)

    n1=0
    for filenameDCM in dicomimgarr:
        ds = dicom.read_file(filenameDCM)
        ImgArray[:, :,n1] = ds.pixel_array
        n1+=1
    
    for i in range(ImgArray.shape[2]):
        ravelimgarr=ImgArray[:,:,i].ravel()
        #ravela=a[0][:,:,j].ravel()
        for j in masslist:
            if(all(a[0][:,:,j].ravel()==ravelimgarr)):
                origargs.append(i)
    
    return origargs    

def createFolderWithOrigDicomMassArgs(dicomimgarr, masslist):
	#Inputs: dicom image array and list of detected mass args in dicom image array
	#Creates folder with detected mass image dicom files

    a=mapBackToOrigSeq(dicomimgarr,masslist)
    strarr=[]
    for i in a:
        massstr=str(i)
        strarr.append(massstr)
    
    current_directory = "./Documents/BREAST IMAGES used in Model/BreastDx-01-0030"
    final_directory = os.path.join(current_directory, r'P30')
    if not os.path.exists(final_directory):
       os.makedirs(final_directory)
    
    Path = "./Documents/BREAST IMAGES used in Model/BreastDx-01-0030"
    massfolders = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(PathDicom):
        for filename in fileList:
            for i in strarr:
                if ".dcm" in filename.lower() and i in filename.lower():
                    shutil.copy2(Path+'/' + filename, Path+ '/' + 'P30') # target filename is /dst/dir/file.ext
                    
def displayTumorImageSnakestestlist(dicomimgarr, listgiven, loc, inputrad):
    morph_snake_patientarr=[]
    a=trigtimetumordetect(dicomimgarr)
    
    if loc==0:
        massloc= LocateTumor(dicomimgarr)
    else:
        massloc=loc
    
    arr = numpy.zeros((a[3], a[3], len(listgiven)))
    j=0
    
    if inputrad==0:
        rad=15
    else: 
        rad=inputrad
    
    for i in listgiven:
        snake_seg = segment(a[0][:,:,i],0,0,2)
        ms= morph_snake(snake_seg, massloc, rad)
        #ms2 = numpy.copy(ms)
        morph_snake_patientarr.append(ms)
        
    return morph_snake_patientarr
