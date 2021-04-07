import h5py as h5
import numpy as np
import cv2 as cv
import DL_HSI_Segmentor as HSI_Segment

# ------- autoencoder package --------------------------
import DL_HSI_Segmentor as cae

#FILE_PATH = 'D:/Research/Remote Sensing/Hyperspectral/PRISMA/DataSets_6_month_interval/PRS_L2B_STD_20200401102434_20200401102438_0001.he5'
#FILE_PATH_TEST = 'D:/Research/Remote Sensing/Hyperspectral/PRISMA/DataSets_6_month_interval/PRS_L2B_STD_20200412101431_20200412101435_0001.he5'
FILE_PATH='/home/hobbes/Projects/PRISMA/DataSets_6_month_interval/PRS_L2B_STD_20200401102434_20200401102438_0001.he5'
FILE_PATH_TEST = '/home/hobbes/Projects/PRISMA/DataSets_6_month_interval/PRS_L2B_STD_20200412101431_20200412101435_0001.he5'
#FILE_PATH = 'C:/projects/RemoteSensing/PRISMA DATA/DataSets_6_month_interval/PRS_L2B_STD_20200401102434_20200401102438_0001.he5'
#FILE_PATH_TEST = 'C:/projects/RemoteSensing/PRISMA DATA/DataSets_6_month_interval/PRS_L2B_STD_20200412101431_20200412101435_0001.he5'



def OpenDataSet( file_name ):
    print ( file_name )
    data = h5.File(file_name, 'r')
    print (data.items())
    print ('------------ data set is opened  ------------------- ')

    return data


def CreateDataSet( file_name_ ):
    print ( file_name_)
    dataOut = h5.File(file_name_, 'w')
    print ( ' ----------- data set created ---------------------')
    return dataOut


def LoadBlocks( file_name ):
    data_block = h5.File(file_name, 'r')
    EOS = data_block["HDFEOS"]
    EOS_Info = data_block["HDFEOS INFORMATION"]
    INFO = data_block["Info"]
    KDP_AUX = data_block["KDP_AUX"]
    return data_block, EOS, EOS_Info, INFO, KDP_AUX

def LoadDataSets( file_name):
    try:
        dataSet = h5.File(file_name, 'r')
    except:
        print ('Error HDF5.File().  Unable to read the file.  Check path and that file exists')
        return None
    SPECTRAL_SWIR = dataSet["HDFEOS/SWATHS/PRS_L2B_HCO/Data Fields/SWIR_Cube"]
    SPECTRAL_SWIR = np.array(SPECTRAL_SWIR)

    SPECTRAL_SWIR_ERR = dataSet["HDFEOS/SWATHS/PRS_L2B_HCO/Data Fields/SWIR_PIXEL_L2_ERR_MATRIX"]
    SPECTRAL_SWIR_ERR = np.array(SPECTRAL_SWIR_ERR)

    SPECTRAL_VNIR = dataSet["HDFEOS/SWATHS/PRS_L2B_HCO/Data Fields/VNIR_Cube"]
    SPECTRAL_VNIR = np.array(SPECTRAL_VNIR)

    SPECTRAL_VNIR_ERR = dataSet["HDFEOS/SWATHS/PRS_L2B_HCO/Data Fields/VNIR_PIXEL_L2_ERR_MATRIX"]
    SPECTRAL_VNIR_ERR = np.array(SPECTRAL_VNIR_ERR)

    PANCHROMATIC = dataSet["HDFEOS/SWATHS/PRS_L2B_PCO/Data Fields/Cube"]
    PANCHROMATIC = np.array(PANCHROMATIC)

    print ( '-------------- LoadDataSets ---------------------------------')
    dataSet.close()

    return SPECTRAL_SWIR, SPECTRAL_VNIR, PANCHROMATIC




def BinDataSets ( SWIR, binWidthSWIR, VNIR, binWidthVNIR ):
    nBinsSWIR = int(SWIR.shape[1]/binWidthSWIR + 1)
    nBinsVNIR = int(VNIR.shape[1]/binWidthVNIR + 1)

    binnedImageSWIR = np.zeros((SWIR.shape[0], nBinsSWIR, SWIR.shape[2]))
    binnedImageVNIR = np.zeros((VNIR.shape[0], nBinsVNIR, SWIR.shape[2]))
    for bin in range(nBinsSWIR):
        for wavelength in range(binWidthSWIR):
            srcBinIndex = int ( bin*binWidthSWIR+wavelength)
            if srcBinIndex < SWIR.shape[1]:
                binnedImageSWIR[:,bin,:] = binnedImageSWIR[:,bin,:]+(SWIR[:,srcBinIndex,:]/nBinsSWIR)


    for bin in range(int(nBinsVNIR)):
        for wavelength in range(binWidthVNIR):
            srcBinIndex = int ( bin*binWidthVNIR+wavelength)
            if srcBinIndex < VNIR.shape[1]:
                binnedImageVNIR[:,bin,:] = binnedImageVNIR[:,bin,:]+(VNIR[:,bin*binWidthVNIR+wavelength,:]/nBinsVNIR)

    return binnedImageSWIR, binnedImageVNIR


def ReformatSamples ( dataSet ):
    dArray = np.array ( dataSet )
    reformattedMat = np.empty((dataSet.shape[0]*dataSet.shape[2], dataSet.shape[1]), dtype=float)
    for l in range ( dArray.shape[1]):
        for i in range ( dArray.shape[2]):
            col = dArray[:,l,i]
            row_start = i*col.shape[0]
            row_end = i*col.shape[0] + col.shape[0]
            reformattedMat[row_start:row_end,l] = col.transpose()
   # arr = dArray.reshape (dataSet.shape[0]*dataSet.shape[2], dataSet.shape[1] )
    arr = reformattedMat
    return arr


def ComputePrincipalComponents( SWIR_sample, VNIR_sample):
    FeatureVector = np.concatenate((SWIR_sample, VNIR_sample), axis=1)

    mean = np.empty((0))
    mean, EigenVectors, EigenValues = cv.PCACompute2(FeatureVector, mean)
    print ( '---------------- Compute PCA to get basis vectors ------------------------- ')
    return FeatureVector, mean, EigenVectors, EigenValues


def TransformBasis ( SampleData, CovarianceMatrix, NbrPrincipalComponents):
    NewBasisTxform = CovarianceMatrix[:,0:NbrPrincipalComponents]

    imSize = np.sqrt(SampleData.shape[0])

    SampleSetPC = np.dot(SampleData, NewBasisTxform)

    reformattedMat = np.empty((int(imSize), int(NbrPrincipalComponents),int(imSize)), dtype=float)
    nbrSlices = SampleSetPC.shape[0]/imSize
    for l in range ( int(nbrSlices)):
        for i in range ( NbrPrincipalComponents):
            col_start = l*imSize
            col_end = col_start+imSize
            reformattedMat[l,i,:] = SampleSetPC[int(col_start):int(col_end), i].transpose()

#    SampleSetPC_image_set = SampleSetPC.reshape(int(imSize), SampleSetPC.shape[1], -1 )
    SampleSetPC_image_set = reformattedMat
    print ( ' ----------------- Basis Transformation complete -----------------------------')
    return SampleSetPC_image_set




def ClusterFeatures ( eigenBands, numberClusters  ):
    #For each (x,y), generate a feature vector across eigenvectors
    featureVectorList = []
    nFeatures = eigenBands.shape[1]
    imgW = eigenBands.shape[2]
    imgH = eigenBands.shape[0]

    for row in range( nFeatures ):
        X = eigenBands[:,row,:].reshape(imgH,imgW)
        featureVectorList.append(X)

    featureVectorArray = np.array(featureVectorList)
    featureArrayHeight = featureVectorArray.shape[1]*featureVectorArray.shape[2]
    featureArrayWidth = featureVectorArray.shape[0]
    featureArray = np.empty ( (int(featureArrayHeight), int(featureArrayWidth)) )
    for row in range(featureVectorArray.shape[1]):
        for col in range ( featureVectorArray.shape[2]):
            featureVector = featureVectorArray[:,row,col]
            featureVectorRowIdx = row * featureVectorArray.shape[1] + col
            featureArray[featureVectorRowIdx,:] = featureVector
 #   featureArray = featureVectorArray.reshape((featureArrayHeight, featureArrayWidth))
    featureArray= np.float32(featureArray)


    #define number of clusters
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 40, 1.0)
    initNbrAttempts = 100
    flags = cv.KMEANS_RANDOM_CENTERS
    ret, labels, centre = cv.kmeans( featureArray, numberClusters, None, criteria, initNbrAttempts, flags )


    #Now, convert back into the shape of the original image
    centre = np.uint8(centre)
    imgReshape = np.empty((imgH, imgW))

    for l in range ( imgH ):
        idx_start = l * imgW
        idx_end = l*imgW + imgW
        imgReshape[l, : ] = labels[idx_start:idx_end].transpose()

    #labelImage = labels.reshape( (imgH, imgW), order='C')
    labelImage = imgReshape
    print ( '------------------- Compute the clusters of the feature vectors ---------------')
    return labelImage



def TrainBatch ( dataArray, nbrBatches ):

    #read 'batchSize' elements and train on each model
    batch_data_set_list = []

    batch_size = int(dataArray.shape[0]/nbrBatches)

    for i in range (nbrBatches):
        batch_array = []
        for j in range(batch_size):
            idx = i*batch_size + j
            batch_array.append(dataArray[idx])
        local_batch_array = np.array(batch_array)
        batch_data_set_list.append(local_batch_array)

    return np.array(batch_data_set_list)


def SegmentPCA(SWIR, VNIR, NumberPrincipalComponents, numberCategories):
    #Remap the stack of 2-D images into column vectors, with each column corresponding to a frequency band
    #Reformat(stack of images)
    #Compute the principle components, and project the images into the bases represented by the PC
    #ComputePrincipalComponents()
    #TransformBasis()
    #Finally, apply k-means clustering
    SWIR_sample = ReformatSamples(SWIR)
    VNIR_sample = ReformatSamples(VNIR)
    FeatureVector, mean, EigenVectors, EigenValues = ComputePrincipalComponents( SWIR_sample, VNIR_sample)
    ImageSetPC = TransformBasis(FeatureVector, EigenVectors, NumberPrincipalComponents)
    map = ClusterFeatures(ImageSetPC, numberCategories)
    return map

def ReformatHSI_HXY(SWIR, VNIR, SWIR_test, VNIR_test):
    #Reformat data, stacking SWIR and VNIR image sets; also need to reformat so x-y plane is single band
    #z-axis is the collection of frequency bands
    nbrBandsSWIR = SWIR.shape[1]
    spectrumList = []
    for band in range(nbrBandsSWIR):
        slice = np.array(SWIR[:,band,:])
        spectrumList.append(slice)

    nbrBandsVNIR = VNIR.shape[1]
    for band in range (nbrBandsVNIR):
        slice = np.array(VNIR[:,band,:])
        spectrumList.append(slice)
    spectrum = np.array(spectrumList )

    nbrBandsSWIR_test = SWIR_test.shape[1]
    spectrum_test = []
    for band in range(nbrBandsSWIR_test):
        slice = np.array(SWIR_test[:,band,:])
        spectrum_test.append(slice)

    nbrBandsVNIR_test = VNIR_test.shape[1]
    for band in range(nbrBandsVNIR_test):
        slice = np.array(VNIR_test[:,band,:])
        spectrum_test.append(slice)
    spectrum_test = np.array(spectrum_test)

    return spectrum, spectrum_test
# *******************************************************************************

def main(NumberPrincipalComponents, numberCategories, outputFile):
    filename = FILE_PATH
    filename_test = FILE_PATH_TEST

    #Load the data from disk
    #First, the training data
    #Second, the test data
    try:
        SWIR, VNIR, PanChroma = LoadDataSets(filename)
        SWIR_test, VNIR_test, PanChroma_test = LoadDataSets(filename_test)
    except:
        print ( 'Unable to load data files')
        return 1


    #Using PCA and k-means clustering for HSI segmentation
#    map = SegmentPCA(SWIR=SWIR,
#                     VNIR=VNIR,
#                     NumberPrincipalComponents=NumberPrincipalComponents,
#                     numberCategories=numberCategories)
#    cv.imwrite(outputFile, map)


    #--------------- alternative approach:  using autoencoder ---------------------------------
    spectrum, spectrum_test = ReformatHSI_HXY(SWIR=SWIR, VNIR=VNIR, SWIR_test=SWIR_test, VNIR_test=VNIR_test )
    #convert fo floating point and normalise over interval (0,1)
    spectrum = spectrum.astype(np.float32)
    spectrum_test = spectrum.astype(np.float32)
    spectrum = spectrum/65535.0
    spectrum_test=spectrum_test/65535.0



    #Create training samples: blk_size x blk_size x 239 blocks -> (1000/blk_size) x (1000/blk_size) blocks
    block_size = int(10)
    nbr_blocks = int(1000 / block_size)

    samples_trg = []
    for col in range(nbr_blocks):
        for row in range(nbr_blocks):
            sample = spectrum[:,block_size*row:block_size*row+block_size, block_size*col:block_size*col+block_size]
            samples_trg.append(sample)
    samples_trg = np.array(samples_trg)
    samples_trg = np.reshape(samples_trg, (samples_trg.shape[0],samples_trg.shape[1], block_size, block_size, 1))

    samples_test = []
    for col in range(nbr_blocks):
        for row in range(nbr_blocks):
            sample = spectrum_test[:,block_size*row:block_size*row+block_size, block_size*col:block_size*col+block_size]
            samples_test.append(sample)
    samples_test = np.array(samples_test)
    samples_test = np.reshape(samples_test, (samples_test.shape[0],samples_test.shape[1], block_size, block_size, 1))


   # IO_Dims = feature_train.shape
    latent_vec_dims = np.array((25, 1))
    nbrFreqBands = samples_trg.shape[1]
    batch_size = 50
    autoEncode = HSI_Segment.CAE3D( nbInputPatterns=8,
                                    blk_size=block_size,
                                    drop_rate=0.5,
                                    drop_seed = 25,
                                    encoded_dim=latent_vec_dims[0],
                                    nFreqBands = nbrFreqBands,
                                    data_format='channels_last',
                                    active_function='relu',
                                    batch_sz = batch_size )

    loss = autoEncode.train(samples_trg, samples_test, batch_size=batch_size, epochs=20)

    encoded_imgs = autoEncode.getDecodedImage(samples_test)
    print ( 'encoded image dims: ', encoded_imgs.shape )
    #Reorder images into full 1k x 1k image(s)
 #   decoded_imgs = autoEncode.getDecodedImage(encoded_imgs)


main( 20, 30, 'Segmented_EigenBands_20_Clusters_30.png')
