import numpy as np
from pylab import *
from scipy import signal
import csv
import pandas as pd

#Arguments
#data: 3-D numpy array of dimensions (subjects x samples x channels)
#sampleRate: Integer
#alphaBuffer: Integer (must be a factor of the sampleRate)
#bandpassRange (optional, default [8, 12]): list of length 2
#compositeScoreBuffer (optional, default 25): Integer

def crossBrainCorrelator(data,sampleRate,alphaBuffer,bandpassRange = [8, 12],compositeScoreBuffer = 25):
		
	if sampleRate % alphaBuffer != 0:
		raise ValueError("alphaBuffer must be a factor of the sampleRate")
		return	
		
	channels = data.shape[-1]
	lookbackVal = int(sampleRate/alphaBuffer)
	subjectData = []
	
	print("Calculating alpha power for all subjects...")
	
	for subject in data:	
	
		subjectAlphaScores = []	
		
		#For each channel in the subject's data, calculate the alpha band power and save it to the subjectAlphaScores list
		for channel in range(0, channels):
		
			alphaArray = []
			bufferCount = lookbackVal
			
			#Divide the data in each channel into buffers. These buffers will be the set of data that we will calculate the alpha band power for
			bufferedData = [[y[channel] for y in subject[x : x + alphaBuffer]] for x in range(0, len(subject), alphaBuffer)]	
			
			for buffer in bufferedData[lookbackVal - 1:]:		#We start at the lookbackVal because each buffer uses some amount of previous data in its calculations 
				buffer = bufferedData[bufferCount - lookbackVal : bufferCount]		#How to iterate through all of the buffers
				flattenedBuffer = [item for sublist in buffer for item in sublist]		#Combine all of the buffers we are using for calculation into one list	
				f_FreqA, Pxx_PowerA = signal.periodogram(np.array(flattenedBuffer).astype(np.float), sampleRate)		#Calculate spectral power at each freq band         
				fStepA = ((sampleRate/2)+1)/len(f_FreqA)		#Calculate the window of f_FreqA matrix for delta freq band
				AlowAF = int(round(bandpassRange[0]/fStepA))   	#Define low then high place in f_FreqA matrix that corresponds to band range in Pxx_PowerA power matrix             
				AhighAF = int(round(bandpassRange[1]/fStepA))		#Note this same process can be used for delta, theta, and beta as well               
				alphaPowerA = np.mean([Pxx_PowerA[AlowAF:AhighAF]])		#Calculate the mean alpha power
				alphaArray = np.append(alphaArray, alphaPowerA)		#Append this mean alpha power to the total alpha array
				bufferCount += 1
	   
			alphaArray = log10(alphaArray) * 10		#Smooth out data so that any excessively large outliers have a minimized effect
			alphaArray = (alphaArray - np.mean(alphaArray)).tolist()	#Mean center the data to create positive and negative values
			subjectAlphaScores.append(alphaArray)

		zippedAlphaScores = np.array(list(zip(*subjectAlphaScores)))	#Reshape the data to get it back into its original shape
		subjectData.append(zippedAlphaScores)

	totalAlphaAverage = np.mean(subjectData,axis=0)[compositeScoreBuffer:,:]	#Average alpha scores across all subjects, starting at same row as correlation scores
	
	scores = []	
	print("Calculating correlation coefficients across all channels...")
	
	for channel in range(0,channels):		#Calculate the correlation coefficients of each channel across all subjects
	
		channelScore = []
		
		for i in range(compositeScoreBuffer,len(subjectData[0])):
		
			#Error handling if the same EEG value occurs a number of times in a row equal to the compositeScoreBuffer (std dev = 0 in corrcoef)
			try:
				#Calculate the average absolute correlation value of all unique subject pairs
				compositeScore = ((sum(abs(np.corrcoef([row[channel] for row in subjectData[0][(i - compositeScoreBuffer):i]],[[row[channel] for row in finalSubject[(i - compositeScoreBuffer):i]] for finalSubject in subjectData[1:]],'same'))) - len(subjectData))/ 2).tolist()
				channelScore.append(compositeScore)
				lastScore = compositeScore
			except:
				#If error occurs, impute the data using the last known score
				channelScore.append(lastScore)
				
		scores.append(channelScore)
		
	return totalAlphaAverage * np.array(list(zip(*scores)))		#Multiply the average alpha scores for each channel by the correlation scores