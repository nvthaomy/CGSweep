import numpy as np
import scipy as sp
import scipy.stats
import math
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'errorbar.capsize': 8}) #makes endcaps of errorbars visible
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import subprocess as prcs
import mdtraj as md
import mdtraj.reporters
import time
from scipy.ndimage.filters import gaussian_filter1d


def DoBootStrappingOnHistogram(data,hist,NumberBins,RangeMin,RangeMax,NormHistByMax):
	''' Generate sample data sets from histogram-ed data '''
	# The PDF is zero above(below) the highest(lowest) bin of the histogram defined by max(min) 
	#	of the original dataset.
	
	''' BootStrapping Options '''
	NormHistByMax = NormHistByMax
	BasicBoot = False # the better default, see Wikipedia BootStrapping
	PercentileBoot = True
	alpha = 0.05
	NumberBootStraps = 10000
	
	''' ************************************ '''
	''' ************************************ '''
	LenData = len(data) # generate bootstrap data sets with same number of samples
	hist_dist = scipy.stats.rv_histogram(hist) # for continuous data
	GenDataSets = []
	GenHistograms = []
	for i in range(NumberBootStraps): # Generated fictitious data sets from pdf
		temp      = hist_dist.rvs(size=LenData)
		GenDataSets.append(temp)
		tempHist  = np.histogram(temp,NumberBins,range=(RangeMin,RangeMax),density=True)
		if NormHistByMax == True:
			tempHist2 = np.divide(tempHist[0],np.max(tempHist[0])).tolist()
		else:
			tempHist2 = tempHist[0].tolist()
		GenHistograms.append(tempHist2)
	HistBins = tempHist[1] # Get the histogram bins (the same for all bins)
	GenHistogramArray = np.asarray(GenHistograms)
	HistAvg = np.mean(GenHistogramArray,axis=0).flatten()
	HistStdDev = np.std(GenHistogramArray,axis=0).flatten()
	HistStdErr0 = HistStdDev/np.sqrt(NumberBootStraps)
	HistUpperPercentile = np.percentile(GenHistogramArray,(100*(1-alpha/2)),axis=0).flatten()
	HistLowerPercentile = np.percentile(GenHistogramArray,(100*alpha/2),axis=0).flatten()
	HistStdErr1 = scipy.stats.sem(GenHistogramArray,axis=0).flatten()
	if PercentileBoot == True:
		tempPlus = HistUpperPercentile
		tempMinus = HistLowerPercentile
	if BasicBoot == True:
		tempPlus = np.subtract(2*HistAvg,HistLowerPercentile)
		tempMinus = np.subtract(2*HistAvg,HistUpperPercentile)
		
	#CIPlus = np.add(HistAvg,(Zscore95*HistStdErr0))
	#CIMinus = np.subtract(HistAvg,(Zscore95*HistStdErr0))
	
	CIPlus = tempPlus
	CIMinus = tempMinus
	
	#Overall averages and stderr
	GenDataSets = np.asarray(GenDataSets)
	HistAvgValue = np.mean(GenDataSets)
	HistAvgValueStdDev = np.std(GenDataSets)
	
	return HistAvg, CIPlus, CIMinus, alpha, NormHistByMax, HistAvgValue, HistAvgValueStdDev
	

def HistogramRee(data, number_bins=25, DoBootStrapping=True, ShowFigures=True, NormHistByMax=True, 
					TrimRee=False, ReeCutoff=1.5, ReeMinimumHistBin=0., scale = 1, gaussian_filter=True, sigma=2):
	''' Histograms end-to-end distances and performs boot-strapping to get estimates of the error.'''
	# End-to-end distance distribution
	number_Ree_hist_bins = number_bins
	TrimRee = TrimRee
	ReeCutoff = ReeCutoff
	ReeMinimumHistBin = ReeMinimumHistBin # NOT USED HERE, CHANGE IN TRAJPARSE
	Ree_list = data.tolist() # originally setup to take a list
	scale = scale
	gaussian_filter = gaussian_filter
	sigma = sigma
	
	# To remove values from Ree
	if TrimRee == True:
		Ree_temp2 = []
		Rg_temp2 = []
		cnt = 0
		for i,ReeValue in enumerate(Ree_list):
			if ReeValue >= ReeCutoff:
				Ree_temp2.append(ReeValue)
				Rg_temp2.append(Rg_list[i])
			else:
				cnt +=1
		Ree_list = Ree_temp2
		Rg_list  = Rg_temp2
		print ("Ree values removed below a cutoff of {} were: {}".format(ReeCutoff,cnt))
	
	Ree_max = (np.divide(np.asarray(Ree_list),scale)).max() + 0.05*(np.divide(np.asarray(Ree_list),scale)).max()
	Ree_min = (np.divide(np.asarray(Ree_list),scale)).min() - 0.05*(np.divide(np.asarray(Ree_list),scale)).min()
	if Ree_min < 0: Ree_min = 0 # Cannot have negative values
	hist = np.histogram(np.divide(np.asarray(Ree_list),scale), number_Ree_hist_bins, range=(Ree_min,Ree_max),density=True)
	Ree_hist = hist[0]
	Ree_bins = hist[1]
	if DoBootStrapping == True:
		[HistAvg, CIPlus, CIMinus, alpha, NormHistByMax, HistAvgValueRee, HistAvgValueStdDevRee] = DoBootStrappingOnHistogram(np.divide(np.asarray(Ree_list),scale),hist,number_Ree_hist_bins,Ree_min,Ree_max, NormHistByMax)
	plt.hist(np.asarray(Ree_list),bins=number_Ree_hist_bins,density=True, facecolor='blue',alpha=0.2,edgecolor='black', linewidth=1.2)
	plt.xlabel("distance [nm]")
	plt.ylabel("probability density")
	plt.title("Ree distribution")
	plt.savefig('ReeDistribution.png', format='png', dpi=1200)
	if ShowFigures == True:
		plt.show()
	plt.close()
	
	# With CI intervals
	plt.hist(np.asarray(np.divide(Ree_list,scale)),bins=number_Ree_hist_bins,range=(Ree_min,Ree_max),density=True, facecolor='blue',alpha=0.2,edgecolor='black', linewidth=1.2)
	delta_bin = Ree_bins[1]-Ree_bins[0] # calculat the width of bin
	Ree_bin_centers = Ree_bins+(delta_bin/2)
	
	if NormHistByMax == True:
		Ree_hist_temp = np.divide(Ree_hist,Ree_hist.max())
	else:
		Ree_hist_temp = Ree_hist
	if gaussian_filter == True: 
		ysmoothed = gaussian_filter1d(Ree_hist_temp, sigma=sigma)
	else: 
		ysmoothed = Ree_hist_temp
	
	plt.plot(Ree_bin_centers[:-1], ysmoothed,'b')
	#plt.plot(Ree_bin_centers[:-1],Ree_hist_temp,'b')
	plt.fill_between(Ree_bin_centers[:-1],CIMinus,CIPlus,alpha=0.25,facecolor='r')
	plt.xlabel("distance [nm]")
	plt.ylabel("probability density")
	plt.title("Ree Distribution: {}% Confidence Intervals".format((100*(1-alpha))))
	plt.savefig('ReeDistribution_CI.png', format='png', dpi=1200)
	plt.close()
	
	with open("Ree_Distribution.txt", 'w') as f:
		f.write("#	Bin_center  Prob.-density \n")
		for i in zip(Ree_bin_centers[:-1], Ree_hist):
			f.write("{} {} \n".format(i[0], i[1]))
			
	with open("Ree_Distribution_CI.txt", 'w') as f:
		f.write("#	Bin_Center HistAvg CIPlus CIMinus \n")
		for i in zip(Ree_bin_centers[:-1], Ree_hist_temp, CIPlus, CIMinus):
			f.write("{0:10.6f}   {1:10.6f}   {2:10.6f}   {3:10.6f} \n".format(i[0], i[1], i[2], i[3]))