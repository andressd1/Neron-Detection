import scipy.io as spio
import numpy as np
from operator import itemgetter
from scipy.signal import butter,sosfiltfilt, find_peaks
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score

def butter_filter(data, type, cutoff,order, nyq):
    """ Runs a butterworth signal filter over a signal

    Args:
        data (Numpy array): Signal to filter
        type (String): Type of Butterworth filter to apply: "high", "low", "band"
        cutoff (Integer or List): Cuttoff frequency/ies for the filter 
        order (Integer): Order of the filter
        nyq (Float): The nyquist frequency of the signal

    Returns:
        filtered_signal [Numpy array]: The filtered signal
    """    
    normal_cutoff = None
    if type == "band":
        normal_cutoff = [ x / nyq for x in cutoff]
    else:
        normal_cutoff = cutoff / nyq
    # Creates butterworth filter and filters the signal
    s = butter(order, normal_cutoff, btype=type, output = "sos", analog=False)
    filtered_signal = sosfiltfilt(s, data)
    return filtered_signal

def dedup_peaks(train_peaks, train_classes, peaks_found):
    """Checks the peaks found using Scipy to the origianl signal. Removes any invalid peaks
        and duplicates. Returns the indexes, classes and peak windows(samples surrounding peak)

    Args:
        train_peaks (List): List of indexes of peaks in signal
        train_classes (List): Class of peaks in signal
        peaks_found (List): List of peaks detected with Scipy in signal
        raw_signal (Numpy array): Raw signal
        wdw_left (Integer): Sample window left of peak
        wdw_right (Integer): Sample window right of peak

    Returns:
        peak_indexes (Numpy array): Indexes of deduped and confirmed peaks in signal
        peak_class (Numpy array): Class of peaks in peak_indexes
        peak_windows (Numpy array): Window of samples around the peaks in peak_indexes
    """    
    index1 = 0
    index2 = 0
    correct = 0
    peak_indx = []
    peak_class = []

    while index1 < len(train_peaks) :
        if index1 >= len(train_peaks) or index2 >= len(peaks_found):
            break
        if train_peaks[index1] <= peaks_found[index2] and train_peaks[index1] + 50 >= peaks_found[index2]:
            if index1 != len(train_peaks)-1:
                if train_peaks[index1 + 1] <= peaks_found[index2] :
                    index1 += 1

            peak_indx.append(peaks_found[index2])
            peak_class.append(train_classes[index1])
            correct += 1
            index1 += 1
            index2 += 1
        else:  
            if peaks_found[index2] >= train_peaks[index1 + 1] - 50:
                index1 += 1
            elif index2 == len(peaks_found) -1 :
                break
            else:
                index2 += 1   
    return np.asarray(peak_indx), np.asarray(peak_class)

def get_peak_window(peaks, signal, wdw_left, wdw_right):
    """Returns the samples around a peak for each peak found
    e.g. peak = index 450, window = 20 left and 20 right
    peak window = raw_signal[450-20 : 450+20]
    
    Args:
        peaks (List): List of indexes of peaks found
        signal (Numpy Array): The signal to analyse
        wdw_left (Integer): Number of samples left of the peak to retrieve
        wdw_right (Integer): Number of samples right of the peak to retrieve

    Returns:
        Window (List): Window for each peak
    """
    windows = []
    for peak in peaks:
        windows.append(signal[peak-wdw_left:peak+wdw_right])
    return windows

def evaluate(prediction, y_eval): 
    """Calculate F1 score of class prediction

    Args:
        prediction (Numpy array): Predicted class of peaks
        y_eval (Numpy array): Actual class of peaks

    Returns:
        F1 score (Float): F1 score of the predictions
    """    
    return f1_score(y_eval, prediction, average='macro')

# Loading data
mat = spio.loadmat('training.mat', squeeze_me=True)
data_source = mat['d']
spikes_train = np.array(sorted(list(zip(mat['Index'], mat['Class'])), key = itemgetter(0)))

# Filtering signal with butterworth filter
fs = 25000      # sample rate, Hz
cutoff = 2500   # cutoff frequency of the filter, Hz 
nyq = 0.5 * fs  # Nyquist Frequency
order = 2
filter_type = "low"
data_filtered = butter_filter(data_source, type = filter_type, cutoff=cutoff, order = order, nyq=nyq)

# Finding peaks in signal using find_peaks()
# Threshold = 3.7 * np.median(np.absolute(y)/0.6745) 
mad = np.median(np.absolute(data_filtered)/0.6745) 
threshold = 3.7 * mad  # Set a threshold for peak detection
# Finds peaks (change of + to - slope) in raw signal. Only returns the indexes of peaks 
# surpassing height and distance parameters in signal. 
a, b = find_peaks(data_filtered, height=threshold , distance=10)
pk_wdw =[15,35]
peak_indx, peak_cls = dedup_peaks(spikes_train[:,0], spikes_train[:,1], a)
peak_wdw = get_peak_window(peak_indx, data_filtered, pk_wdw[0], pk_wdw[1])

# PCA
# Splits data into 2: training and evaluation
x_train, x_eval, y_train, y_eval = train_test_split(peak_wdw, peak_cls, test_size=0.2)
pcanalisis = PCA(n_components = 0.99)
# Performs principal component analysis on data
xtrain_pca = pcanalisis.fit_transform(x_train)
xeval_pca = pcanalisis.transform(x_eval)

# KNN
# Creates KNN with K = 5
knn = KNeighborsClassifier(n_neighbors=5)
# Fits data and adds labels to data
knn.fit(xtrain_pca, y_train)
# Inserts data into training data space and classifies it as the class 
# of the majority of training samples near (K) it
prediction = knn.predict(xeval_pca)
print(evaluate(prediction, y_eval))
# Confusion matrix of evaluation targets and predictions
print(confusion_matrix(y_eval, prediction))
 