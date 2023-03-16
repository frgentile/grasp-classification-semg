import numpy as np
import scipy
import matplotlib.pyplot as plt


def iemg_fragment(signal):
    # Calculate lenght of the windows in samples
    N = int(signal.shape[0])
    # Calculate IEMG
    iemg = np.abs(signal).sum()
    # Return
    return iemg

def zero_crossing_fragment(signal, threshold = 0.0):
    # Calculate lenght of the windows in samples
    N = int(signal.shape[0])
    # Prepare storage
    zc = 0
    # Count Zero Crossing
    for k in range(0,N-1):
        if (signal[k] > threshold and signal[k+1] < threshold) or \
           (signal[k] < threshold and signal[k+1] > threshold):
            zc += 1
    # Return
    return zc

def variance_fragment(signal):
    # Calculate lenght of the windows in samples
    N = int(signal.shape[0])
    # Calculate Variance, ddof is used in the denominator of VAR (N - ddof)
    var_ = signal.var(ddof=1.0)
    # Return
    return var_

def slope_sign_changes_fragment(signal):
    # Calculate lenght of the windows in samples
    N = int(signal.shape[0])
    # Prepare storage
    ssc = 0
    # Count Slope Sign Changes
    for k in range(1,N-1):
        if (signal[k] > signal[k+1] and signal[k] > signal[k-1]) or \
           (signal[k] < signal[k+1] and signal[k] < signal[k-1]):
            ssc += 1
    # Return
    return ssc

def waveform_lenght_fragment(signal):
    # Calculate lenght of the windows in samples
    N = int(signal.shape[0])
    # Prepare storage
    wl = 0.0
    # Calculate
    for k in range(0,N-1):
        wl += abs(signal[k+1]-signal[k])
    # Return
    return wl

def willison_amplitude_fragment(signal, threshold):
    # Calculate lenght of the windows in samples
    N = int(signal.shape[0])
    # Prepare storage
    wamp = 0
    # Count Zero Crossing
    for k in range(0, N-1):
        if abs(signal[k+1]-signal[k]) > threshold:
            wamp += 1
    # Return
    return wamp

def kurtosis_fragment(signal):
    # Calculate lenght of the windows in samples
    N = int(signal.shape[0])
    # Calculate
    k_ = scipy.stats.kurtosis(signal)
    # Return
    return k_

def skewness_fragment(signal):
    # Calculate lenght of the windows in samples
    N = int(signal.shape[0])
    # Calculate
    s_ = scipy.stats.skew(signal)
    # Return
    return s_

def rms_fragment(signal):
    # Calculate lenght of the windows in samples
    N = int(signal.shape[0])
    # Calculate
    rms = np.sqrt( (1/N) * np.sum( signal @ signal.T  ) )
    # Return
    return rms

def get_features(signal, ts, tw, overlap=0.0, zc_thr=0.0, wamp_thr=1.0, plot=False):
    '''
    signal     Signal to extract feautures
    ts         Sampling Time
    tw         Sliding Windows Lenght in seconds
    overlap    Overlaping percentage, being 0.0 no overlap, 0.5 50% overlap and 1.0 100% overlap
    plot       Plot
    '''
    
    # Calculate lenght of the windows in samples
    N = int(tw/ts)

    # Overlap
    Noverlap = int(overlap*N)
    
    # Delta Samples
    delta_samples = N-Noverlap
    # print(f"N = {N}, Overlap = {Noverlap}, Delta = {delta_samples}")
    
    # Delta Samples Time
    tds = delta_samples * ts
    
    # Number of repetitions
    Nw = int(((signal.shape[0]-N)/delta_samples)+1)
    # print(f"Nw = {Nw}")
    
    # Prepare storage
    number_of_columns = 12
    # Each column is a feature
    features = np.zeros((Nw, number_of_columns), dtype='float')

    # Index where starts
    idx1 = 0
    # Index where ends
    idx2 = N
    # Aux
    i = 0
    while idx2 <= int(signal.shape[0]):
        # Get features
        features[i][ 0] = iemg_fragment(signal[idx1:idx2])
        features[i][ 1] = zero_crossing_fragment(signal[idx1:idx2], threshold = zc_thr)
        features[i][ 2] = variance_fragment(signal[idx1:idx2])
        features[i][ 3] = slope_sign_changes_fragment(signal[idx1:idx2])
        features[i][ 4] = waveform_lenght_fragment(signal[idx1:idx2])
        features[i][ 5] = willison_amplitude_fragment(signal[idx1:idx2], threshold = wamp_thr)
        features[i][ 6] = kurtosis_fragment(signal[idx1:idx2])
        features[i][ 7] = skewness_fragment(signal[idx1:idx2])
        features[i][ 8] = rms_fragment(signal[idx1:idx2])
        features[i][ 9] = None
        features[i][10] = None
        features[i][11] = None
        #
        # Next loop
        i += 1
        idx1 += delta_samples
        idx2 = idx1 + N
        
    # Prepare time vector
    t = np.linspace(0.0, Nw*tds, Nw, endpoint=False)

    if plot:
        # fig = plt.figure(figsize=(15, 3))
        fig, axs =  plt.subplots(3, 3, figsize=(15, 10))
        # IEMG
        axs[0][0].plot(t, features[:, 0])
        axs[0][0].set_xlabel('Time [seg]')
        axs[0][0].set_ylabel('Amplitude [V]')
        axs[0][0].set_title('IEMG')
        axs[0][0].set_xlim(0, t[-1])
        axs[0][0].grid(True)        
        # Zero Crossing
        axs[0][1].plot(t, features[:, 1])
        axs[0][1].set_xlabel('Time [seg]')
        axs[0][1].set_ylabel('Count')
        axs[0][1].set_title(f'ZC Count, Thr: {zc_thr}')
        axs[0][1].set_xlim(0, t[-1])
        axs[0][1].grid(True)        
        # Variance
        axs[0][2].plot(t, features[:, 2])
        axs[0][2].set_xlabel('Time [seg]')
        axs[0][2].set_ylabel('Amplitude [V]')
        axs[0][2].set_title('Variance')
        axs[0][2].set_xlim(0, t[-1])
        axs[0][2].grid(True)        

        # Slope Sign Changes Count
        axs[1][0].plot(t, features[:, 3])
        axs[1][0].set_xlabel('Time [seg]')
        axs[1][0].set_ylabel('Count')
        axs[1][0].set_title('Slope Sign Changes')
        axs[1][0].set_xlim(0, t[-1])
        axs[1][0].grid(True)        
        # Waveform Lenght
        axs[1][1].plot(t, features[:, 4])
        axs[1][1].set_xlabel('Time [seg]')
        axs[1][1].set_ylabel('Amplitude')
        axs[1][1].set_title('Waveform Lenght')
        axs[1][1].set_xlim(0, t[-1])
        axs[1][1].grid(True)        
        # Willison Amplitude
        axs[1][2].plot(t, features[:, 5])
        axs[1][2].set_xlabel('Time [seg]')
        axs[1][2].set_ylabel('Amplitude')
        axs[1][2].set_title(f'WAMP, Thr: {wamp_thr}')
        axs[1][2].set_xlim(0, t[-1])
        axs[1][2].grid(True)        

        # Kurtosis
        axs[2][0].plot(t, features[:, 6])
        axs[2][0].set_xlabel('Time [seg]')
        axs[2][0].set_ylabel('Amplitude')
        axs[2][0].set_title('Kurtosis')
        axs[2][0].set_xlim(0, t[-1])
        axs[2][0].grid(True)        
        # Skewness
        axs[2][1].plot(t, features[:, 7])
        axs[2][1].set_xlabel('Time [seg]')
        axs[2][1].set_ylabel('Amplitude')
        axs[2][1].set_title('Skewness')
        axs[2][1].set_xlim(0, t[-1])
        axs[2][1].grid(True)        
        # RMS
        axs[2][2].plot(t, features[:, 8])
        axs[2][2].set_xlabel('Time [seg]')
        axs[2][2].set_ylabel('Amplitude [V]')
        axs[2][2].set_title('RMS')
        axs[2][2].set_xlim(0, t[-1])
        axs[2][2].grid(True)
        
        plt.tight_layout()

    # Return Tuple: Features, Time
    return features, t

def get_fft(signal, Ts, plot=False):
    '''
    signal    Signal to obtain FFT
    Ts        Sampling Period in seconds
    plot      Optinal flag to plot FFT

    Doc: https://docs.scipy.org/doc/scipy/tutorial/fft.html
    '''
    # Obtain lenght
    N = len(signal)
    
    # Sample spacing
    x = np.linspace(0.0, N*Ts, N, endpoint=False)
    fft_coefs = scipy.fft.fft(signal)
    freqs = scipy.fft.fftfreq(N, Ts)[:N//2]

    if plot:
        # fig = plt.figure()
        plt.plot(freqs, 2.0/N * np.abs(fft_coefs[0:N//2]))
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude [V]')
        plt.title('FFT')
        plt.grid()
        plt.show()

    # Return Tuple: FFT coeficients, Freq
    return fft_coefs, freqs

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
def design_butter_bandpass(lowcut, highcut, fs, order=5):
    return scipy.signal.butter(order, [lowcut, highcut], fs=fs, btype='band')

def design_butter_highpass(lowcut, fs, order=5):
    return scipy.signal.butter(order, lowcut, fs=fs, btype='highpass')

def design_butter_lowpass(highcut, fs, order=5):
    return scipy.signal.butter(order, highcut, fs=fs, btype='lowpass')


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = scipy.signal.lfilter(b, a, data)
    return y

def butter_highpass_filter(data, lowcut, fs, order=5):
    b, a = butter_highpass(lowcut, fs, order=order)
    y = scipy.signal.lfilter(b, a, data)
    return y

def zero_phase_highpass_filter(data, lowcut, fs, order=5):
    b, a = butter_highpass(lowcut, fs, order=order)
    y = scipy.signal.filtfilt(b, a, data)
    return y


