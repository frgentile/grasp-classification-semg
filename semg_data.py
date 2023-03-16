import numpy as np
import scipy
import matplotlib.pyplot as plt

class SEMGData:    
    grasp_labels = {
        # a
        'spher_ch1': 'Spherical Grasp Ch1',
        'spher_ch2': 'Spherical Grasp Ch2',
        # b
        'tip_ch1': 'Tip Grasp Ch1',
        'tip_ch2': 'Tip Grasp Ch2',
        # c
        'palm_ch1': 'Palmar Grasp Ch1',
        'palm_ch2': 'Palmar Grasp Ch2',
        # d
        'lat_ch1': 'Lateral Grasp Ch1',
        'lat_ch2': 'Lateral Grasp Ch2',
        # e
        'cyl_ch1': 'Cylinder Grasp Ch1',
        'cyl_ch2': 'Cylinder Grasp Ch2',
        # f
        'hook_ch1': 'Hook Grasp Ch1',
        'hook_ch2': 'Hook Grasp Ch2',
    }
    
    def __init__(self, mat_file, fs):
        # Save path to mat file
        self.mat_file_ = mat_file
        # Sample Frequency
        self.fs_ = fs
        # Load and generate another members
        self.load_mat(mat_file)
        
    def load_mat(self, mat_file):
        self.raw_mat_data_ = None
        self.raw_mat_data_ = scipy.io.loadmat(mat_file)
        self.cyl_ch1_ = self.raw_mat_data_['cyl_ch1']
        self.cyl_ch2_ = self.raw_mat_data_['cyl_ch2']
        self.hook_ch1_ = self.raw_mat_data_['hook_ch1']
        self.hook_ch2_ = self.raw_mat_data_['hook_ch2']
        self.tip_ch1_ = self.raw_mat_data_['tip_ch1']
        self.tip_ch2_ = self.raw_mat_data_['tip_ch2']
        self.palm_ch1_ = self.raw_mat_data_['palm_ch1']
        self.palm_ch2_ = self.raw_mat_data_['palm_ch2']
        self.spher_ch1_ = self.raw_mat_data_['spher_ch1']
        self.spher_ch2_ = self.raw_mat_data_['spher_ch2']
        self.lat_ch1_ = self.raw_mat_data_['lat_ch1']
        self.lat_ch2_ = self.raw_mat_data_['lat_ch2']

    def get_ts(self):
        return 1/self.fs_
        
    def get_time_vector(self):
        # Get total number of samples
        N = self.cyl_ch1_.shape[1]
        # Sample spacing
        t = np.linspace(0.0, N/self.fs_, N, endpoint=False)
        #
        return t

    def zero_phase_filter_signals(self, b, a):
        '''
        b     Numerator Coeficients of the filter
        a     Denominator Coeficients of the filter
        '''
        self.spher_ch1_ = scipy.signal.filtfilt(b, a, self.raw_mat_data_['spher_ch1'])
        self.spher_ch2_ = scipy.signal.filtfilt(b, a, self.raw_mat_data_['spher_ch2'])
        self.tip_ch1_ = scipy.signal.filtfilt(b, a, self.raw_mat_data_['tip_ch1'])
        self.tip_ch2_ = scipy.signal.filtfilt(b, a, self.raw_mat_data_['tip_ch2'])
        self.palm_ch1_ = scipy.signal.filtfilt(b, a, self.raw_mat_data_['palm_ch1'])
        self.palm_ch2_ = scipy.signal.filtfilt(b, a, self.raw_mat_data_['palm_ch2'])
        self.lat_ch1_ = scipy.signal.filtfilt(b, a, self.raw_mat_data_['lat_ch1'])
        self.lat_ch2_ = scipy.signal.filtfilt(b, a, self.raw_mat_data_['lat_ch2'])        
        self.cyl_ch1_ = scipy.signal.filtfilt(b, a, self.raw_mat_data_['cyl_ch1'])
        self.cyl_ch2_ = scipy.signal.filtfilt(b, a, self.raw_mat_data_['cyl_ch2'])
        self.hook_ch1_ = scipy.signal.filtfilt(b, a, self.raw_mat_data_['hook_ch1'])
        self.hook_ch2_ = scipy.signal.filtfilt(b, a, self.raw_mat_data_['hook_ch2'])
    
    def get_iemg(self, signal_name, experiment_number, tw, plot=False):
        '''
        signal_name       Name of the signal to use
        experiment_number Row in the signal variable
        tw                It is the length of the sliding windows in second units
        '''
        
        # Calculate lenght of the windows in samples
        N = int(tw*self.fs_)
        # print(f"N = {N}")

        # Get signal
        signal = self.raw_mat_data_[signal_name][experiment_number]
        # Number of slices
        Nw = int(signal.shape[0]/N)
        
        # Prepare storage
        iemg = np.zeros(Nw, dtype='float')

        for i in range(0, Nw):
            # Index where starts
            idx1 = i*N
            # Index where ends
            idx2 = ((i+1)*N)
            # print(idx1, idx2)
            
            # Calculate IEMG
            iemg[i] = np.abs(signal[idx1:idx2]).sum()
        
        # Prepare time vector
        t = np.linspace(0.0, Nw*tw, Nw, endpoint=False)
        
        if plot:
            plt.plot(t, iemg)
        
        # Return Tuple: IEMG, Time
        return iemg, t
    
    def get_zero_crossing(self, signal_name, experiment_number, tw, threshold = 0.0, plot = False):
        '''
        signal_name       Name of the signal to use
        experiment_number Row in the signal variable
        tw                It is the length of the sliding windows in second units
        threshold         Threshold to use in the comparisson
        '''
        
        # Calculate lenght of the windows in samples
        N = int(tw*self.fs_)
        # print(f"N = {N}")

        # Get signal
        signal = self.raw_mat_data_[signal_name][experiment_number]
        # Number of slices
        Nw = int(signal.shape[0]/N)
        
        # Prepare storage
        zc = np.zeros(Nw, dtype='int')

        for i in range(0, Nw):
            # Index where starts
            idx1 = i*N
            # Index where ends
            idx2 = ((i+1)*N)
            # print(idx1, idx2)
            
            # Count Zero Crossing
            for k in range(idx1,idx2-1):
                if (signal[k] > threshold and signal[k+1] < threshold) or \
                   (signal[k] < threshold and signal[k+1] > threshold):
                    zc[i] += 1
        
        # Prepare time vector
        t = np.linspace(0.0, Nw*tw, Nw, endpoint=False)
        
        if plot:
            plt.plot(t, zc)
        
        # Return Tuple: ZC, Time
        return zc, t
    
    def get_variance(self, signal_name, experiment_number, tw, plot = False):
        '''
        signal_name       Name of the signal to use
        experiment_number Row in the signal variable
        tw                It is the length of the sliding windows in second units
        '''
        
        # Calculate lenght of the windows in samples
        N = int(tw*self.fs_)
        # print(f"N = {N}")

        # Get signal
        signal = self.raw_mat_data_[signal_name][experiment_number]
        # Number of slices
        Nw = int(signal.shape[0]/N)
        
        # Prepare storage
        var_ = np.zeros(Nw, dtype='float64')

        for i in range(0, Nw):
            # Index where starts
            idx1 = i*N
            # Index where ends
            idx2 = ((i+1)*N)
            # print(idx1, idx2)
            
            # Calculate Variance, ddof is used in the denominator of VAR (N - ddof)
            var_[i] = signal[idx1:idx2].var(ddof=1.0)            
        
        # Prepare time vector
        t = np.linspace(0.0, Nw*tw, Nw, endpoint=False)
        
        if plot:
            plt.plot(t, var_)
        
        # Return Tuple: VAR, Time
        return var_, t

    def get_slope_sign_changes(self, signal_name, experiment_number, tw, plot = False):
        '''
        signal_name       Name of the signal to use
        experiment_number Row in the signal variable
        tw                It is the length of the sliding windows in second units
        '''
        
        # Calculate lenght of the windows in samples
        N = int(tw*self.fs_)
        # print(f"N = {N}")

        # Get signal
        signal = self.raw_mat_data_[signal_name][experiment_number]
        # Number of slices
        Nw = int(signal.shape[0]/N)
        
        # Prepare storage
        ssc = np.zeros(Nw, dtype='int')

        for i in range(0, Nw):
            # Index where starts
            idx1 = i*N
            # Index where ends
            idx2 = ((i+1)*N)
            # print(idx1, idx2)
            
            # Count Slope Sign Changes
            for k in range(idx1+1,idx2-1):
                if (signal[k] > signal[k+1] and signal[k] > signal[k-1]) or \
                   (signal[k] < signal[k+1] and signal[k] < signal[k-1]):
                    ssc[i] += 1
        
        # Prepare time vector
        t = np.linspace(0.0, Nw*tw, Nw, endpoint=False)
        
        if plot:
            plt.plot(t, ssc)
        
        # Return Tuple: SSC, Time
        return ssc, t
        
    def get_waveform_lenght(self, signal_name, experiment_number, tw, plot = False):
        '''
        signal_name       Name of the signal to use
        experiment_number Row in the signal variable
        tw                It is the length of the sliding windows in second units
        '''
        
        # Calculate lenght of the windows in samples
        N = int(tw*self.fs_)
        # print(f"N = {N}")

        # Get signal
        signal = self.raw_mat_data_[signal_name][experiment_number]
        # Number of slices
        Nw = int(signal.shape[0]/N)
        
        # Prepare storage
        wl = np.zeros(Nw, dtype='float64')

        for i in range(0, Nw):
            # Index where starts
            idx1 = i*N
            # Index where ends
            idx2 = ((i+1)*N)
            # print(idx1, idx2)
            
            # Calculate
            for k in range(idx1,idx2-1):
                wl[i] += abs(signal[k+1]-signal[k])
        
        # Prepare time vector
        t = np.linspace(0.0, Nw*tw, Nw, endpoint=False)
        
        if plot:
            plt.plot(t, wl)
        
        # Return Tuple: WL, Time
        return wl, t

    def get_willison_amplitude(self, signal_name, experiment_number, tw, threshold, plot = False):
        '''
        signal_name       Name of the signal to use
        experiment_number Row in the signal variable
        tw                It is the length of the sliding windows in second units
        threshold         Threshold to use in the comparisson
        '''
        
        # Calculate lenght of the windows in samples
        N = int(tw*self.fs_)
        # print(f"N = {N}")

        # Get signal
        signal = self.raw_mat_data_[signal_name][experiment_number]
        # Number of slices
        Nw = int(signal.shape[0]/N)
        
        # Prepare storage
        wamp = np.zeros(Nw, dtype='int')

        for i in range(0, Nw):
            # Index where starts
            idx1 = i*N
            # Index where ends
            idx2 = ((i+1)*N)
            # print(idx1, idx2)
            
            # Count Zero Crossing
            for k in range(idx1,idx2-1):
                if abs(signal[k+1]-signal[k]) > threshold:
                    wamp[i] += 1
        
        # Prepare time vector
        t = np.linspace(0.0, Nw*tw, Nw, endpoint=False)
        
        if plot:
            plt.plot(t, wamp)
        
        # Return Tuple: WAMP, Time
        return wamp, t

    def get_kurtosis(self, signal_name, experiment_number, tw, plot = False):
        '''
        signal_name       Name of the signal to use
        experiment_number Row in the signal variable
        tw                It is the length of the sliding windows in second units
        '''
        
        # Calculate lenght of the windows in samples
        N = int(tw*self.fs_)
        # print(f"N = {N}")

        # Get signal
        signal = self.raw_mat_data_[signal_name][experiment_number]
        # Number of slices
        Nw = int(signal.shape[0]/N)
        
        # Prepare storage
        k_ = np.zeros(Nw, dtype='float64')

        for i in range(0, Nw):
            # Index where starts
            idx1 = i*N
            # Index where ends
            idx2 = ((i+1)*N)
            # print(idx1, idx2)
            
            # Calculate
            k_[i] = scipy.stats.kurtosis(signal[idx1:idx2])
        
        # Prepare time vector
        t = np.linspace(0.0, Nw*tw, Nw, endpoint=False)
        
        if plot:
            plt.plot(t, k_)
        
        # Return Tuple: k, Time
        return k_, t

    def get_skewness(self, signal_name, experiment_number, tw, plot = False):
        '''
        signal_name       Name of the signal to use
        experiment_number Row in the signal variable
        tw                It is the length of the sliding windows in second units
        '''
        
        # Calculate lenght of the windows in samples
        N = int(tw*self.fs_)
        # print(f"N = {N}")

        # Get signal
        signal = self.raw_mat_data_[signal_name][experiment_number]
        # Number of slices
        Nw = int(signal.shape[0]/N)
        
        # Prepare storage
        s_ = np.zeros(Nw, dtype='float64')

        for i in range(0, Nw):
            # Index where starts
            idx1 = i*N
            # Index where ends
            idx2 = ((i+1)*N)
            # print(idx1, idx2)
            
            # Calculate
            s_[i] = scipy.stats.skew(signal[idx1:idx2])
        
        # Prepare time vector
        t = np.linspace(0.0, Nw*tw, Nw, endpoint=False)
        
        if plot:
            plt.plot(t, s_)
        
        # Return Tuple: s, Time
        return s_, t

    def get_rms(self, signal_name, experiment_number, tw, plot = False):
        '''
        signal_name       Name of the signal to use
        experiment_number Row in the signal variable
        tw                It is the length of the sliding windows in second units
        '''
        
        # Calculate lenght of the windows in samples
        N = int(tw*self.fs_)
        # print(f"N = {N}")

        # Get signal
        signal = self.raw_mat_data_[signal_name][experiment_number]
        # Number of slices
        Nw = int(signal.shape[0]/N)
        
        # Prepare storage
        rms = np.zeros(Nw, dtype='float64')

        for i in range(0, Nw):
            # Index where starts
            idx1 = i*N
            # Index where ends
            idx2 = ((i+1)*N)
            # print(idx1, idx2)
            
            # Calculate
            rms[i] = np.sqrt( (1/N) * np.sum( signal[idx1:idx2] @ signal[idx1:idx2].T  ) )
        
        # Prepare time vector
        t = np.linspace(0.0, Nw*tw, Nw, endpoint=False)
        
        if plot:
            fig = plt.figure(figsize=(15, 3))
            plt.plot(t, rms)
            plt.xlabel('Time [seg]')
            plt.ylabel('Amplitude [V]')
            plt.xlim(0, t[-1])
            plt.title(f"{SEMGData.grasp_labels[signal_name]} - RMS")
            plt.grid(True)        
            
        # Return Tuple: RMS, Time
        return rms, t
    
    # # Hilbert
    # signal = male1.cyl_ch1_[0].copy()
    # analytic_signal = scipy.signal.hilbert(signal)
    # amplitude_envelope = np.abs(analytic_signal)