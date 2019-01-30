---
layout: post
date: 2019-01-30 11:33:29
tags: [python, "signal processing"]
title: "Fundamental Signal Processing Tools applied to IMU Data"
summary: >
  I applied some fundamental tools of signal processing to recorded
  data from the intertial measurement unit (IMU) of my smartphone.
row_span: 2
thumbnail: signal_processing_output_31_0.png
---


### Libraries


```python
%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
```

### Datei-Import


```python
#data = np.loadtxt("data/Pickup.txt", comments="%", usecols = (0,1,2,3,4,5,6,7,8,9))
data = np.loadtxt("data/pickup2_till.csv", delimiter=";")
# subtract start delay from time stamps
data[:,0] = data[:,0]-data[0,0]
#convert milliseconds into seconds:
data[:,0]=data[:,0]*0.001

# basic properties of the data series
sample_interval = 0.02
sample_freq = 1.0/sample_interval
sample_num = np.shape(data[:,0])[0]
sample_time = sample_num*sample_interval
```

# Die Signale
### Beschleunigungssensor & Gyroskop


```python
plt.figure(figsize=(16, 16))
plt.subplot(2,1,1)
plt.title('x-, y-, z-Achse - Gyroskop')
plt.xlabel('Zeit (Sekunden)')
plt.ylabel('Winkelbeschleunigung (m/s^2)')
plt.plot(data[:,0], data[:,4:-8])
plt.xlim(data[0,0],data[-1,0])

plt.subplot(2,1,2)
plt.title('x-, y-, z-Achse - Beschleunigungssensor')
plt.xlabel('Zeit (Sekunden)')
plt.ylabel('Linearbeschleunigung (m/s^2)')
plt.plot(data[:,0],data[:,1:-11])
plt.xlim(data[0,0],data[-1,0])
None
```


![png](img/signal_processing_output_5_0.png)


# Signalstatistik
## Univariat


```python
# calculate mean and create vector for plotting
mean = np.mean(data[:,1])
# median 
median = np.median(data[:,1])
# standard deviation
std = np.std(data[:,1])

# plot the signal
plt.figure(figsize=(16, 10))
plt.title('y-Achse Beschleunigungssensor')
plt.xlabel('Zeit (Sekunden)')
plt.ylabel('Linearbeschleunigung (m/s^2)')
plt.plot(data[:,0],data[:,1], label='x-Achse', color='#555555')
plt.xlim(data[0,0],data[-1,0])

# mean line
plt.axhline(y=mean, color='g', ls='dashed')
plt.annotate('Mittelwert ({0:.3f})'.format(mean), xy=(2, 1), xytext=(0.25, 2.5), fontsize=14, color='g')

# standard deviation line
plt.axhline(y=mean+std, color='#aaaaaa', ls='dashed')
plt.axhline(y=mean-std, color='#aaaaaa', ls='dashed')

# median line
plt.axhline(y=median, color='r', ls='dashed')
plt.annotate('Median ({0:.3f})'.format(median), xy=(2, 1), xytext=(0.25, 0.5), fontsize=14, color='r')

None
```


![png](img/signal_processing_output_7_0.png)


### Normiertes (kumuliertes) Histogramm


```python
# calculate the normalized counts in each bin
hist_cnt, hist_idx = np.histogram(data[:,1], bins=200)
hist_cnt = hist_cnt/np.sum(hist_cnt)
hist_max = np.max(hist_cnt)

prev_cnt = hist_cnt[0]
hist_cum = np.zeros(len(hist_cnt))
for i_cnt in range(len(hist_cnt)):
    prev_cnt = hist_cum[i_cnt] = hist_cnt[i_cnt]+prev_cnt
hist_cum_max = np.max(hist_cum)


plt.figure(figsize=(20,15))

# plot the histogram
plt.subplot(2,1,1)
plt.title('Normiertes Histogramm')
plt.xlabel('Beschleunigung (m/s^2)')
plt.ylabel('relative Häufigkeit')
plt.xlim(hist_idx[0],hist_idx[-2])
markerline, stemlines, baseline = plt.stem(hist_idx[0:-1], hist_cnt)
plt.setp(markerline, markerfacecolor='#555555')
plt.setp(stemlines, color='#555555')
# mean line
plt.axvline(x=mean, color='g', ls='dashed')
plt.annotate('Mittelwert ({0:.3f})'.format(mean), xy=(0, 0), xytext=(mean+0.2, hist_max*0.97), 
             fontsize=14, color='g', rotation=90)
# standard deviation line
plt.axvline(x=mean+std, color='#aaaaaa', ls='dashed')
plt.axvline(x=mean-std, color='#aaaaaa', ls='dashed')
# median line
plt.axvline(x=median, color='r', ls='dashed')
plt.annotate('Median ({0:.3f})'.format(median), xy=(0, 0), xytext=(median+0.2, hist_max*0.97), 
             fontsize=14, color='r', rotation=90)


# plot the cumulative histogram
plt.subplot(2,1,2)
plt.title('Kumulatives Histogramm')
plt.xlabel('Beschleunigung (m/s^2)')
plt.ylabel('kumulierte relative Häufigkeit')
plt.xlim(hist_idx[0],hist_idx[-2])
markerline, stemlines, baseline = plt.stem(hist_idx[0:-1], hist_cum)
plt.setp(markerline, markerfacecolor='#555555')
plt.setp(stemlines, color='#555555')
# mean line
plt.axvline(x=mean, color='g', ls='dashed')
plt.annotate('Mittelwert ({0:.3f})'.format(mean), xy=(0, 0), xytext=(mean-0.5, hist_cum_max*1.15), 
             fontsize=14, color='g', rotation=90)
# standard deviation line
plt.axvline(x=mean+std, color='#aaaaaa', ls='dashed')
plt.axvline(x=mean-std, color='#aaaaaa', ls='dashed')
# median line
plt.axvline(x=median, color='r', ls='dashed')
plt.annotate('Median ({0:.3f})'.format(median), xy=(0, 0), xytext=(median-0.5, hist_cum_max*1.15), 
             fontsize=14, color='r', rotation=90)
None
```


![png](img/signal_processing_output_9_0.png)


### Gewöhnliche, zentrierte & normierte Momente


```python
num_raw = 5
num_central = 5
m_raw = np.zeros(num_raw)
m_central = np.zeros(num_central)
m_norm = np.zeros(num_central)
inv_len = np.divide(1.0,np.shape(data[:,1])[0])

for k in range(0, num_raw):
    m_raw[k] = np.multiply(np.sum(np.power(data[:,1], k)), inv_len)
    print("{0}. gewöhnliches Moment: {1:8.3f} (m/s^2)^{0}".format(k, m_raw[k]))

print("\n")
for k in range(0, num_central):
    m_central[k] = np.multiply(np.sum(np.power(np.subtract(data[:,1], m_raw[1]), k)), inv_len)
    print("{0}. zentriertes Moment: {1:8.3f} (m/s^2)^{0}".format(k, m_central[k]))
    
print("\n")
for k in range(0, num_central):
    m_norm[k] = np.divide(m_central[k], np.sqrt(np.power(m_central[2], k)))
    print("{0}. normiertes Moment: {1:8.3f}".format(k, m_norm[k]))
```

    0. gewöhnliches Moment:    1.000 (m/s^2)^0
    1. gewöhnliches Moment:    1.990 (m/s^2)^1
    2. gewöhnliches Moment:   17.942 (m/s^2)^2
    3. gewöhnliches Moment:  159.736 (m/s^2)^3
    4. gewöhnliches Moment: 1646.641 (m/s^2)^4
    
    
    0. zentriertes Moment:    1.000 (m/s^2)^0
    1. zentriertes Moment:   -0.000 (m/s^2)^1
    2. zentriertes Moment:   13.982 (m/s^2)^2
    3. zentriertes Moment:   68.383 (m/s^2)^3
    4. zentriertes Moment:  754.406 (m/s^2)^4
    
    
    0. normiertes Moment:    1.000
    1. normiertes Moment:   -0.000
    2. normiertes Moment:    1.000
    3. normiertes Moment:    1.308
    4. normiertes Moment:    3.859
    

Die Verteilung ist linksschief und spitz.

### Entropie


```python
hist_cnt, hist_idx = np.histogram(data[:,1], bins=200000)
hist_cnt = hist_cnt/np.sum(hist_cnt)
entropy = -np.sum(np.multiply(hist_cnt, np.log2(hist_cnt+0.000000001)))
print("Entropie: {0:.2f} bit/Zeichen".format(entropy))
```

    Entropie: 8.44 bit/Zeichen
    

### Stationarität & Ergodizität


```python
# calculate the normalized counts in each bin
hist_cnt1, hist_idx1 = np.histogram(data[:np.int32(len(data[:,1])/2),1], bins=200)
hist_cnt2, hist_idx2 = np.histogram(data[np.int32(len(data[:,1])/2)+1:,1], bins=200)
hist_cnt1 = hist_cnt1/np.sum(hist_cnt1)
hist_cnt2 = hist_cnt2/np.sum(hist_cnt2)
hist_max1 = np.max(hist_cnt1)
hist_max2 = np.max(hist_cnt2)
hist_mean1 = np.mean(data[:np.int32(len(data[:,1])/2),1])
hist_mean2 = np.mean(data[np.int32(len(data[:,1])/2)+1:,1])
hist_median1 = np.median(data[:np.int32(len(data[:,1])/2),1])
hist_median2 = np.median(data[np.int32(len(data[:,1])/2)+1:,1])
hist_var1 = np.var(data[:np.int32(len(data[:,1])/2),1])
hist_var2 = np.var(data[np.int32(len(data[:,1])/2)+1:,1])

print('\n\t \t H1 \t H2\nMittelwerte: \t {0:.3f} \t {1:.3f}\nVarianzen: \t {2:.3f} \t {3:.3f}\n\n'
     .format(hist_mean1, hist_mean2, hist_var1, hist_var2))

# plot the histogram1
plt.figure(figsize=(24,12))
plt.subplot2grid((2,2), (0, 0))
plt.title('Histogram 1. Hälfte')
plt.xlabel('Beschleunigung (m/s^2)')
plt.ylabel('relative Häufigkeit')
plt.xlim(hist_idx1[0],hist_idx1[-2])
markerline1, stemlines1, baseline1 = plt.stem(hist_idx1[0:-1], hist_cnt1)
plt.setp(markerline1, markerfacecolor='#555555')
plt.setp(stemlines1, color='#555555')
# mean line
plt.axvline(x=hist_mean1, color='g', ls='dashed')
plt.annotate('Mittelwert ({0:.3f})'.format(hist_mean1), xy=(0, 0), xytext=(hist_mean1+0.4, hist_max1*0.97), 
             fontsize=14, color='g', rotation=90)
# median lin
plt.axvline(x=hist_median1, color='r', ls='dashed')
plt.annotate('Median ({0:.3f})'.format(hist_median1), xy=(0, 0), xytext=(hist_median1-0.7, hist_max1*0.97), 
             fontsize=14, color='r', rotation=90)

# plot the histogram2
plt.subplot2grid((2,2), (1, 0))
plt.title('Histogram 2. Hälfte')
plt.xlabel('Beschleunigung (m/s^2)')
plt.ylabel('relative Häufigkeit')
plt.xlim(hist_idx2[0],hist_idx2[-2])
markerline2, stemlines2, baseline2 = plt.stem(hist_idx2[0:-1], hist_cnt2)
plt.setp(markerline2, markerfacecolor='#555555')
plt.setp(stemlines2, color='#555555')
# mean line
plt.axvline(x=hist_mean2, color='g', ls='dashed')
plt.annotate('Mittelwert ({0:.3f})'.format(hist_mean2), xy=(0, 0), xytext=(hist_mean2+0.2, hist_max2*0.95), 
             fontsize=14, color='g', rotation=90)
# median lin
plt.axvline(x=hist_median2, color='r', ls='dashed')
plt.annotate('Median ({0:.3f})'.format(hist_median2), xy=(0, 0), xytext=(hist_median2+0.2, hist_max2*0.95), 
             fontsize=14, color='r', rotation=90)

# plot the histogram3 - diff
plt.subplot2grid((2,2), (0, 1), rowspan=2)
plt.title('Differenz')
plt.xlabel('Beschleunigung (m/s^2)')
plt.ylabel('Differenz rel. Häufigkeiten')
plt.xlim(hist_idx2[0],hist_idx2[-2])
markerline2, stemlines2, baseline2 = plt.stem(hist_idx2[0:-1], hist_cnt2-hist_cnt1)
plt.setp(markerline2, markerfacecolor='#555555')
plt.setp(stemlines2, color='#555555')
None
```

    
    	 	 H1 	 H2
    Mittelwerte: 	 0.295 	 3.688
    Varianzen: 	 2.401 	 19.827
    
    
    


![png](img/signal_processing_output_15_1.png)


## Multivariat

### Korrelation


```python
# TODO: Cov, Cor with other accelerometer axis
num_shifts = 32
corr_matrix = np.zeros((num_shifts, num_shifts))
cov_matrix = np.zeros((num_shifts, num_shifts))
cov_input = np.zeros((num_shifts,len(data[:-num_shifts])))
for i in range(num_shifts):
    cov_input[i,:] = data[i:-num_shifts+i,1]
    for j in range(num_shifts):
        corr_matrix[i][j] = np.correlate(data[i:-num_shifts+i,1], data[j:-num_shifts+j,1])
cov_matrix = np.cov(cov_input)

plt.figure(figsize=(24, 3))
plt.subplot(1, 4, 1)
plt.title('Kovarianzmatrix')
plt.xlabel('Verschiebung')
plt.ylabel('Verschiebung')
plt.imshow(cov_matrix, cmap=plt.get_cmap('Greens'), interpolation='none')

plt.colorbar()
plt.subplot(1, 4, 2)
plt.title('Korrelationsmatrix')
plt.xlabel('Verschiebung')
plt.ylabel('Verschiebung')
plt.imshow(corr_matrix, cmap=plt.get_cmap('Greens'), interpolation='none')
plt.colorbar()

plt.figure(figsize=(24, 12))
for s in range(1,9):
    shift = s*s
    plt.subplot(2, 4, s)
    plt.title('d={0}, Cor: {1:.3f}'.format(shift, np.correlate(data[:-shift,1], data[shift:,1])[0]))
    plt.xlabel('')
    plt.plot(data[:-shift,1], data[shift:,1], linestyle='', marker='+', color='g')
    plt.xlabel('Beschleunigung (m/s^2)')
    plt.ylabel('Beschleunigung Verschoben um d={0} (m/s^2)'.format(shift))
    plt.xlim(-15,15)
    plt.ylim(-15,15)
None
```


![png](img/signal_processing_output_18_0.png)



![png](img/signal_processing_output_18_1.png)


### Faltung im Zeitbereich


```python
smoothing_factor = 15
change_conv = np.convolve(data[:,1], (-1, 1), mode='same')
smooth_conv = np.convolve(data[:,1], np.multiply(np.ones(smoothing_factor), 1/smoothing_factor), mode='same')

plt.figure(figsize=(16, 16))

plt.subplot(2,1,1)
plt.title('Faltung mit Glättungs-Kernel (1/{0}, ..., 1/{0})'.format(smoothing_factor))
plt.plot(data[:,0], data[:,1], color='#555555')
plt.plot(data[:,0], smooth_conv[:], color='b')
plt.xlabel('Zeit (Sekunden)')
plt.ylabel('Linearbeschleunigung (m/s^2)')
plt.xlim(data[0,0], data[-1,0])

plt.subplot(2,1,2)
plt.title('Faltung mit Differenzen-Kernel (-1, +1)')
plt.plot(data[:,0], data[:,1], color='#555555')
plt.plot(data[:,0], change_conv[:], color='b')
plt.xlabel('Zeit (Sekunden)')
plt.ylabel('Linearbeschleunigung (m/s^2)')
plt.xlim(data[0,0], data[-1,0])
None
```


![png](img/signal_processing_output_20_0.png)


# Frequenzbereich


```python
# Nuttall Window
a0 = 0.355768
a1 = 0.487396
a2 = 0.144232
a3 = 0.012604

n = np.arange(0,sample_num)
c1 = (2.0*np.pi)/(sample_num-1)

window_nuttall = a0 - a1*np.cos(np.multiply(n,c1)) \
                    + a2*np.cos(np.multiply(n,2.0*c1)) \
                    - a3*np.cos(np.multiply(n,3.0*c1))
```


```python
data_ft = np.fft.fft(data[:,1])
ft_freq = np.fft.fftfreq(n=data[:,1].shape[-1], d=(data[:,0][120]-data[:,0][119]))

# frequency in linear order
ft_lin_freq = np.concatenate([ft_freq[np.int32(sample_num/2.0):], ft_freq[:np.int32(sample_num/2.0)]])
# distance beween samples in frequency domain (cycles per second)
ft_delta_w = ft_freq[1]-ft_freq[0]
```

### Signalstatistik im Frequenzbereich


```python
num_shifts = 32
corr_matrix = np.zeros((num_shifts, num_shifts))
cov_matrix = np.zeros((num_shifts, num_shifts))
cov_input = np.zeros((num_shifts,len(data_ft[:-num_shifts])))
for i in range(num_shifts):
    cov_input[i,:] = np.absolute(data_ft[i:-num_shifts+i])
    for j in range(num_shifts):
        corr_matrix[i][j] = np.correlate(np.absolute(data_ft[i:-num_shifts+i]), 
                                         np.absolute(data_ft[j:-num_shifts+j]))[0]
cov_matrix = np.cov(cov_input)

plt.figure(figsize=(24, 3))

plt.subplot(1, 4, 1)
plt.title('Kovarianzmatrix')
plt.xlabel('Verschiebung')
plt.ylabel('Verschiebung')
plt.imshow(cov_matrix, cmap=plt.get_cmap('Greens'), interpolation='none')
plt.colorbar()

plt.subplot(1, 4, 2)
plt.title('Korrelationsmatrix')
plt.xlabel('Verschiebung')
plt.ylabel('Verschiebung')
plt.imshow(corr_matrix, cmap=plt.get_cmap('Greens'), interpolation='none')
plt.colorbar()

plt.figure(figsize=(24, 12))
for s in range(1,9):
    shift = s*s
    plt.subplot(2, 4, s)
    plt.title('d={0}, Cor: {1:.3f}'
              .format(shift, np.correlate(np.absolute(data_ft[:-shift]), np.absolute(data_ft[shift:]))[0]))
    plt.xlabel('Betragsspektrum')
    plt.ylabel('Betragsspektrum Verschoben um d={0}'.format(shift))
    plt.plot(np.absolute(data_ft[:-shift]), np.absolute(data_ft[shift:]), linestyle='', marker='+', color='g')
    plt.xlim(0,2200)
    plt.ylim(0,2200)
None
```


![png](img/signal_processing_output_25_0.png)



![png](img/signal_processing_output_25_1.png)


### Tiefpassfilterung - Rechteck Filter


```python
low_cut_freq = 20 # Hz
low_cut_omega = low_cut_freq/(2.0*np.pi)
low_idx = np.int32(low_cut_omega/ft_delta_w)

low_rect = np.zeros(sample_num)
low_rect[:low_idx] = 1.0
low_rect[-low_idx:] = 1.0

data_inv = np.fft.ifft(np.multiply(data_ft, low_rect))
low_inv = np.fft.ifft(low_rect)
low_inv = np.concatenate([low_inv[np.int32(sample_num/2.0):], low_inv[:np.int32(sample_num/2.0)]])


plt.figure(figsize=(20,24))

# 1. Zeile
plt.subplot2grid((4,3), (0,0))
plt.title('Original Signal')
plt.xlabel('Zeit (Sekunden)')
plt.ylabel('Beschleunigung (m/s^2)')
plt.plot(data[:,0], data[:,1], color='#555555')
plt.xlim(np.min(data[:,0]), np.max(data[:,0]))

plt.subplot2grid((4,3), (0,1))
plt.title('Realteil der diskreten Fourier Transformierten')
plt.xlabel('Kreisfrequenz (rad/s)')
plt.ylabel('Re(F(w))')
markerline_ft, stemlines_ft, baseline_ft = plt.stem(ft_freq, data_ft.real, markerfmt=' ')
plt.setp(stemlines_ft, color='#555555')
plt.xlim(np.min(ft_freq),np.max(ft_freq))

plt.subplot2grid((4,3), (0,2))
plt.title('Imaginärteil der diskreten Fourier Transformierten')
plt.xlabel('Kreisfrequenz (rad/s)')
plt.ylabel('Im(F(w))')
markerline_ft, stemlines_ft, baseline_ft = plt.stem(ft_freq, data_ft.imag, markerfmt=' ')
plt.setp(stemlines_ft, color='#555555')
plt.xlim(np.min(ft_freq),np.max(ft_freq))

# 2. Zeile
plt.subplot2grid((4,3), (1,0))
plt.title('Realteil der inversen FT des Rechteckfilters')
plt.xlabel('Zeit (s)')
plt.ylabel('g(t)')
markerline_ft, stemlines_ft, baseline_ft = plt.stem(data[:,0], low_inv.real, markerfmt=' ')
plt.setp(stemlines_ft, color='#555555')
plt.xlim(np.min(data[:,0]),np.max(data[:,0]))

plt.subplot2grid((4,3), (1,1))
plt.title('Rechteck Fenster')
plt.xlabel('Kreisfrequenz (rad/s)')
plt.ylabel('G(w)')
plt.plot(ft_freq, low_rect, color='#555555')
plt.xlim(np.min(ft_freq),np.max(ft_freq))
plt.ylim(-0.1, 1.1)

# 3. & 4. Zeile
plt.subplot2grid((4,3), (2,0), colspan=3, rowspan=2)
plt.title('Original und im Frequenzbereich mit Rechteck mutlipliziertes Signal')
plt.xlabel('Zeit (Sekunden)')
plt.ylabel('Beschleunigung (m/s^2)')
plt.plot(data[:,0], data[:,1], color='#999999')
plt.plot(data[:,0], data_inv.real, color='r')
plt.xlim(np.min(data[:,0]), np.max(data[:,0]))
None
```


![png](img/signal_processing_output_27_0.png)


### Tiefpassfilterung - Butterworth Filter


```python
butter_order = 2
butterworth = 1/(1+np.power(np.divide(ft_lin_freq, low_cut_omega), 2.0*butter_order))
low_butter = np.concatenate([butterworth[np.int32(sample_num/2.0):], butterworth[:np.int32(sample_num/2.0)]])

data_inv = np.fft.ifft(np.multiply(data_ft, low_butter))
butter_inv = np.fft.ifft(low_butter)


plt.figure(figsize=(20,24))

# 1. Zeile
plt.subplot2grid((4,3), (0,0))
plt.title('Original Signal')
plt.xlabel('Zeit (Sekunden)')
plt.ylabel('Beschleunigung (m/s^2)')
plt.plot(data[:,0], data[:,1], color='#555555')
plt.xlim(np.min(data[:,0]), np.max(data[:,0]))

plt.subplot2grid((4,3), (0,1))
plt.title('Betragsspektrum der Fouriertrasformierten')
plt.xlabel('Kreisfrequenz (rad/s)')
plt.ylabel('|F(w)|')
markerline_ft, stemlines_ft, baseline_ft = plt.stem(ft_freq, np.absolute(data_ft), markerfmt=' ')
plt.setp(stemlines_ft, color='#555555')
plt.xlim(np.min(ft_freq),np.max(ft_freq))

plt.subplot2grid((4,3), (0,2))
plt.title('Winkelspektrum der Fouriertransformierten')
plt.xlabel('Kreisfrequenz (rad/s)')
plt.ylabel('Winkel(F(w))')
markerline_ft, stemlines_ft, baseline_ft = plt.stem(ft_freq, np.angle(data_ft), markerfmt=' ')
plt.setp(stemlines_ft, color='#555555')
plt.xlim(np.min(ft_freq),np.max(ft_freq))

# 2. Zeile
plt.subplot2grid((4,3), (1,0))
plt.title('Realteil der inversen FT des Butterworth Filter')
plt.xlabel('Zeit (s)')
plt.ylabel('g(t)')
markerline_ft, stemlines_ft, baseline_ft = plt.stem(ft_freq, butter_inv.real, markerfmt=' ')
plt.setp(stemlines_ft, color='#555555')
plt.xlim(np.min(ft_freq),np.max(ft_freq))

plt.subplot2grid((4,3), (1,1))
plt.title('Rechteck-Filter & Butterworth-Filter {0}. Ordnung'.format(butter_order))
plt.xlabel('Kreisfrequenz (rad/s)')
plt.ylabel('G(w)')
plt.plot(ft_freq, low_rect, color='#555555')
plt.plot(ft_freq, low_butter, color='r')
plt.xlim(np.min(ft_freq),np.max(ft_freq))
plt.ylim(-0.1, 1.1)

# 3. & 4. Zeile
plt.subplot2grid((4,3), (2,0), colspan=3, rowspan=2)
plt.title('Original und im Frequenzbereich mit Butterworth mutlipliziertes Signal')
plt.xlabel('Zeit (Sekunden)')
plt.ylabel('Beschleunigung (m/s^2)')
plt.plot(data[:,0], data[:,1], color='#999999')
plt.plot(data[:,0], data_inv.real, color='r')
plt.xlim(np.min(data[:,0]), np.max(data[:,0]))
None
```


![png](img/signal_processing_output_29_0.png)


### Butterworth Filterung im Zeit- und Frequenzbereich


```python
butter_conv = np.convolve(data[:,1], 
                          np.concatenate([butter_inv[np.int32(sample_num/2.0):], 
                                          butter_inv[:np.int32(sample_num/2.0)]]).real, 
                          mode='same')

plt.figure(figsize=(16,8))
plt.title('Butterworth Filterung im Zeit- und Frequenzbereich')
plt.xlabel('Zeit (s)')
plt.ylabel('Beschleunigung (m/s^2)')
plt.plot(data[:,0], data[:,1], color='#cccccc')
plt.plot(data[1:,0], butter_conv[1:], color='b')
plt.plot(data[1:,0], data_inv.real[:-1], color='r')
plt.xlim(data[0,0], data[-1,0])

None
```


![png](img/signal_processing_output_31_0.png)

