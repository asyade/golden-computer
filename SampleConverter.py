import librosa
import librosa.display
import IPython
from skimage import io as skio
from pylab import np, plt
from os import listdir
from os.path import isfile, join


def read_audio(conf, pathname, trim_long_data):
    y, sr = librosa.load(pathname, sr=conf.sampling_rate)
    # trim silence
    if 0 < len(y): # workaround: 0 length causes error
        y, _ = librosa.effects.trim(y) # trim, top_db=default(60)
    # make it unified length to conf.samples
    if len(y) > conf.samples: # long enough
        if trim_long_data:
            y = y[0:0+conf.samples]
    else: # pad blank
        padding = conf.samples - len(y)    # add padding at both ends
        offset = padding // 2
        y = np.pad(y, (offset, conf.samples - len(y) - offset), 'constant')
    return y

def audio_to_melspectrogram(conf, audio):
    spectrogram = librosa.feature.melspectrogram(audio, 
                                                 sr=conf.sampling_rate,
                                                 n_mels=conf.n_mels,
                                                 hop_length=conf.hop_length,
                                                 n_fft=conf.n_fft,
                                                 fmin=conf.fmin,
                                                 fmax=conf.fmax)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram

def show_melspectrogram(conf, mels, title='Log-frequency power spectrogram'):
    librosa.display.specshow(mels, x_axis='time', y_axis='mel', 
                             sr=conf.sampling_rate, hop_length=conf.hop_length,
                            fmin=conf.fmin, fmax=conf.fmax)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()

def read_as_melspectrogram(conf, pathname, trim_long_data, debug_display=False):
    x = read_audio(conf, pathname, trim_long_data)
    mels = audio_to_melspectrogram(conf, x)
    if debug_display:
        IPython.display.display(IPython.display.Audio(x, rate=conf.sampling_rate))
        show_melspectrogram(conf, mels)
    return mels


#x = read_as_melspectrogram(conf, '/home/acorbeau/Documents/Drum Shots/Kicks/Jon Sine - Deep House Essential Kick 01.wav', trim_long_data=False, debug_display=True)

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def convert_sample_to_img(conf, inputFile, outputFile, trim_long_data):
    sample = read_audio(conf, inputFile, trim_long_data)
    mels = audio_to_melspectrogram(conf, sample)
    return mels

    # IPython.display.display(IPython.display.Audio(sample, rate=conf.sampling_rate))
    # show_melspectrogram(conf, mels)


def convert_samples_directory(conf):
    for fileName in listdir(conf.inputPath):
        print(fileName)
        filePath = join(conf.inputPath, fileName)
        if isfile(filePath):
            img = convert_sample_to_img(conf, filePath, conf.outputPath.join(fileName), True)
            
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

            fig = plt.Figure()
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            p = librosa.display.specshow(img, ax=ax, y_axis='log', x_axis='time')
            fig.savefig(join(conf.outputPath, fileName) + '.png')

            # skio.imsave(join(conf.outputPath, fileName) + '.png', img)
            

class conf:
    # Preprocessing settings
    sampling_rate = 44100*2
    duration = 600
    hop_length = 347*duration # to make time steps 128
    fmin = 20
    fmax = sampling_rate // 2
    n_mels = 128
    n_fft = n_mels * 20
    samples = sampling_rate * duration
    inputPath = '/home/acorbeau/Documents/Drum Shots/full/'
    outputPath = '/tmp/kicksimg'

convert_samples_directory(conf)
