# One Minute Human

## About

This repository contains the code used for the final project of my MA in Computational Arts at Goldsmiths, 2019. 

The attempts to demonstrate the possilibities that machine learning offer when working with very small dataset. 
It is acknowledged how realistic results can be when working with huge dataset, curated for months by teams of hundreds. In voice synthesis, close to perfect results can be obtained by recording hours of speech with voice actors--who come from the same place as the target--in anechoid rooms. In artistic applications, this is only possible in project that are widely backed by the private sector, such as [Bill Posters and Daniel Howe](http://billposters.ch/spectre-launch/) deep fakes. However, how to work with small dataset is the field in which both artists using machine learning and ML researchers are trying to progress. The self-imposed contraint was to work with about a minute of audiovisual data only. The resulting models, however limited, demonstrate the ease to produce soon-to-be convincing results. The wider idea is to update this project yearly, as new implementations of similar technologies are made public by researchers. 

The voice synthesis part of the project is part of a wider research project by Anna Engelhart, from Goldsmiths' MA Research/Forensic Architecture, in which she investigates the spatial conditions of the Crimean annexation of 2014. 

This folder contains forks of three repositories that I have worked with. The changes I made are specified below:


1.  The tensorflow implementation of [DC-TTS](https://github.com/Kyubyong/dc_tts) by Kyubyong Park. 
    It consists of two networks (deep convolutional networks): _text2mel_, that learns how to translate text data into mel spectrograms (MFCC), and _SSRN_ (for Super-resolution Network), that converts mel spectrograms to full STFT (Short-Time Fourier Transform) spectrogram. The few modifications I have made to Park's original code relate to how to perform transfer learning: training a sucessful model with the standard, massive LJ dataset, then continue training using a different, small dataset (Vladimir Putin speaking in english). This requires a lot of back-and-forth, as the successful model very quickly "unlearns" how to "speak english". Tweaking parameters of the neural net (amount of hidden units, ) mel spectrograms, the batch size, curating the parallel (text == wav file) dataset, and testing every epoch/checkpoint are the critical operations. "Cleaning" the training audio files (noise reduction, silence removal...) has been a key factor. I have tested Park's other implementation of Tacotron2 and though it seems more sophisticated, it is harder to work with small dataset and I could only obtain decent results with dc-tts. 

    The models have been removed from this folder, so as to keep it lightweight and practical. I left the alignment graphs as a "witness" of the training that took place, in _dc-tts/logdir/LJPutin1.3-1_. The relating dataset was also removed from _dc-tts/data/private/voice/LJPutin-1.3_. Various produced samples are available in _dc-tts/samples_. The source texts used to generate content are visible in _dc-tts/source-texts_

    You can find a really good model trained on the LJ dataset (one of the 2-3 standard english dataset in computational linguistics) [here](https://www.dropbox.com/s/1oyipstjxh2n5wo/LJ_logdir.tar?dl=0), place it in dc-tts/logdir, and update the hyperparameters. You can also download the [LJ dataset](https://keithito.com/LJ-Speech-Dataset/) and train your own model from it. Or find a model pretrained on the LJ dataset [here](https://www.dropbox.com/s/1oyipstjxh2n5wo/LJ_logdir.tar?dl=0).

2.  The tensorflow implementation of [pix2pix](https://github.com/affinelayer/pix2pix-tensorflow) by Christopher Hesse.

    I have used this repository 'as is' without any significant modification except for environments/local path related variables. 
The models have been removed from this folder, so as to keep it lightweight and practical. The main effort were put in producing quality dataset by preprocessing videos to provide varied data (scale, position) with neutral background. Some model give better results than others, mostly depending on the amount of variation provided in the training samples. 


3.  The [face2face](https://github.com/datitran/face2face-demo) demo by Dat Tran, for reducing & freezing pix2pix models, and running them with dlib's shape predictor.
    
    The main modifications I made are on the "run.py" file, that runs dlib's shape predictor on a video source and passes it through reduced-and-frozen pix2pix models trained with Hesse's code. I tailored it to the need of my physical setup (monitors, webcam), removed some of the flags and functions and added a function to slide through different models. I have used the other scripts written by Tran (preprocess data, reduce and freeze models) as provided. The process is made abundantly clear on his repository and I was able to learn a lot from his code. 

    Each of these forks contain updated README files that I wrote according to the particular structure of this multi-headed repository, with step-by-step process. The pix2pix folder is untouched, as it is used remotely from the face2face instructions directly. 


## Notes

The models have been removed from the folders, so as to keep it lightweight and practical. 
- You'll need to download dlib's facial landmark model from [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2), unzip it, and place it in face2face/dlib/
- You'll also need a--reduced, frozen--pix2pix model to run the webcam application. You can train your own by following the process in face2face/README.md
- You'll need to train a text-to-speech model, or download a model pretrained on the LJ dataset [here](https://www.dropbox.com/s/1oyipstjxh2n5wo/LJ_logdir.tar?dl=0), place it in dc-tts/logdir, and update the hyperparameters.


## Requirements

  * CUDA enabled NVIDIA GPU. Tested on Linux Mint 19.1 with RTX 2070 max-Q design. 
  * Anaconda

### DC-TTS*

  * python == 2.7.16
  * tensorflow >= 1.3 (will not work on 1.4) 
  * numpy >= 1.11.1
  * librosa
  * tqdm
  * matplotlib
  * scipy

### pix2pix

  * python >= 3.5.0
  * tensorflow-gpu == 1.4.1
  * numpy == 1.12.0 
  * protobuf == 3.2.0 
  * scipy

### face2face

  * python >= 3.5.0
  * tensorflow >= 1.2
  * OpenCV 3.0 (tested with 4.0)
  * Dlib 19.4


## Acknowledgments

  * Thanks to [Kyubyong Park](https://github.com/Kyubyong) for his various text-to-speech implementations (tacotron, deepvoice3, dc-tts...) and his work on foreign datasets. His DC-TTS implementation is based on this paper: [Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention](https://arxiv.org/abs/1710.08969).
  * Thanks to [Christopher Hesse](https://github.com/christopherhesse) for the pix2pix TensorFlow implementation, based on 
@article{pix2pix2016,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  journal={arxiv},
  year={2016}
}

  * Thanks to [Dat Tran](https://github.com/datitran) for the incredibly explainable process and code, that made my debut with Python easier.  
  * Thanks to [Gene Kogan](http://genekogan.com/) and [Memo Akten](http://www.memo.tv/works/#selected-works) for all the learning ressources on machine learning they make available. 
  * Thanks to my classmates Raphael Theolade, Ankita Anand, Eirini Kalaitzidi, Harry Morley, Luke Dash, and George Simms for their help and support. Special thanks to Keita Ikeda for sharing his coding wisdom on every occasion. 
  * Thanks to Dr. Theo Papatheodorou, Dr. Helen Pritchard, Dr. Atau Tanaka, [Dr. Rebecca Fiebrink](https://www.kadenze.com/courses/machine-learning-for-musicians-and-artists/info), and Lior Ben Gai for their inspirational lectures. 
  * Thanks to Anna Clow, Konstantin Leonenko and Pete Mackenzie for the technical support. 
The models have been removed from this folder, so as to keep it lightweight and practical. The main effort were put in producing quality dataset by preprocessing videos to provide varied data (scale, position) with neutral background. Some model give better results than others, mostly depending on the amount of variation provided in the training samples. 


3.  The [face2face](https://github.com/datitran/face2face-demo) demo by Dat Tran, for reducing & freezing pix2pix models, and running them with dlib's shape predictor.
    
    The main modifications I made are on the "run.py" file, that runs dlib's shape predictor on a video source and passes it through reduced-and-frozen pix2pix models trained with Hesse's code. I tailored it to the need of my physical setup (monitors, webcam), removed some of the flags and functions and added a function to slide through different models. I have used the other scripts written by Tran (preprocess data, reduce and freeze models) as provided. The process is made abundantly clear on his repository and I was able to learn a lot from his code. 

    Each of these forks contain updated README files that I wrote according to the particular structure of this multi-headed repository, with step-by-step process. The pix2pix folder is untouched, as it is used remotely from the face2face instructions directly. 


## Notes

The models have been removed from the folders, so as to keep it lightweight and practical. 
- You'll need to download dlib's facial landmark model from [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2), unzip it, and place it in face2face/dlib/
- You'll also need a--reduced, frozen--pix2pix model to run the webcam application. You can train your own by following the process in face2face/README.md
- You'll need to train a text-to-speech model, or download a model pretrained on the LJ dataset [here](https://www.dropbox.com/s/1oyipstjxh2n5wo/LJ_logdir.tar?dl=0), place it in dc-tts/logdir, and update the hyperparameters.


## Requirements

  * CUDA enabled NVIDIA GPU. Tested on Linux Mint 19.1 with RTX 2070 max-Q design. 
  * Anaconda

# DC-TTS*

  * python == 2.7.16
  * tensorflow >= 1.3 (will not work on 1.4) 
  * numpy >= 1.11.1
  * librosa
  * tqdm
  * matplotlib
  * scipy

# pix2pix

  * python >= 3.5.0
  * tensorflow-gpu == 1.4.1
  * numpy == 1.12.0 
  * protobuf == 3.2.0 
  * scipy

# face2face

  * python >= 3.5.0
  * tensorflow >= 1.2
  * OpenCV 3.0 (tested with 4.0)
  * Dlib 19.4


## Acknowledgments

  * Thanks to [Kyubyong Park](https://github.com/Kyubyong) for his various text-to-speech implementations (tacotron, deepvoice3, dc-tts...) and his work on foreign datasets. His DC-TTS implementation is based on this paper: [Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention](https://arxiv.org/abs/1710.08969).
  * Thanks to [Christopher Hesse](https://github.com/christopherhesse) for the pix2pix TensorFlow implementation, based on 
@article{pix2pix2016,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  journal={arxiv},
  year={2016}
}

  * Thanks to [Dat Tran](https://github.com/datitran) for the incredibly explainable process and code, that made my debut with Python easier.  
  * Thanks to [Gene Kogan](http://genekogan.com/) and [Memo Akten](http://www.memo.tv/works/#selected-works) for all the learning ressources on machine learning they make available. 
  * Thanks to my classmates Raphael Theolade, Ankita Anand, Eirini Kalaitzidi, Harry Morley, Luke Dash, and George Simms for their help and support. Special thanks to Keita Ikeda for sharing his coding wisdom on every occasion. 
  * Thanks to Dr. Theo Papatheodorou, Dr. Helen Pritchard, Dr. Atau Tanaka, [Dr. Rebecca Fiebrink](https://www.kadenze.com/courses/machine-learning-for-musicians-and-artists/info), and Lior Ben Gai for their inspirational lectures. 
  * Thanks to Anna Clow, Konstantin Leonenko and Pete Mackenzie for the technical support. 
