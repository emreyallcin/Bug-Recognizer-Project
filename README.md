# Bug-Recognizer-Project

This is our directory structure:


Bug Recognizer Project
├── App
│   ├── models
│   │   ├── model.h5
│   │   └── labelencoder.npy
│   ├── static
│   │   ├── images
│   │   ├── neu.png
│   │   ├── neu2.png
│   │   └── style.css
│   ├── templates
│   │   ├── index.html
│   │   └── result.html
│   └── app.py
├── Code
│   └── Recognizer
├── Database
│   ├── audio
│   ├── image
│   └── metadata
└── uploads


Code Execution :

Run app.py in your code editor (I used visual studio code).
Run your browser and paste this on your url section http://127.0.0.1:5000
You can either record an audio by clicking the microphone button (It will listen to your microphone for 6 seconds) or upload a file clicking the upload a file button. It will give you the results.


Purposes of the files :

In Database file you can find the audios and images of insects and a metadata file which contains BugSounds.csv, csv file categorizes the insects and associates the sound files. (You can add insects modifying these files, upload auido file and modify csv file.)

In Code file we have the main model of the system, it is a trained machine learning model.

The uploads file does not contain any files, it is only for users when they upload an audio file, our model uses that file to recognize the bug from auido file and then removes it.

App file is our web application's main file, in models file we have our model saved in h5 format, in static file we have images of bugs again to show users and we have two near east logos for the web app and the main css file, in templates file we have home page and result page of the web app as index.html and result.html, app.py is the main code for running the web app.