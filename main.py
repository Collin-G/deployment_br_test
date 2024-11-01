from flask import Flask, request, jsonify
import numpy as np
import torch
import tempfile
import os
import librosa
from moviepy.editor import VideoFileClip


import model_holder
from data_processor import main_handler

import ffmpeg
from flask_cors import CORS

# Load your model

engage_model = model_holder.ModelHolder("cnn", "./model_states/EngagingTone_model", map_location=torch.device("cpu"))
calm_model = model_holder.ModelHolder("cnn",  "./model_states/Calm_model", map_location=torch.device("cpu"))
excited_model = model_holder.ModelHolder("cnn", "./model_states/Excited_model", map_location=torch.device("cpu"))
friendly_model = model_holder.ModelHolder("cnn", "./model_states/Friendly_model", map_location=torch.device("cpu"))

eye_model = model_holder.ModelHolder("cnn", "./model_states/Videos_EyeContact_model", map_location=torch.device("cpu"))

star_model = model_holder.ModelHolder("star", "./model_states/star_states", map_location=torch.device("cpu"))
coherency_model = model_holder.ModelHolder("coherency")
linkword_model = model_holder.ModelHolder("linkword")



app = Flask(__name__)
CORS(app) 


def extract_audio_from_video(video_path):
    try:
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        (
            ffmpeg
            .input(video_path)
            .output(temp_audio, acodec='pcm_s16le')
            .overwrite_output()  # This avoids the y/N prompt
            .run()
        )
        return temp_audio
    except Exception as e:
        print(f"Error with ffmpeg: {e}")
        return None


def process_audio(audio_path):
    """Process the audio using librosa to extract features."""
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return torch.tensor(mfcc).unsqueeze(0)  # Shape: (1, 13, Time)

@app.route("/predict", methods=["POST"])
def predict():
    video = request.files.get("file")
    if not video:
        return jsonify({"error": "No video file provided"}), 400

    main_handle = main_handler.MainHandler(video)
    data = main_handle.handle_data()

    engage_output = engage_model.predict(data.get("audio"))
    calm_output = calm_model.predict(data.get("audio"))
    excited_output = excited_model.predict(data.get("audio"))
    friendly_output = friendly_model.predict(data.get("audio"))

    eye_output = eye_model.predict(data.get("video"))

    star_output = star_model.predict(data.get("text"))
    coherency_output = coherency_model.predict(data.get("text"))
    linkword_output = linkword_model.predict(data.get("text"))


    print({"EngagedTone": float(engage_output), "Calmness" : float(calm_output), \
           "Eagerness" : float(excited_output), "Friendliness" : float(friendly_output),\
            "EyeContact": float(eye_output)})
    
    return jsonify({"EngagedTone": float(engage_output), "Calmness" : float(calm_output), \
                    "Eagerness" : float(excited_output), "Friendliness" : float(friendly_output),\
                        "EyeContact": float(eye_output), "STAR" : star_output, "Coherency": coherency_output, \
                            "Linkword Usage": linkword_output})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
