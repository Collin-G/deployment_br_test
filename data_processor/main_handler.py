import tempfile
import ffmpeg
import dlib
import whisper
import os

from video_handler import VideoHandler
from audio_handler import AudioHandler


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
whisper_model = whisper.load_model("base")

class MainHandler():
    def __init__(self, video):
        self.video = video

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(self.video.read())
            temp_video.close()
            temp_audio_path = self.extract_audio_from_video(temp_video.name)

        self.temp_audio_path = temp_audio_path
        self.video_handler = VideoHandler(temp_video.name, detector, predictor)
        self.audio_handler = AudioHandler(temp_audio_path)



    def handle_data(self):
        video_features = self.video_handler.get_video_features()
        audio_features = self.audio_handler.get_mfcc_features()
        text_features  = self.get_speech_to_text(self.temp_audio_path)
        
        os.unlink(self.video.name)
        os.unlink(self.temp_audio_path)  # Delete the temporary audio file

        return {"video" : video_features, "audio": audio_features, "text": text_features}

    
    def get_speech_to_text(self, audio_file):
        result = whisper_model.transcribe(audio_file)
        # Print the transcribed text
        return result["text"]


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