import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from statistics import mode
import imutils
import pickle
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
from flask import Flask, request, jsonify
from flask_cors import cross_origin
import json
import time
import os


ALLOWED_AUD_EXTENSIONS = ['wav', 'mp3']
if not os.path.exists('staticfiles'):
    os.makedirs('staticfiles')
if not os.path.exists('staticfiles/image'):
    os.makedirs('staticfiles/image')
if not os.path.exists('staticfiles/audio'):
    os.makedirs('staticfiles/audio')
if not os.path.exists('staticfiles/spectro'):
    os.makedirs('staticfiles/spectro')
PROJECT_ROOT = os.getcwd()

def allowed_aud_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_AUD_EXTENSIONS

def melspectro(file):
    signal, sr = librosa.load(file, duration=30)
    N_FFT = 1024
    HOP_SIZE = 1024
    N_MELS = 128
    WIN_SIZE = 1024
    WINDOW_TYPE = 'hann'
    FEATURE = 'mel'
    FMIN = 0

    S = librosa.feature.melspectrogram(y=signal, sr=sr,
                                       n_fft=N_FFT,
                                       hop_length=HOP_SIZE,
                                       n_mels=N_MELS,
                                       htk=True,
                                       fmin=FMIN,
                                       fmax=sr / 2)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(S ** 2, ref=np.max), fmin=FMIN, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    image_name = file.split('.')[0].split('/')[-1]
    image_path = PROJECT_ROOT + "/staticfiles/spectro/" + image_name+'.jpg'
    plt.savefig(image_path, bbox_inches='tight', transparent=True, pad_inches=0)
    return image_path


def crop(file):
    img = Image.open(file)
    w, h = img.size
    area = (70, 0, 700, 318)
    img = img.crop(area)
    img.save(file.replace('.jpg', '.jpg'))


def predict_audio(audio):
    K_L = []
    K_P = []
    T = []
    image_path = melspectro(audio)
    crop(image_path)
    image = cv2.imread(image_path)
    output = imutils.resize(image, width=400)
    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    model = load_model('audio_350.model')
    mlb = pickle.loads(open('audio_mlb_350.pickle', "rb").read())
    proba = model.predict(image)[0]
    idxs = np.argsort(proba)[::-1][:2]
    for (i, j) in enumerate(idxs):
        label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
        cv2.putText(output, label, (10, (i * 30) + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    for (label, p) in zip(mlb.classes_, proba):
        pass
    cv2.imwrite(image_path, output)
    for (label, p) in zip(mlb.classes_, proba):
        K_L.append(label)
        K_P.append(p)
        t = "{}: {:.2f}%".format(label, p * 100)
        T.append(t)
    k = K_P.index(max(K_P))
    label = "{}: {:.2f}%".format(mlb.classes_[k], proba[k] * 100)
    os.remove(image_path)
    return label, T

emotion_model_path = './emotion_model/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')
frame_window = 10
emotion_offsets = (20, 40)
face_cascade = cv2.CascadeClassifier('./emotion_model/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path)
emotion_target_size = emotion_classifier.input_shape[1:3]

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)


def get_emotion(img):
    emotion_window = []
    bgr_image1 = cv2.imread(img)
    bgr_image = ResizeWithAspectRatio(bgr_image1, width=1280)
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

    for face_coordinates in faces:
        (x1, x2, y1, y2) = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, emotion_target_size)
        except Exception as e:
            continue
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode, color, 0, -45, 1, 1)
        prob = int(emotion_probability*100)
        return emotion_mode, prob


ALLOWED_IMG_EXTENSIONS = ['jpg', 'png', 'jfif', 'raw', 'gif']

def allowed_img_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMG_EXTENSIONS

app = Flask(__name__)
@app.route('/api/v1/emotion_analysis', methods=['POST', 'GET'])
@cross_origin()
def emotion():
    if request.method == 'POST':
        images = request.files.getlist('image')
        for image in images:
            if image == '':
                resp = jsonify({'message': 'bad request'})
                resp.status_code = 400
                return resp

        errors = {}
        success = True
        res_jason = '{}'
        res_jason = json.loads(res_jason)
        results_list = []
        for image in images:
            if image and allowed_img_file(image.filename):
                img_file_path = PROJECT_ROOT + '/staticfiles/image/' + image.filename
                image.save(img_file_path)
                try:
                    label, prob = get_emotion(img_file_path)
                except TypeError as e:
                    pass

                res = {'emotion': str(label).capitalize(), 'file_name': str(image.filename), 'probability': float(prob)}
                os.remove(img_file_path)
                results_list.append(res)

            else:
                errors[image.filename] = 'Image type is not allowed in one of upload images.'
                errors[image.filename] = 'allowed image types: jpg, png, jfif, raw, gif'
                success = False

        if success:
            jsonStr = json.dumps(results_list)
            return jsonify(response=results_list)

    return '''
                <!doctype html>
                <title>Images Upload </title>
                <h1>Upload the Images</h1>
                <form method=post enctype=multipart/form-data action=http://nipunanisaga.xyz/api/v1/emotion_analysis>
                  <label for="file_image">Image file:</label> 
                  <input type=file name=image id="image" multiple><br><br>
                  <input type=submit value=Upload>
                </form>
                '''

@app.route('/api/v1/audio_analysis', methods=['POST', 'GET'])
@cross_origin()
def audio_api():
    if request.method == 'POST':
        audio = request.files.get('audio')

        if audio not in request.files:
            if audio == '':
                resp = jsonify({'message': 'incomplete request'})
                resp.status_code = 200
                return resp

        errors = {}
        success = False
        audio_file_name = audio.filename

        if audio and allowed_aud_file(audio_file_name):
            aud_path = PROJECT_ROOT + '/staticfiles/audio/' + audio.filename
            audio.save(aud_path)
            label, T = predict_audio(aud_path)
            os.remove(aud_path)
            success = True

        else:
            errors[audio_file_name] = 'video type is not allowed'

        if success:
            lis = []
            for item in T:
                inner_item = {'emotion': item.split(" ", 1)[0][:-1], 'probability': float(item.split(" ", 1)[1][:-1]) }
                lis.append(inner_item)

            resp = jsonify({'main_prediction': { 'emotion': label.split(" ", 1)[0][:-1], 'probability': float(label.split(" ", 1)[1][:-1]) },
                            'predictions': lis, 'file_name': audio_file_name})
            resp.status_code = 200
            return resp

    return '''
                <!doctype html>
                <title>Audio Upload </title>
                <h1>Upload the Audio</h1>
                <form method=post enctype=multipart/form-data action=http://nipunanisaga.xyz/api/v1/audio_analysis>
                  <label for="file_audio">Audio file:</label> 
                  <input type=file name=audio id="audio"><br><br>
                  <input type=submit value=Upload>
                </form>
                '''

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8000, debug=False)
