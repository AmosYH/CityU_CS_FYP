import os
from flask import Response, Flask, request, send_file, flash, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import uuid
import datetime
import cv2
from process import enhance_light_image, enhance_light_video
from subprocess import Popen, PIPE

WEBSERVICE_PORT = int(os.environ["WEBSERVICE_PORT"])
UPLOAD_FOLDER = os.environ["UPLOAD_FOLDER"]
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'mpeg', 'avi', 'mov', 'wmv', 'flv', 'avchd', 'webm', 'mkv'}
ALLOWED_PHOTO_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'}

def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def encode_videofile(output_filepath, visualize_filepath):
    # Encode
    cmd_ffmpeg = ['ffmpeg', '-y', '-threads', '8', '-i', output_filepath, '-pix_fmt', 'yuv420p', '-c:v',
                  'libx264', '-crf', '30', '-preset', 'veryfast', '-c:a', 'aac', '-b:a', '128k',
                  visualize_filepath]
    with Popen(cmd_ffmpeg, text=True, stdout=PIPE, stderr=PIPE) as p:
        for line in p.stderr:
            print(line, end="")


application = Flask(__name__, static_folder=UPLOAD_FOLDER, static_url_path='')

@application.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@application.route('/process_video/', methods=['GET', 'POST'])
def process_video():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
            filename = secure_filename(file.filename)
            extension=filename.split('.')[-1]
            uploaded_filename = f'{str(uuid.uuid4())}.{extension}'
            path=os.path.join(application.config['UPLOAD_FOLDER'], uploaded_filename)
            file.save(path)
            print(f'Videofile upload successfully, path={path}')

            output_filepath = path + '.output_merged.avi'
            output_filepath2 = path + '.output.avi'
            visualize_filepath = os.path.join(application.config['UPLOAD_FOLDER'], 'results/'+path.split('/')[-1]) + '.output_merged.mp4'
            visualize_filepath2 = os.path.join(application.config['UPLOAD_FOLDER'], 'results/'+path.split('/')[-1]) + '.output.mp4'
            print(f'Vis merged filepath={visualize_filepath}')
            print(f'Vis filepath={visualize_filepath2}')
            enhance_light_video(path, output_filepath, output_filepath2)

            #Merged Video
            encode_videofile(output_filepath, visualize_filepath)
            #Single Video
            encode_videofile(output_filepath2, visualize_filepath2)

            return render_template('process_video.html', output_video_filename='/results/'+visualize_filepath.split('/')[-1])
        else:
            print('Unknown file extension')
            return {"status": "error", "message": 'Unknown file extension'}, 404

    return render_template('submit_video.html'), 200


@application.route('/process_photo/', methods=['GET', 'POST'])
def process_photo():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print('Not selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename, ALLOWED_PHOTO_EXTENSIONS):
            filename = secure_filename(file.filename)
            extension=filename.split('.')[-1]
            uploded_filename = f'{str(uuid.uuid4())}.{extension}'
            path=os.path.join(application.config['UPLOAD_FOLDER'], uploded_filename)
            file.save(path)

            processed_image_path = enhance_light_image(path)

            print(f'Upload is done, path={path}')
            return render_template('process_photo.html', input_filename=uploded_filename, output_filename=processed_image_path)
        else:
            print('Unknown file extension')
            return {"status": "error", "message": 'Unknown file extension'}, 404

    return render_template('submit_image.html'), 200


if __name__ == '__main__':
    application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    application.run(debug=False, port=WEBSERVICE_PORT, host='0.0.0.0')
