from flask import Flask, render_template, request, redirect, flash, url_for
from werkzeug.utils import secure_filename
import tempfile
import os

app = Flask(__name__)

UPLOAD_FOLDER = tempfile.TemporaryDirectory()
SECRET_KEY = os.urandom(24)

app.config['SECRET_KEY'] = SECRET_KEY
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER.name)

print(UPLOAD_FOLDER.name)


@app.route("/", methods=['GET', 'POST'])
def index():
    # handle post requests
    if request.method == 'POST':
        # check if the post request has the file part
        if 'video' not in request.files:
            flash('No video file')
            return redirect(request.url)
        
        if 'logs' not in request.files:
            flash('No logs file')
            return redirect(request.url)
        
        video = request.files['video']
        logs = request.files['logs']
        # if user does not select file, browser also
        # submit an empty part without filename
        if video.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if logs.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        filename = secure_filename(video.filename)
        video.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        filename = secure_filename(logs.filename)
        logs.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        flash('Upload successful, data processing...')
        
        # TODO: PROCESS
        # TODO: Add filetype checks
        
        flash('Uploading data to rvision')
    
    return render_template('upload.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)