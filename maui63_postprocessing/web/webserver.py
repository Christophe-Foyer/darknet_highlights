from flask import Flask, render_template, request, redirect, flash, url_for
from werkzeug.utils import secure_filename
import tempfile
import os
from pathlib import Path
import threading

from maui63_postprocessing import Maui63DataProcessor


class UploadPage(Flask):
    
    # TODO: Add config file support
    data_file = 'maui_sf_and_100m.data'
    config_file = 'yolov4-tiny-maui-sf-and-100m.cfg'
    weights_file = 'yolov4-tiny-maui-sf-and-100m_best.weights'
    names_file = 'maui.names'
    
    rvision_url = None
    
    highlighter_kwargs = {'clip_length': 10, 'padding': 3}
    export_kwargs = {}
    processor_kwargs = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        app = self
        
        self.UPLOAD_FOLDER = tempfile.TemporaryDirectory()
        SECRET_KEY = os.urandom(24)
        
        app.config['SECRET_KEY'] = SECRET_KEY
        app.config['UPLOAD_FOLDER'] = str(self.UPLOAD_FOLDER.name)
        
        # Add the routes
        self._add_routes()
        
    def run(self, *args, **kwargs):
        # Check some things before running
        assert self.rvision_url, "Please specify an rvision url. (UploadPage.rvision_url)"
        
        super().run(*args, **kwargs)
    
    def process_upload(self, uav_logs, media_file):
        
        # TODO: Add option to offload to azure instance
        
        output_path = self.config['UPLOAD_FOLDER'] + '/output.mp4'
        print(output_path)
        
        processor = Maui63DataProcessor(
                uav_logs,       # CSV file for UAV data logs
                media_file,     # video/image file or image folder
                self.data_file,      # darknet .data file
                self.config_file,    # darknet .cfg file
                self.weights_file,   # darknet .weights file
                self.names_file,     # darknet .names file
                output_path,     # ouput file/directory
                highlighter_kwargs = self.highlighter_kwargs,
                **self.processor_kwargs
                )
                
        # Run the process routine
        processor.process()
        
        # TODO: Add to rvision
        processor.export_rvision(url = self.rvision_url, **self.export_kwargs)
            
    
    def _add_routes(self):
        @self.route("/", methods=['GET', 'POST'])
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
                videopath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                video.save(videopath)
                
                filename = secure_filename(logs.filename)
                logspath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                logs.save(logspath)
                
                flash('Upload successful.')
                
                # TODO: Add filetype checks
                                
                flash('Processing...')
                
                try:
                    # Process the info in a thread (keeps things from hanging I think?)
                    thread = threading.Thread(
                        target=self.process_upload,
                        args=(logspath, videopath)
                        )
                    thread.start()
                    
                    thread.join()
                    
                    flash("Processing complete")
                    
                    # TODO: Add files to blob storage
                    pass
                    
                except:
                    flash('Something went wrong, please contact the software administrator.')
                    raise
            
            return render_template('upload.html')


if __name__ == "__main__":
    
    os.chdir('../../../')
    
    app = UploadPage(__name__)
    app.rvision_url = "dummyurl"
    
    threading.Thread(target=app.run, 
                     kwargs = {'host': '0.0.0.0',
                               'port': 8080,
                               'debug': False}
                     ).start()