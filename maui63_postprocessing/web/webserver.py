from flask import Flask, render_template, request, redirect, flash, url_for
from werkzeug.utils import secure_filename
import tempfile
import os
import sys
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

import threading
from multiprocessing import Process, get_context
from multiprocessing.queues import Queue

from maui63_postprocessing import Maui63DataProcessor

_filedirpath = Path(__file__).parent


class StdoutQueue(Queue):
    def __init__(self, *args, **kwargs):
        self.block = kwargs.get('block')
        self.timeout = kwargs.get('timeout')
        super().__init__(*args, ctx = get_context(), **kwargs)
        
    def write(self, msg):
        self.put(msg)
        
    def flush(self):
        sys.__stdout__.flush()


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
        template_path = _filedirpath / 'templates/'
        
        super().__init__(*args, 
                         template_folder=str(template_path),
                         **kwargs)
        
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
    
    def process_upload(self, uav_logs, media_file, r_url):
        
        # TODO: Add option to offload to azure instance
        
        # TODO: add support for multiple uploads (this will break if two people upload at the same time)
        output_path = self.config['UPLOAD_FOLDER'] + '/output.mp4'
        
        def process(stdout_queue, stderr_queue):
            
            # redirect stdout
            with redirect_stdout(stdout_queue), redirect_stderr(stderr_queue):
                # sys.stdout = sys.__stdout__
                
                try:
                    print('Processing...')
                    
                    processor = Maui63DataProcessor(
                            uav_logs,            # CSV file for UAV data logs
                            media_file,          # video/image file or image folder
                            self.data_file,      # darknet .data file
                            self.config_file,    # darknet .cfg file
                            self.weights_file,   # darknet .weights file
                            self.names_file,     # darknet .names file
                            output_path,         # ouput file/directory
                            highlighter_kwargs = self.highlighter_kwargs,
                            **self.processor_kwargs
                            )
                            
                    # Run the process routine
                    processor.process()
                    
                    print('Exporting...')
                    
                    # TODO: Add to rvision
                    processor.export_rvision(url = r_url, **self.export_kwargs)
                
                    print('Done!')
                    
                except Exception as e:
                    print(e)
                
            # Lazy way of passing the state through the queue
            print('__STOP__')
        
        # queue to catch standard output from subprocess
        stdout_queue = StdoutQueue()
        stderr_queue = StdoutQueue()
        self.queue_stdoud = stdout_queue
        
        # create a subprocess
        self.child_process = Process(target=process, args=(stdout_queue, stderr_queue,))
        self.child_process.start()
        
        print('Starting task...')
        
        # catch the various process outputs
        while True:
            try:
                output = stdout_queue.get_nowait()
            except:
                output = None
            
            try:                
                stderr = stderr_queue.get_nowait()
            except:
                stderr = None
            
            if stderr:
                print(stderr)
            
            # Kill signal
            if output == '__STOP__':
                print('STOP')
                break
            
            # process output
            if output:
                print(output)
            
        self.child_process.join()
        self.child_process.close()
            
    
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
                
                r_url = str(request.form['rvision_url'])
                
                # if user does not select file, browser also
                # submit an empty part without filename
                if video.filename == '':
                    flash('No selected file')
                    return redirect(request.url)
                if logs.filename == '':
                    flash('No selected file')
                    return redirect(request.url)
                if r_url == '':
                    flash('No URL')
                    return redirect(request.url)
                
                filename = secure_filename(video.filename)
                videopath = os.path.join(self.config['UPLOAD_FOLDER'], filename)
                video.save(videopath)
                
                filename = secure_filename(logs.filename)
                logspath = os.path.join(self.config['UPLOAD_FOLDER'], filename)
                logs.save(logspath)
                
                flash('Upload successful.')
                
                # TODO: Add filetype checks
                                
                flash('Processing...')
                
                try:
                    # Process the info in a thread (keeps things from hanging I think?)
                    thread = threading.Thread(
                        target=self.process_upload,
                        args=(logspath, videopath, r_url)
                        )
                    thread.start()
                    
                    thread.join()
                    
                    flash("Processing complete")
                    
                    # TODO: Catch exceptions here
                    
                    # TODO: Add files to blob storage
                    pass
                    
                except Exception as e:
                    flash('Something went wrong, please check the URL, or contact the software administrator:')
                    flash(str(e))
                    return redirect(request.url)
            
            return render_template('upload.html', rvision_url=self.rvision_url)


if __name__ == "__main__":
    
    os.chdir('../../../')
    
    app = UploadPage(__name__)
    app.rvision_url = "dummyurl"
    
    kwargs = {'host': '0.0.0.0', 'port': 8080, 'debug': False}
    threading.Thread(target=app.run, 
                      kwargs = kwargs
                      ).start()
    # app.run(**kwargs)