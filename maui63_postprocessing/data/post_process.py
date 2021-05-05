from __future__ import annotations
from typing import Union

from maui63_postprocessing.data.uav_import import Maui63UAVImporter
from maui63_postprocessing.videoedit.highlights import Highlighter
from maui63_postprocessing.cv import process_video, process_image

import os
import shutil
import requests
import json
import pandas as pd
import tempfile
import filetype  # This might be unnecessary
from pathlib import Path
from tqdm import tqdm
import warnings
import time
import copy
import scipy
import scipy.interpolate
from moviepy.editor import  VideoFileClip
import cv2

class Maui63DataProcessor:
    
    def __init__(self,
                 logs,
                 media, 
                 data_file, 
                 config_file, 
                 weights, 
                 names_file,
                 output_path = '__temp__',
                 csv_output_path = None,
                 tag_media = True,  # add boxes
                 highlighter_kwargs = {},
                 cv_kwargs = {},
                 media_start_time = None,
                 image_dir_fps = None,
                 image_dir_timestamps = None,
                 export_type: str = 'video',
                 ):
        
        if csv_output_path is not None:
            csv_output_path = str(csv_output_path)
        
        self.logs = str(logs)
        self.media = str(media)
        self.data_file = str(data_file)
        self.config_file = str(config_file)
        self.weights = str(weights)
        self.names_file = str(names_file)
        self.output_path = str(output_path)
        self.tag_media = tag_media
        self.highlighter_kwargs = highlighter_kwargs  # TODO: document
        self.cv_kwargs = cv_kwargs                    # TODO: document
        self.csv_output_path = csv_output_path
        self.media_start_time = media_start_time
        
        # Make sure we don't have both
        assert image_dir_fps == None or image_dir_timestamps == None
        self.image_dir_fps = image_dir_fps
        self.image_dir_timestamps = image_dir_timestamps
        
        # Get media filetype
        self._media_type, self._media_extension = self._get_filetype()
        assert self._media_type in ['image', 'video', 'dir']
        
        # Check output type makes sense
        self._output_type, self._output_extension = self._get_filetype(output_path)
        assert self._media_type == self._output_type or \
             (self._output_extension == '' and self._media_type == 'video'), \
            "Input and output types must match (or dir for video highlights)" \
            + '\n\nOutput_type = {} | Media_type = {}'.format(
                self._output_type, self._media_type)
            
        # For exporting to rvision
        if export_type != None:
            # assert self._output_type == 'dir'
            assert export_type in ['video', 'image']
            self.export_type = export_type
        
    def _get_filetype(self, file=None):
        """
        This isn't perfect, but close enough

        """
        
        if file is None:
            file = self.media
        file = str(file)
        
        try:
            kind = filetype.guess(file)
            mime = kind.mime.split('/')
        except IsADirectoryError:
            mime = ['dir', '']
        except FileNotFoundError:
            # Manually find the type
            extension = os.path.splitext(file)[1].lstrip('.')
            if extension == '':
                mime = ['dir', '']
            else:
                for ftype in filetype.types:
                    fmime = ftype.mime.split('/')
                    if fmime[1] == extension:
                        mime = fmime
                        break
                    elif fmime[1] == 'jpeg' and extension == 'jpg':
                        mime = fmime
                        mime[1] = 'jpg'
                        break
           
        # TODO: fix this so it works with both, quick fix to match both
        # media_extension = os.path.splitext(self.media)[1].lstrip('.')
        # if mime[1] == 'jpeg' and media_extension == 'jpg':
        #     mime[1] = 'jpg'
            
        return mime[0], mime[1]
        
        
    def _import_data(self):
        self.importer = Maui63UAVImporter(self.logs)
        
        # make a copy of the data
        df = copy.deepcopy(self.importer.df)
        
        # add a prefix to the columns
        df.rename(columns = {x:('uav_'+x) for x in df.columns}, inplace = True) 
        
        self.uav_df = df
        
    def _merge_uav_cv_datasets(self):
    
        # TODO: do not assume equal start times
        df = copy.deepcopy(self.uav_df)
        
        # TODO: assuming video and logs are synced at the beginning for now
        if self.media_start_time == None:
            self.media_start_time = df['uav_unix_time'].min()
            
        df['timestamp'] = df['uav_unix_time'] - self.media_start_time
        
        # create linear interpolators based on log data
        data_cols = list(df.columns)
        data_cols.remove('timestamp')
        interpolators = {}
        x = df.timestamp
        for column in data_cols:
            y = df[column]
            
            if y.dtype.kind in 'biufc':
                interpolators[column] = scipy.interpolate.interp1d(x, y)
            
            else:
                # if column is non-numeric interp1d(kind='nearest')
                # https://stackoverflow.com/questions/62015823/interpolating-categorical-data-in-python-nearest-previous-value
                
                def interp_f_factory():
                    def interp_f(x):
                        f = scipy.interpolate.interp1d(
                            x, range(len(y)), kind='nearest',
                            fill_value=(0, len(y)-1), bounds_error=False
                            )            
                        y_idx = f(x)
                        y_new = [y[int(i)] for i in y_idx]
                        return y_new
                    return interp_f
                
                interpolators[column] = interp_f_factory()
                
        self._interpolators = interpolators
        
        # Now put it in with interpolation
        x = self.data.timestamp
        for column in data_cols:
            self.data[column] = interpolators[column](x)
        
    def _generate_video_highlights(self, df_in = None):
        self.highlighter = Highlighter(self._video_temp_file,
                                       self.dnn_df.timestamp,
                                       **self.highlighter_kwargs)
        
        clips, groups = self.highlighter.create_clips()
        
        df = pd.DataFrame()
        for i, clip in enumerate(clips):
            
            print('Saving highlight {} of {}: \n'.format(i, len(clips)-1))
            time.sleep(0.2)
            
            #create a folder if needed
            Path(self.output_path).mkdir(parents=True, exist_ok=True)
            
            # filename format = filename + mingroup timestamp
            filename = (
                self.output_path.rstrip('/') + '/' +
                self.media.split(os.sep)[-1].rstrip('.' + self._media_extension) +  
                '-' + str(min(groups[i])) + '.' + self._media_extension
                )
            
            clip.write_videofile(filename, verbose=False)
            
            df_tmp = pd.DataFrame({'timestamp': groups[i]})
            df_tmp['filename'] = filename
            df = df.append(df_tmp, ignore_index = True)
            
        self._clips = clips
        self._groups = groups
        
        if type(df_in) != type(None):
            df = df_in.merge(df, how='inner', on='timestamp')
        
        return df
    
    def _get_detection_frame(self, timestamp):
        # To avoid creating a clip object for just one frame
        if not hasattr(self, '_media_clip'):
            self._media_clip = VideoFileClip(self.media)
        clip = self._media_clip
        
        # Simply get them from the original file
        frame = clip.get_frame(timestamp)
        
        return frame
        
    def _run_cv(self):
        
        if self._media_type == 'video':
            if self._output_extension != '':
                # we're making a single file
                if self.tag_media:
                    file = self.output_path
                else:
                    file = None  # No output from processing
                    shutil.copyfile(self.media, self.output_path)  # copy the original to the output
            else:
                # We're creating subclips
                self._video_temp_file = tempfile.NamedTemporaryFile(
                    suffix='.' + self._media_extension).name
                
                if self.tag_media:
                    file = self._video_temp_file
                else:
                    file = None  # No output from processing
                    shutil.copyfile(self.media, self._video_temp_file)  # copy the original to the output
            
            df = process_video(self.media,
                               self.data_file,
                               self.config_file,
                               self.weights,
                               self.names_file,
                               output_file = file,
                               **self.cv_kwargs)
            
            df.filename = file
            
        if self._media_type == 'image':
            if self.tag_media:
                file = self.output_path
            else:
                file = None  # No output from processing
                shutil.copyfile(self.media, self.output_path)  # copy the original to the output
                
            df = process_image(self.media,
                               self.data_file,
                               self.config_file,
                               self.weights,
                               self.names_file,
                               output_file = file,
                               **self.cv_kwargs)
            
            df['filename'] = file
            df['timestamp'] = 0
            
        if self._media_type == 'dir':
            if self.image_dir_fps != None:
                fps = self.image_dir_fps
            
            df = pd.DataFrame()
            output_dir = self.output_path.rstrip('/') + '/' # just to make sure it has a slash
            for i, filename in tqdm(enumerate(os.listdir(self.media))):
                f_type, _ = self._get_filetype(filename, check = False)
                if f_type == 'image':
                    if self.tag_media:
                        file = output_dir + filename # TODO: figure out if we're happy with this
                    else:
                        file = None  # No output from processing
                        shutil.copyfile(self.media.rstrip('/') + '/' + filename, output_dir + filename)  # copy the original to the output
                    df = process_image(self.media,
                                       self.data_file,
                                       self.config_file,
                                       self.weights,
                                       self.names_file,
                                       output_file = file,
                                       **self.cv_kwargs)
                    
                    df['filename'] = file
                    
                    if self.image_dir_fps != None:
                        df['timestamp'] = fps*i
                        
                    series = df.squeeze()
                    df.append(series, ignore_index=True)
                else:
                    warnings.warn('Non-media file in media directory ({}), skipping...'.format(filename))
                    continue
    
        if self.image_dir_timestamps != None:
            df['timestamp'] = self.image_dir_timestamps
            
        self.dnn_df = df
        return df
    
        
    def process(self):
        
        # Import log data
        self._import_data()
        
        # Todo: ask to rerun if self.dnn_df exists
        df = self._run_cv()
        
        if self._media_type == 'video' and self._output_extension == '':
            if self.export_type == 'video':
                # generate some highlights from the tempfile
                df = self._generate_video_highlights(df_in = df)
            if self.export_type == 'image':
                raise NotImplementedError()
        
        self.data = df
        
        # Now merge the datasets
        self._merge_uav_cv_datasets()
        
        return df

    
    # in case you don't want to rerun the opencv code
    def _save_temp_output(self,
                          data_df_csv = '__temp__.csv',
                          dnn_df_csv = '__temp__.df_dnn.csv',
                          video_name = '__temp__.video_dnn',  # Only if _video_temp_file is defined
                          ):
        """
        A debug function for saving important data to reload later
        """
        
        # make sure there's a trailing slash
        
        if dnn_df_csv != None:
            print('Exporting dnn data...')
            self.dnn_df.to_csv(dnn_df_csv.rstrip('.csv') + '.csv')
        
        print('Moving video data...') 
        if hasattr(self, '_video_temp_file'):
            shutil.copyfile(self._video_temp_file, 
                            video_name + self._media_extension)
            
        if data_df_csv != None:
            self.data.to_csv(data_df_csv)
            
        print('Done')
        
    def _load_processed_data(self,
                             data_df_csv = "__temp__.csv",
                             dnn_df_csv = None,
                             video_name = None,  # Only if _video_temp_file is defined
                             ):
        """
        A debug function for reloading data outputs saved prior
        """
        
        print("WARNING: This method uses eval to convert strings")
        print("         Do no use with non-trusted datasets")
        input("press ENTER to continue, ctrl+C to cancel")
        
        colums_to_eval = ['box', 'confidence', 'object_class']
        
        if dnn_df_csv != None:
            self.df_dnn = pd.read_csv(dnn_df_csv)
            
            for col in colums_to_eval:
                if col in self.df_dnn.columns:
                    self.df_dnn[col] = self.df_dnn[col].apply(lambda x: eval(x))
        
        if video_name != None:
            self._video_temp_file = video_name
        
        if data_df_csv != None:
            self.data = pd.read_csv(data_df_csv)
            
            for col in colums_to_eval:
                if col in self.data.columns:
                    self.data[col] = self.data[col].apply(lambda x: eval(x))
        
        
    
    def export_csv(self, csv_output_path = None):
        
        if csv_output_path == None:
            csv_output_path = self.csv_output_path
        
        assert csv_output_path != None, \
            'Please specify csv_output_path.'
            
        path = os.getcwd()
            
        print("Exporting data to {}".format(path + '/' + csv_output_path))
        
        self.data.to_csv(path + '/' + self.csv_output_path)
        
    
    rvision_url = "https://be.uat.rvision.rush.co.nz/api/v1/alpr/camera/<camera>/analyse-image/?token=<camera_token>"
    def export_rvision(self,
                       url: str = None,
                       rvision_camera: str = None,
                       rvision_token: str = None,       # <camera token>
                       min_spacing: float = 30,    # Minimum spacing in seconds
                       ):
        
        assert ((rvision_camera != None and rvision_token != None) or
                url != None), "Please specify the camera and token, or provide the full URL"
        
        # TODO: Add warnings if you add multiple
        
        if url == None:
            url = self.rvision_url.replace('<camera_token>', str(rvision_token))
            url = url.replace('<camera>', str(rvision_camera))
        
        # Upload images for each detection (or nth detection)    
        data_to_send = copy.deepcopy(self.data)
        
        # Filter events that are too close to each other
        print("Finding spaced frames:")
        time.sleep(0.2)
        
        data_to_send = data_to_send.sort_values('timestamp')
        last_timestamp = data_to_send['timestamp'].min()
        idx = [0]  # Should be 0 since it's sorted
        
        for i in tqdm(range(len(data_to_send))):
            row = data_to_send.iloc[i]
            
            if row.timestamp >= last_timestamp + min_spacing:
                last_timestamp = row.timestamp
                idx.append(i)
        
        data_to_send = data_to_send.iloc[idx]
        
        time.sleep(0.2)
        print('Uploading frames:')
        time.sleep(0.2)
        for i in tqdm(range(len(data_to_send))):
            row = data_to_send.iloc[i]
            
            detections = []
            for idx in range(row.num_objects):
                bbox = row.box[idx]
                
                (x, y) = (int(bbox[0]), int(bbox[1]))
                (w, h) = (int(bbox[2]), int(bbox[3]))
                
                detections.append({
                        "bb": {  
                                       "b": y, 
                                       "l": x, 
                                       "r": x + w, 
                                       "t": y + h
                        },
                        "object_class": int(row.object_class[idx]),
                        "confidence": float(row.confidence[idx])*100
                    })
            
            data_dict = {
                "model": -2, # Dolphins
                "detections": detections,
                "lat": float(row['uav_ UAV lat']),  # TODO: Fix this column
                "lng": float(row['uav_ UAV long'])
                }
            
            image = self._get_detection_frame(row.timestamp)
            
            data_json = json.dumps(data_dict)
            
            self._send_frame(url ,image, data_json)
            
    def _send_frame(self, url, image, data_json):
        
        # Create a temporary file to then upload
        image_path = tempfile.NamedTemporaryFile(
            suffix='.png').name
        
        cv2.imwrite(image_path, image)
        
        with open(image_path, 'rb') as image:
            response = requests.post(
                url,
                files={
                    'image': image,
                    'json': data_json
                }
            )
            
            assert response.status_code != 404, \
                "Error 404: Not Found\nPlease check your rvision camera token."
        
            assert response.status_code == 200, \
                "Recieved status code {}, expected status code 200 (OK)".format(response.status_code)
    
if __name__ == '__main__':
    
    # logs = '../../../drone_Xavier_log_27.01.2021.txt'
    logs = '../../../Drone_Flight_Path_Dummy_Data.csv'
    
    media = '../../../testingvideos/mauitest_11_40s_1080.mp4'
    data_file = '../../../maui_sf_and_100m.data'
    config_file = '../../../yolov4-tiny-maui-sf-and-100m.cfg'
    weights = '../../../yolov4-tiny-maui-sf-and-100m_best.weights'
    names_file = '../../../maui.names'
    
    highlighter_kwargs = {'clip_length': 10, 'padding': 3}
    
    pro = Maui63DataProcessor(
        logs,
        media, 
        data_file, 
        config_file, 
        weights, 
        names_file,
        highlighter_kwargs = highlighter_kwargs
        )
    
    # pro.process()
    pro._load_processed_data(data_df_csv = '../../examples/__temp__.csv')
    url = input('Rvision URL: ')
    pro.export_rvision(url = url)
