from __future__ import annotations
from typing import Union

from darknet_highlights.data.uav_import import Maui63UAVImporter
# from darknet_highlights.data.export import Maui63UAVExporters
from darknet_highlights.videoedit.highlights import Highlighter
from darknet_highlights.cv import process_video, process_image
import os
import shutil
import pandas as pd
import tempfile
import filetype  # This might be unnecessary
from pathlib import Path
from tqdm import tqdm
import warnings
import time
import copy
import numpy as np
import datetime

class Maui63DataProcessor:
    
    def __init__(self,
                 logs,
                 media, 
                 data_file, 
                 config_file, 
                 weights, 
                 names_file,
                 output_path = '__temp__.data_out',
                 csv_output_path = None,
                 tag_media = True,  # add boxes
                 highlighter_kwargs = {},
                 ):
        
        self.logs = logs
        self.media = media
        self.data_file = data_file
        self.config_file = config_file
        self.weights = weights
        self.names_file = names_file
        self.output_path = output_path
        self.tag_media = tag_media
        self.highlighter_kwargs = highlighter_kwargs  # TODO: clarify
        self.csv_output_path = csv_output_path
        
        self._media_type, self._media_extension = self._get_filetype()
        assert self._media_type in ['image', 'video', 'dir']
        
        # Might not exist yet so use os
        _, self._output_extension = os.path.splitext(output_path)
        assert self._output_extension == self._media_extension or \
            (self._output_extension == '' and self._media_type == 'video'), \
            "Input and output types must match (or dir for video highlights)"
        
    def _get_filetype(self, file=None):
        """
        This isn't perfect, but close enough

        """
        
        if file is None:
            file = self.media
        
        try:
            kind = filetype.guess(file)
            mime = kind.mime.split('/')
        except IsADirectoryError:
            mime = ['dir', '']
            
        return mime[0], mime[1]
        
        
    def _import_data(self):
        self.importer = Maui63UAVImporter(self.logs)
        
        df = copy.deepcopy(self.importer.df)
        
        df.rename(columns = {x:('uav_'+x) for x in df.columns}, inplace = True) 
        
        self.uav_df = df
        
    def _merge_uav_cv_datasets(self):
    
        # TODO: do not assume equal start times
        df = copy.deepcopy(self.uav_df)
        
        df['timestamp'] = (df.datetime - df.datetime.min()).apply(
            lambda x: x / datetime.timedelta(microseconds=1) / 1E6)
        
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
                               output_file = file)
            
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
                               output_file = file)
            
            df['filename'] = file
            
        if self._media_type == 'dir':
            df = pd.DataFrame()
            output_dir = self.output_path.rstrip('/') + '/' # just to make sure it has a slash
            for filename in tqdm(os.listdir(self.media)):
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
                                       output_file = file)
                    
                    df['filename'] = file
                    series = df.squeeze()
                    df.append(series, ignore_index=True)
                else:
                    warnings.warn('Non-media file in media directory ({}), skipping...'.format(filename))
                    continue
    
        self.dnn_df = df
        
        
    def process(self):
        
        # Todo: ask to rerun if self.dnn_df exists
        self._run_cv()
        
        self._import_data()
        
        if self._media_type == 'video':
            if self._output_extension == '':
                # generate some highlights from the tempfile
                df = self._generate_video_highlights(df_in = self.dnn_df)
        
        self.data = df
        
        return df

    
    # in case you don't want to rerun the opencv code
    def _save_temp_output(self, directory = './',
                          dnn_df_csv = '__temp__.df_dnn.csv',
                          video_name = '__temp__.video_dnn',  # Only if _video_temp_file is defined
                          data_df_csv = None,
                          ):
        """
        A debug function for saving important data to reload later
        """
        
        # make sure there's a trailing slash
        directory = directory.rstrip('/') + '/'
        
        print('Exporting dnn data...')
        self.dnn_df.to_csv(directory + dnn_df_csv.rstrip('.csv') + '.csv')
        
        print('Moving video data...')
        if hasattr(self, '_video_temp_file'):
            shutil.copyfile(self._video_temp_file, 
                            directory + video_name + self._media_extension)
            
        if data_df_csv != None:
            self.data.to_csv(directory + data_df_csv)
            
        print('Done')
        
    def _load_processed_data(self,
                             directory = './',
                             dnn_df_csv = '__temp__.df_dnn.csv',
                             video_name = None,  # Only if _video_temp_file is defined
                             data_df_csv = None,
                             ):
        """
        A debug function for reloading data outputs saved prior
        """
        
        self.df_dnn = pd.read_csv(directory + dnn_df_csv)
        
        if video_name != None:
            self._video_temp_file = directory + video_name
        
        if data_df_csv != None:
            self.data = pd.read_csv(directory + data_df_csv)
        
        pass
    
    def export_csv(self):
        assert self.csv_output_path != None, \
            'Please specify csv_output_path attribute or at init.'
        
        self.data.write_csv(self.csv_output_path)
        
    def export_web(self):
        raise NotImplementedError()
        
    
if __name__ == '__main__':
    
    logs = '../../../drone_Xavier_log_27.01.2021.txt'
    
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
    
    pro.process()
