from __future__ import annotations
from typing import Union

import time
import pandas as pd
from moviepy.video.VideoClip import VideoClip
from moviepy.editor import VideoFileClip
import datetime
from tqdm import tqdm


class Maui63UAVImporter:
    
    def __init__(self,
                 logfile: str = None, 
                 # delimiter: str = ',',
                 # cv_dateformat: str = "%Y.%m.%d %H.%M.%S ",
                 cv_logfile: str = None,
                 ):
        
        if logfile != None:
            print('Importing logs...')
            self.df = pd.read_csv(logfile)
        
        # if we'd like the cv logs imported (not sure why but I've coded it already)
        if cv_logfile != None:
            self.cv_df = self._import_cv_logs(cv_logfile)
        
        
    def _import_cv_logs(self, logfile):
        
        print('Importing cv logs:')
        time.sleep(0.5)
        
        lines = []
        f = open(logfile, "r")
        while(True):
        	#read next line
        	line = f.readline()
        	#check if line is not null
        	if not line:
        		break
        	#you can access the line
        	lines.append(line.strip())
            
        columns = ['datetime_utc', 'num', 'prob', 'lat', 'lon']
        df = pd.DataFrame(columns = columns)
        
        startidx = 2  # start at prob
        numpara = 3  # number of parameters per object
        
        for line in tqdm(lines):
            values = [x for x in line.split(',') if x != '']
            
            out = [None] * len(columns)
            
            out[0] = datetime.datetime.strptime(values[0].strip(), "%Y.%m.%d %H.%M.%S")
            out[0] = out[0].replace(tzinfo=datetime.timezone.utc).timestamp()
            
            num = [x.strip() for x in values[1].split('=')]
            assert num[0].lower() == 'num'
            out[1] = int(num[1])
            
            for j in range(numpara):
                out[startidx+j] = []
            
            for i in range(out[1]):
                
                for j in range(numpara):
                    para = values[startidx + i * numpara + j]
                    para = [x.strip() for x in para.split('=')]
                    
                    assert para[0] == columns[startidx+j]
                    
                    try:
                        out[startidx+j].append(float(para[1]))
                    except ValueError:
                        out[startidx+j].append(para[1])
                    
            # append to dataframe
            series = pd.Series(dict(zip(df.columns, out)))
            df = df.append(series, ignore_index=True)
            
        return df

class Maui63UAVVideoImporter(Maui63UAVImporter):
    
    def __init__(self,
                 video: Union[VideoClip, str],  # Moviepy clip
                 logfile: str, 
                 **kwargs,
                 ):
        
        assert isinstance(video, VideoClip) or isinstance(video, str), \
            "'video' must be of type VideoClip or the path to a video file"
            
        if isinstance(video, VideoClip):
            self.video = video
        elif isinstance(video, str):
            self.video = VideoFileClip(video)
            
        super().__init__(logfile, **kwargs,)
        
        
if __name__ == '__main__':
    log = '../../../drone_Xavier_log_27.01.2021.txt'
    
    importer = Maui63UAVImporter()