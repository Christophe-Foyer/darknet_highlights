from __future__ import annotations
from typing import Union

from moviepy.video.VideoClip import VideoClip
from moviepy.editor import VideoFileClip
import numpy as np

import warnings

class Highlighter:
    
    def __init__(self,
                 video: Union[VideoClip, str],  # Moviepy clip
                 times: list[float],  # video timestamps of points of interest
                 padding: float = 30,  # padding before and after points of interest
                 max_spacing_before_merge: float = 30,  # spacing before clips are merged
                 ):
        
        assert isinstance(video, VideoClip) or isinstance(video, str), \
            "'video' must be of type VideoClip or the path to a video file"
            
        if isinstance(video, VideoClip):
            self.video = video
        elif isinstance(video, str):
            self.video = VideoFileClip(video)
            
        # TODO: Warn if padding andspacing don't make sense
            
        self.times = np.sort(times)  # Make it easier on ourselves
        self.padding = padding
        self.max_spacing_before_merge = max_spacing_before_merge
        
    def merge_points_of_interest(self, max_spacing = None):
        """
        A simple class to merge point of interest clips
        """
        
        if max_spacing == None:
            max_spacing = self.max_spacing_before_merge
        
        highlight_times = []
        
        time_groups = []
        for i, time in enumerate(self.times):
            # If it's outside the padding distance then new group
            if i == 0 or time >= self.times[i-1] + max_spacing:
                time_groups.append([time])
                
            # If within padding distance same group
            else:
                time_groups[-1].append(time)
            
        # Find the clip start_ends
        for group in time_groups:
            
            start = np.min(group) - self.padding
            end = np.max(group) + self.padding
            
            if start <= 0:
                start = 0
            if end >= self.video.duration:
                end = self.video.duration
                
            if start >= self.video.duration or end <= 0:
                warnings.warn('Time group is invalid: ' + str(group))
                
                continue  # Skip this entry
            
            highlight_times.append((start, end))
                
            
        return highlight_times
        
    def create_subclip(self, start, end):
        """
        Take a subclip from the video based on start and end time
        """
        
        clip = self.video.subclip(start, end)
        
        return clip
        

if __name__ == '__main__':
    clip = "../../../testingvideos/mauitest_11_40s_1080.mp4"
    
    times = [1, 3, 15, 50, 52]
    
    hl = Highlighter(clip, times, padding = 3)
    
    video = hl.video
    
    highlights = hl.merge_points_of_interest(5)
    
    clip = hl.create_subclip(*highlights[0])
