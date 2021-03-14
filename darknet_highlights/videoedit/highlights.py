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
                 padding: float = 10,  # padding before and after points of interest
                 max_spacing_before_merge: float = None,  # spacing between events before clips are merged
                 clip_length: float = 30,  # Clip length excluding padding
                 ):
        
        assert isinstance(video, VideoClip) or isinstance(video, str), \
            "'video' must be of type VideoClip or the path to a video file"
            
        if isinstance(video, VideoClip):
            self.video = video
        elif isinstance(video, str):
            self.video = VideoFileClip(video)
            
        self.times = np.sort(times)  # Make it easier on ourselves, sort it
        
        self.padding = padding
        
        if max_spacing_before_merge == None:
            self.max_spacing_before_merge = padding
        else:
            self.max_spacing_before_merge = max_spacing_before_merge
            
        assert clip_length > padding * 2, \
            "Padding is too large for clip length (padding < clip_length/2)"
        self.clip_length = clip_length
        
    def merge_points_of_interest(self, max_spacing = None):
        """
        A simple class to merge point of interest clips
        """
        
        # TODO: Add max bin length
        
        if max_spacing == None:
            max_spacing = self.max_spacing_before_merge
        
        highlight_times = []
        
        time_groups = []
        for i, time in enumerate(self.times):
            # If it's outside the padding distance then new group
            # Also check if the clip has gotten too long
            if i == 0 or time >= self.times[i-1] + max_spacing \
                or (len(time_groups) > 0 and
                    time - time_groups[-1][0] >= 
                    self.clip_length):
                    
                time_groups.append([time])
                
            # If within padding distance same group
            else:
                time_groups[-1].append(time)
            
        # Find the clip start_ends
        # TODO: Fix first clip being short?
        for group in time_groups:
            
            start = np.min(group) - self.padding
            end = np.max(group) + self.padding
            
            if start <= 0:
                start = 0
            if end >= self.video.duration:
                end = self.video.duration
                
            if start >= end:
                warnings.warn('Time group is invalid: ' + str(group))
                continue  # Skip this entry
            
            highlight_times.append((start, end))
                
            
        return highlight_times, time_groups
        
    def create_subclip(self, start, end):
        """
        Take a subclip from the video based on start and end time
        """
        
        clip = self.video.subclip(start, end)
        
        return clip
    
    def create_clips(self):
        
        highlights, groups = self.merge_points_of_interest()
    
        clips = []
        for highlight in highlights:
            clip = self.create_subclip(*highlight)
            clips.append(clip)
            
        self.clips = clips
        self.groups = groups
        
        return clips, groups


if __name__ == '__main__':
    
    from darknet_highlights.cv import process_video 
    
    video = '../../../testingvideos/mauitest_11_40s_1080.mp4'
    data_file = '../../../maui_sf_and_100m.data'
    config_file = '../../../yolov4-tiny-maui-sf-and-100m.cfg'
    weights = '../../../yolov4-tiny-maui-sf-and-100m_best.weights'
    names_file = '../../../maui.names'
    
    output_file = "__temp__.avi"
    
    df = process_video(video, data_file, config_file, weights, names_file,
                       output_file = output_file)
    
    times = df.timestamp
    
    hl = Highlighter(output_file, times)
    
    video = hl.video
    
    highlights, groups = hl.merge_points_of_interest()
    
    clips, groups = hl.create_clips()
        
    # clip.
