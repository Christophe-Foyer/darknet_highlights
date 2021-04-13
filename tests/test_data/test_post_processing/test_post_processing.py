import pytest
from pathlib import Path

from maui63_postprocessing.data.post_process import Maui63DataProcessor 



logs = Path(__file__).parent / 'Drone_Flight_Path_Dummy_Data.csv'

# Testfiles
video = Path(__file__).parent / 'Maui Dolphins Aerial Footage - 2017.mp4'
# image = Path(__file__).parent / 'maui-dolphins-the-gaps-boren-1200x783.jpg'
image = Path(__file__).parent / 'Maui Dolphins Aerial Footage - 2017.png'

# These files not on github repo, please provide your own
data_file = Path(__file__).parent / '../../../../maui_sf_and_100m.data'
config_file = Path(__file__).parent / '../../../../yolov4-tiny-maui-sf-and-100m.cfg'
weights = Path(__file__).parent / '../../../../yolov4-tiny-maui-sf-and-100m_best.weights'
names_file = Path(__file__).parent / '../../../../maui.names'

highlighter_kwargs = {'clip_length': 10, 'padding': 3}


def test_video_init():
    processor = Maui63DataProcessor(
        logs,
        video, 
        data_file, 
        config_file, 
        weights, 
        names_file,
        highlighter_kwargs = highlighter_kwargs
        )

def test_videoprocessing_detection():
    processor = Maui63DataProcessor(
        logs,
        video, 
        data_file, 
        config_file, 
        weights, 
        names_file,
        highlighter_kwargs = highlighter_kwargs
        )
    
    processor._run_cv()
    
    # 50 is arbitrary, but should work for most decent CV code (currently 86)
    assert len(processor.dnn_df) > 50, \
        'Dolphins should be detected in most of the frames'
    
def test_imageprocessing_detection():
    processor = Maui63DataProcessor(
        logs,
        image, 
        data_file, 
        config_file, 
        weights, 
        names_file,
        highlighter_kwargs = highlighter_kwargs,
        output_path = '__temp__.jpg'
        )
    
    processor._run_cv()
    
    # 50 is arbitrary, but should work for most decent CV code (currently 86)
    assert processor.dnn_df.num_objects.to_numpy()[0] >= 1, \
        'Dolphins should be detected in the test image'
    
def test_videoprocessing_process():
    processor = Maui63DataProcessor(
        logs,
        video, 
        data_file, 
        config_file, 
        weights, 
        names_file,
        highlighter_kwargs = highlighter_kwargs,
        output_path = '__temp__'
        )
    
    processor.process()
    
    # 50 is arbitrary, but should work for most decent CV code (currently 86)
    assert len(processor.data) > 50, \
        'Dolphins should be detected in most of the frames'
        
def test_imageprocessing_process():
    processor = Maui63DataProcessor(
        logs,
        image, 
        data_file, 
        config_file, 
        weights, 
        names_file,
        highlighter_kwargs = highlighter_kwargs,
        output_path = '__temp__.jpg'
        )
    
    processor.process()
    
    assert processor.dnn_df.num_objects.to_numpy()[0] >= 1, \
        'Dolphins should be detected in the test image'
    

if __name__ == '__main__':
    pytest.main()