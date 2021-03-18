from pathlib import Path
from darknet_highlights.data.post_process import Maui63DataProcessor 

logs = Path(__file__).parent / '../tests/test_data/test_post_processing/Drone_Flight_Path_Dummy_Data.csv'

# %% File paths

# Media
video = Path(__file__).parent / '../../testingvideos/mauitest_11_40s_1080.mp4'

# These files not on github repo, please provide your own
data_file = Path(__file__).parent / '../../maui_sf_and_100m.data'
config_file = Path(__file__).parent / '../../yolov4-tiny-maui-sf-and-100m.cfg'
weights = Path(__file__).parent / '../../yolov4-tiny-maui-sf-and-100m_best.weights'
names_file = Path(__file__).parent / '../../maui.names'

# Give the option to change the inputs
logs = str(input("Log file path (leave empty for default path): ") or logs)
media = str(input("Media file path (leave empty for default path): ") or video)
data_file = str(input("Data file path (leave empty for default path): ") or data_file)
config_file = str(input("Cfg file path (leave empty for default path): ") or config_file)
weights = str(input("Weights file path (leave empty for default path): ") or weights)
names_file = str(input("Names file path (leave empty for default path): ") or names_file)
output_path = str(input("Output path (leave empty for default path): ") or '__temp__')
csv_output_path = str(input("CSV output path (leave empty for default path): ") or '__temp__.csv')

# Make sure there are inputs
assert logs; assert video; assert data_file; assert config_file
assert weights; assert names_file; assert output_path

# %% Post-processing
highlighter_kwargs = {'clip_length': 10, 'padding': 3}

processor = Maui63DataProcessor(
        logs,
        media, 
        data_file, 
        config_file, 
        weights, 
        names_file,
        highlighter_kwargs = highlighter_kwargs
        )

processor.process()
processor.export_csv(csv_output_path)