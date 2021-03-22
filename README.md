# Darknet Highlights & Post-Processing - Maui63

The following library processes Maui63's output data by running their darknet model on media files and combining the output with UAV logs.

_____

## Installation

From the root directory, run:

```
pip install .
```

## Usage

The media file(s) are automatically be tagged and objects be added to a 
pandas dataframe.

In the case of a video input, if a directory is specified as an output, 
highlights will also be generated (see examples for highlighter args).

To create a data processing instance and run it:
```python
from darknet_highlights import Maui63DataProcessor 

processor = Maui63DataProcessor(
        uav_logs,       # CSV file for UAV data logs
        media_file,     # video/image file or image folder
        data_file,      # darknet .data file
        config_file,    # darknet .cfg file
        weights_file,   # darknet .weights file
        names_file,     # darknet .names file
        output_path     # ouput file/directory
        )
        
# Run the process routine
processor.process()
```

To export the dataframe to a csv file:
```python
processor.export_csv(csv_output_path)
```

## TODO
- Fix padding for start/end clips when creating highlights.
- Push data to R/Vision
- Speed up opencv processing (currently ~3 fps on a GTX 960m)
- Object Tracking?
