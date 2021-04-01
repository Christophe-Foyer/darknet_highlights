# Darknet Highlights & Post-Processing - Maui63

The following library processes [Maui63](http://maui63.org/)'s output data by running their darknet model on media files and combining the output with UAV logs.

_____

## Installation

From the root directory, run:

```
pip install .
```

Note: The example and test files assume the darknet files are in the repository's parent directory.

## Usage

The media file(s) are automatically tagged and objects added to a 
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

To export the data to [rvision](https://rvision.rush.co.nz/):
```python
# export data with a minimum spacing of 30s between frames
processor.export_rvision(camera_token, min_spacing=30)
```
*Note: the script posts to "https://be.uat.rvision.rush.co.nz/api/v1/alpr/camera/<camera_token>", if this is not the correct url either contact me to change it in the source code or simply change the url:
```python
processor.rvision_url = "https://domain.name/path/to/ressource/<camera_token>"
```

## TODO
- Fix padding for start/end clips when creating highlights.
- Push data to R/Vision
- Speed up opencv processing (currently ~3 fps on a GTX 960m)
- Object Tracking?
- Upload video clips to rvision (waiting for API)
