# Darknet Highlights & Post-Processing - Maui63

To create a data processing instance and run it:
```python
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

The media file(s) will automatically be tagged and objects be added to a 
pandas dataframe.

In the case of a video input, if a directory is specified as an output, 
highlights will also be generated.

To export the dataframe to a csv file:
```python
processor.export_csv(csv_output_path)
```