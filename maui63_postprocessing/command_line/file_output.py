from maui63_postprocessing.data.post_process import Maui63DataProcessor
import argparse

def main(*args):
    
    # Defaults
    logs = 'logs.csv'
    media = 'video.mp4'
    data_file = 'data.data'
    config_file = 'config.cfg'
    weights = 'weights.weights'
    names_file = 'names.names'
    highlighter_kwargs = {'clip_length': 10, 'padding': 3}
    output_path = '__temp__.output'
    csv_output = '__temp__.output.csv'
    
    # Parser
    parser = argparse.ArgumentParser(description=
        "A post-processing tool for Maui63 logs. Outputs to local files."
        )
    
    parser.add_argument("logfile",
                        type=str,
                        default = logs,
                        help="Maui63 Logfile (CSV)")
    
    parser.add_argument("media",
                        type=str,
                        default = media,
                        help="Media input file / directory")
    
    parser.add_argument("outputpath",
                        type=str,
                        default = output_path,
                        help="Media output file / directory")
    
    parser.add_argument("csvoutputpath",
                        type=str,
                        default = csv_output,
                        help="CSV output file / directory")
    
    parser.add_argument("--datafile",
                        type=str,
                        default = data_file,
                        help="Maui63 darknet data file.")
    
    parser.add_argument("--configfile",
                        type=str,
                        default = config_file,
                        help="Maui63 darknet config file.")
    
    parser.add_argument("--namesfile",
                        type=str,
                        default = names_file,
                        help="Maui63 darknet names file.")
    
    parser.add_argument("--weightsfile",
                        type=str,
                        default = weights,
                        help="Maui63 darknet weights file.")
    
    parser.add_argument('--cliplength', metavar='l', 
                        type=float, default=10,
                        help='Highlight clip length')
    
    parser.add_argument('--padding', metavar='p', 
                        type=float, default=3,
                        help="Highlight padding")
    
    pro = Maui63DataProcessor(
        args.logfile,
        args.media, 
        args.datafile, 
        args.configfile, 
        args.weightsfile, 
        args.namesfile,
        output_path = args.outputpath,
        csv_output_path=args.csvoutputpath,
        highlighter_kwargs = {
                'clip_length': args.cliplength,
                'padding': args.padding
            }
        )
    
    pro.process()
    pro.export_csv()