#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:03:30 2019

@author: amirouyed
"""

import do_everything as de
import argparse
from datetime import datetime



class ProcessParameters:
     def __init__(self, **kwargs):
         prop_defaults={
                 "do_builder": False,
                 "do_analysis": True,
                 "do_optical_flow": False,
                 "do_downloader": False,
                 "do_processor": True
                 }
         for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))

def valid_date(d):
    try:
        return datetime.strptime(d, "%Y-%m-%d-%H:%M")
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(d)
        raise argparse.ArgumentTypeError(msg)
        

def parser(process_parameters,parameters):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cutoffs", help="Input list of speed VD cutoffs.", type=float, nargs='+')
    parser.add_argument("-v", "--variable", help="Choose GEOS-5 variable to analyze", type= str)
    parser.add_argument("-b", "--builder", help="Whether we build the dataframe", action ="store_true")
    parser.add_argument("-a", "--analysis", help="Whether we analyze the dataframe", action ="store_true")
    parser.add_argument("-of", "--optical_flow", help="Whether we run optical flow algorithm", action ="store_true")
    parser.add_argument("-d", "--downloader", help="Whether we run downloader", action ="store_true")
    parser.add_argument("-p", "--processor", help="Whether we run processor", action ="store_true")
    parser.add_argument("-sd", "--start_date", help="The Start Date - format YYYY-MM-DD-00:00", 
                        type=valid_date)
    parser.add_argument("-ed", "--end_date", help="The End Date - format YYYY-MM-DD-00:00", 
                        type=valid_date)
    args=parser.parse_args()
    
    if args.cutoffs is not None:
        parameters.cutoffs=args.cutoffs
        print(parameters.cutoffs)
    if args.variable is not None:
        parameters.var=args.variable
    process_parameters.do_builder=args.builder
    process_parameters.do_analysis=args.analysis
    process_parameters.do_optical_flow=args.optical_flow
    process_parameters.do_downloader=args.downloader
    process_parameters.do_processor=args.processor
    if args.start_date is not None:
        parameters.start_date=args.start_date
    if args.end_date is not None:
        parameters.end_date=args.end_date
        

def main():
    parameters=de.Parameters()
    process_parameters=ProcessParameters()
    
    parser(process_parameters,parameters)
    
    if process_parameters.do_downloader:
        de.downloader(parameters)
    if process_parameters.do_processor:
        de.processor(parameters, process_parameters)
    print('All tasks completed.')

if __name__ == '__main__':
   main()