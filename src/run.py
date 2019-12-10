#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:03:30 2019

@author: amirouyed
"""

import do_everything as de
import argparse


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

def parser(process_parameters,parameters):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cutoffs", help="Input list of speed VD cutoffs.", type=float, nargs='+')
    parser.add_argument("-v", "--variable", help="Choose GEOS-5 variable to analyze", type= str)

    parser.add_argument("-b", "--builder", help="Whether we build the dataframe", type= bool)
    parser.add_argument("-a", "--analysis", help="Whether we analyze the dataframe", type= bool)
    parser.add_argument("-of", "--optical_flow", help="Whether we run optical flow algorithm", type= bool)
    parser.add_argument("-d", "--downloader", help="Whether we run downloader", type= bool)
    parser.add_argument("-p", "--processor", help="Whether we run processor", type= bool)


    args=parser.parse_args()
    if args.cutoffs is not None:
        parameters.cutoffs=args.cutoffs
        print(parameters.cutoffs)
    if args.variable is not None:
        parameters.var=args.variable
        print(parameters.var)
    if args.builder is not None:
        process_parameters.do_builder=args.builder
    if args.analysis is not None:
        process_parameters.do_analysis=args.analysis
    if args.optical_flow is not None:
        process_parameters.do_optical_flow=args.optical_flow
    if args.downloader is not None:
        process_parameters.do_downloader=args.downloader
    if args.processor is not None:
        process_parameters.do_processor=args.processor
        

def main():
    parameters=de.Parameters()
    process_parameters=ProcessParameters()
    
    parser(process_parameters,parameters)
    
    if process_parameters.do_downloader:
        de.downloader(parameters)
    if process_parameters.do_processor:
        de.processor(parameters, process_parameters)
    print('Done_final')

if __name__ == '__main__':
   main()