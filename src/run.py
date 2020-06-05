#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:03:30 2019

@author: amirouyed
"""

import do_everything as de
import argparse
from datetime import datetime
from copy import deepcopy


class ProcessParameters:
    def __init__(self, **kwargs):
        prop_defaults = {
            "do_builder": False,
            "do_analysis": True,
            "do_optical_flow": False,
            "do_cross_correlation": False,
            "do_downloader": False,
            "do_processor": True,
            "do_summary": True,
            "do subpixel": False
        }
        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))


def valid_date(d):
    try:
        return datetime.strptime(d, "%Y-%m-%d-%H:%M")
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(d)
        raise argparse.ArgumentTypeError(msg)


def parser(process_parameters, parameters):
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--variable",
                        help="Choose GEOS-5 variable to analyze", type=str)
    parser.add_argument(
        "-g", "--grid", help="Choose the spatial resolution in degrees", type=float)
    parser.add_argument(
        "-dt", "--timestep", help="Choose the temporal  resolution in seconds", type=float)
    parser.add_argument(
        "-cg", "--coarse_grid", help="grid size for coarsening flow", type=float)

    parser.add_argument(
        "-tri", "--triplet", help="triplet date,format YYYY-MM-DD-00:00", type=valid_date)
    parser.add_argument(
        "-b", "--builder", help="Whether we build the dataframe", action="store_true")
    parser.add_argument(
        "-a", "--analysis", help="Whether we analyze the dataframe", action="store_true")
    parser.add_argument("-of", "--optical_flow",
                        help="Whether we run optical flow algorithm", action="store_true")
    parser.add_argument("-cc", "--cross_correlation",
                        help="Whether we run cross_correlation algorithm", action="store_true")
    parser.add_argument("-d", "--downloader",
                        help="Whether we run downloader", action="store_true")
    parser.add_argument("-p", "--processor",
                        help="Whether we run processor", action="store_true")
    parser.add_argument("-jl", "--jpl_loader",
                        help="Whether we low jpl dataset", action="store_true")
    parser.add_argument(
        "-tk", "--track", help="Whether we load jpl amvs", action="store_true")
    parser.add_argument(
        "-df", "--deep_flow", help="Whether we use DeepFlow' algorithm for optical flow", action="store_true")

    args = parser.parse_args()

    if args.grid is not None:
        parameters.grid = args.grid
    process_parameters.do_builder = args.builder
    process_parameters.do_analysis = args.analysis
    process_parameters.do_optical_flow = args.optical_flow
    process_parameters.do_downloader = args.downloader
    process_parameters.do_processor = args.processor
    parameters.track = args.track
    parameters.jpl_loader = args.jpl_loader
    parameters.deep_flow = args.deep_flow
    if args.variable is not None:
        parameters.var = args.variable
    if args.timestep is not None:
        parameters.dt = args.timestep
    if args.triplet is not None:
        parameters.triplet = args.triplet
    if args.coarse_grid is not None:
        parameters.coarse_grid = args.coarse_grid


def main():
    parameters = de.Parameters()
    process_parameters = ProcessParameters()

    parser(process_parameters, parameters)
    de.downloader(deepcopy(parameters))
    de.processor(deepcopy(parameters), process_parameters)
    print('All tasks completed.')


if __name__ == '__main__':
    main()
