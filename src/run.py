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
    parser.add_argument(
        "-c", "--cutoffs", help="Input list of speed VD cutoffs.", type=float, nargs='+')
    parser.add_argument(
        "-t", "--target_box", help="Input list of targetboxes.", type=int, nargs='+')
    parser.add_argument("-v", "--variable",
                        help="Choose GEOS-5 variable to analyze", type=str)
    parser.add_argument(
        "-g", "--grid", help="Choose the spatial resolution in degrees", type=float)
    parser.add_argument(
        "-dt", "--timestep", help="Choose the temporal  resolution in seconds", type=float)
    parser.add_argument(
        "-us", "--up_speed", help="upper bound of speed", type=float)
    parser.add_argument(
        "-ls", "--low_speed", help="lower  bound of speed", type=float)
    parser.add_argument(
        "-nc", "--cores", help="Choose the number of cores", type=int)
    parser.add_argument(
        "-np", "--poly_n", help="Number of pixels used for dof algorithm", type=int)
    parser.add_argument(
        "-ws", "--winsize", help="Size of smoothing window for DOF algorithm", type=int)
    parser.add_argument(
        "-psi", "--poly_sigma", help="stdev for polynomial fitting", type=float)
    parser.add_argument(
        "-levs", "--levels", help="pyramid scheme levels for DOF", type=int)
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
    parser.add_argument(
        "-sp", "--sub_pixel", help="Whether we run subpixel in cross correlation", action="store_true")
    parser.add_argument("-sd", "--start_date", help="The Start Date - format YYYY-MM-DD-00:00",
                        type=valid_date)
    parser.add_argument("-ed", "--end_date", help="The End Date - format YYYY-MM-DD-00:00",
                        type=valid_date)
    parser.add_argument("-jl", "--jpl_loader",
                        help="Whether we low jpl dataset", action="store_true")
    parser.add_argument(
        "-tk", "--track", help="Whether we load jpl amvs", action="store_true")
    parser.add_argument(
        "-jd", "--jpl_disk", help="Whether we load jpl disk data", action="store_true")

    parser.add_argument(
        "-sc", "--speed_cutoff", help="Whether we filter wind speeds by magnitude", action="store_true")
    parser.add_argument(
        "-t1", "--tvl1", help="Whether we use tvl1 for optical flow", action="store_true")
    parser.add_argument( "-fb", "--farneback", help="Whether we use Farneback's algorithm for optical flow", action="store_true")

    args = parser.parse_args()

    if args.cutoffs is not None:
        parameters.cutoffs = args.cutoffs
    if args.grid is not None:
        parameters.grid = args.grid
    if args.target_box is not None:
        parameters.target_box = args.target_box
    process_parameters.do_builder = args.builder
    process_parameters.do_analysis = args.analysis
    process_parameters.do_optical_flow = args.optical_flow
    process_parameters.do_cross_correlation = args.cross_correlation
    parameters.do_cross_correlation = args.cross_correlation
    process_parameters.do_downloader = args.downloader
    process_parameters.do_processor = args.processor
    parameters.subpixel = args.sub_pixel
    parameters.track = args.track
    parameters.jpl_loader = args.jpl_loader
    parameters.speed_cutoff = args.speed_cutoff
    parameters.tvl1 = args.tvl1
    parameters.farneback = args.farneback
    parameters.jpl_disk = args.jpl_disk
    if args.start_date is not None:
        parameters.start_date = args.start_date
    if args.end_date is not None:
        parameters.end_date = args.end_date
    if args.variable is not None:
        parameters.var = args.variable
    if args.timestep is not None:
        parameters.dt = args.timestep
    if args.low_speed is not None:
        parameters.low_speed = args.low_speed
    if args.up_speed is not None:
        parameters.up_speed = args.up_speed
    if args.cores is not None:
        parameters.cores = args.cores
    if args.poly_n is not None:
        parameters.poly_n = args.poly_n
    if args.poly_sigma is not None:
        parameters.poly_sigma = args.poly_sigma
    if args.levels  is not None:
        parameters.levels = args.levels
    if args.winsize is not None:
        parameters.winsize = args.winsize


def main():
    parameters = de.Parameters()
    process_parameters = ProcessParameters()

    parser(process_parameters, parameters)

    if process_parameters.do_downloader:
        de.downloader(deepcopy(parameters))
    if process_parameters.do_processor:
        de.processor(deepcopy(parameters), process_parameters)
    print('All tasks completed.')


if __name__ == '__main__':
    main()
