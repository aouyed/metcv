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


def valid_date(d):
    try:
        return datetime.strptime(d, "%Y-%m-%d-%H:%M")
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(d)
        raise argparse.ArgumentTypeError(msg)


def parser(parameters):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dt", "--timestep", help="Choose the temporal  resolution in seconds", type=float)
    parser.add_argument(
        "-tri", "--triplet", help="triplet date,format YYYY-MM-DD-00:00", type=valid_date)
    parser.add_argument(
        "-g", "--grid", help="Choose the spatial resolution in degrees", type=float)
    parser.add_argument(
        "-p", "--pressure", help="Type the pressure level in hPa", type=float)
    args = parser.parse_args()

    if args.timestep is not None:
        parameters.dt = args.timestep
    if args.triplet is not None:
        parameters.triplet = args.triplet
    if args.grid is not None:
        parameters.grid = args.grid
    if args.pressure is not None:
        parameters.pressure = args.pressure


def main():
    parameters = de.Parameters()

    parser(parameters)
    de.downloader(deepcopy(parameters))
    de.processor(deepcopy(parameters))
    print('All tasks completed.')


if __name__ == '__main__':
    main()
