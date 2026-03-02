<!--
SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>

SPDX-License-Identifier: Apache-2.0
-->

# Mass Spectrometry NeXus Tools
This repo is hope to a collection of tools for converting mass spectrometry data into the [NeXus][https://www.nexusformat.org/] format.
In addition there are tools to transform the data.

## NeXus details
Nexus is a general metadata structure, and so can host any data shape.
For the RFI we will be storing our data as one large 4 dimensional block. 
So far the four dimensions are layers, image_width, image_height, and spectrum.

## Ionoptika 
The Ionoptika software has the option to export files to the h5 file format. 
This is an improvement of closed or proprietary software formats.
The data is stored twice: 
Once by spectrum, listing every layer and pixel.
Once by image, listing every layer and bin of the spectrum.
Both of these are slow to process, but especially the second one. 
The _ion2rfi_ tool can be used to transform these h5 files into a simpler, smaller and more performant structure.
Both sets of data can be transformed. However it is recommended that the "mass images" not be used for large files as it is spectacularly slow.

