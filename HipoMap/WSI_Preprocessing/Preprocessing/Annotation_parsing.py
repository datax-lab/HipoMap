import xml.etree.ElementTree as ET
from math import ceil
import random

import numpy as np
def parse(inputfile, regions, vertices):
    tree = ET.parse(inputfile)
    root = tree.getroot()
    images = {}
    for region in root.iter(regions):
        coordinates = []
        for vertex in region.iter(vertices):
            temp = [int(i) for i in vertex.attrib.values()]
            coordinates.append(temp)
        images.update({region.attrib["Id"]: np.array(coordinates)})

    return images


def annotation_conversion(slide_dimensions, coordinates,annotatedlevel,reqiuredlevel):
    y_axis = slide_dimensions[reqiuredlevel][1] / slide_dimensions[reqiuredlevel][1]
    x_axis = slide_dimensions[reqiuredlevel][0] / slide_dimensions[reqiuredlevel][0]
    new_cordinate_list = []
    new_cordinate_list_temp = []
    for j in coordinates.values():
        coordinates_list = j
        new_cordinate_list_temp = []
        for k in range(len(coordinates_list)):
            new_cordinate_list_temp.append(
                (ceil(coordinates_list[k][0] * x_axis), ceil(coordinates_list[k][1] * y_axis)))
        new_cordinate_list.append(new_cordinate_list_temp)
    return new_cordinate_list

def eudlieandistance(A,B):
    d1 = (A[0]-B[0])**2
    d2 = (A[1]-B[1])**2
    d = np.sqrt(d1+d2)
    return d

def polygon_or_cross(cordinate_list):
    dist1 = eudlieandistance(cordinate_list[0],cordinate_list[-1])
    dist2 = eudlieandistance(cordinate_list[int(len(cordinate_list)/2)],cordinate_list[0])
    if dist2>dist1:
        return "polygen"
    else:
        return "cross"

def extracting_roi_annotations(inputfile,slide_dims,annotatedlevel,reqiuredlevel):
    coordinates = parse(inputfile, regions='Region', vertices='Vertex')
    new_cordinate_list = annotation_conversion(slide_dims, coordinates,annotatedlevel,reqiuredlevel)
    coordinate_filter_list = []
    for i in range(len(new_cordinate_list)):
        if polygon_or_cross(new_cordinate_list[i]) == "polygen":
            coordinate_filter_list.append(new_cordinate_list[i])
        else:
            None
#     print(coordinate_filter_list)
    return coordinate_filter_list
