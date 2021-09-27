import xml.etree.ElementTree as ET
from math import ceil

import numpy as np


def parse(input_file, regions, vertices):
    tree = ET.parse(input_file)
    root = tree.getroot()
    images = {}
    for region in root.iter(regions):
        coordinates = []
        for vertex in region.iter(vertices):
            temp = [int(i) for i in vertex.attrib.values()]
            coordinates.append(temp)
        images.update({region.attrib["Id"]: np.array(coordinates)})

    return images


def annotation_conversion(slide_dimensions, coordinates, annotated_level, required_level):
    y_axis = slide_dimensions[required_level][1] / slide_dimensions[required_level][1]
    x_axis = slide_dimensions[required_level][0] / slide_dimensions[required_level][0]
    new_coordinate_list = []

    for j in coordinates.values():
        coordinates_list = j
        new_coordinate_list_temp = []
        for k in range(len(coordinates_list)):
            new_coordinate_list_temp.append(
                (ceil(coordinates_list[k][0] * x_axis), ceil(coordinates_list[k][1] * y_axis)))
        new_coordinate_list.append(new_coordinate_list_temp)

    return new_coordinate_list


def eudlieandistance(A, B):
    d1 = (A[0] - B[0]) ** 2
    d2 = (A[1] - B[1]) ** 2
    d = np.sqrt(d1 + d2)
    return d


def polygon_or_cross(coordinate_list):
    dist1 = eudlieandistance(coordinate_list[0], coordinate_list[-1])
    dist2 = eudlieandistance(coordinate_list[int(len(coordinate_list) / 2)], coordinate_list[0])

    if dist2 > dist1:
        return "polygen"
    else:
        return "cross"


def extracting_roi_annotations(input_file, slide_dims, annotated_level, required_level):
    coordinates = parse(input_file, regions='Region', vertices='Vertex')
    new_coordinate_list = annotation_conversion(slide_dims, coordinates, annotated_level, required_level)
    coordinate_filter_list = []
    for i in range(len(new_coordinate_list)):
        if polygon_or_cross(new_coordinate_list[i]) == "polygen":
            coordinate_filter_list.append(new_coordinate_list[i])

    return coordinate_filter_list
