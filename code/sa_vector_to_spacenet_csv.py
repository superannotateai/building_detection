# -*- coding: utf-8 -*-
"""
Read Superannotate json annotation files in vector format and create a csv file in spacenet format, or vice versa.

Sample usage: 
    python3 sa_vector_to_spacenet_csv.py sa-vectors-folder-to-spacenet-csv /data/testset_annotations predictions.csv
    python3 sa_vector_to_spacenet_csv.py spacenet-csv-to-sa-vectors-folder predictions.csv /data/testset_annotations 

After getting the csv file compare it to model outputs to get f1 score:
    java -jar /root/visualizer-2.0/visualizer.jar -truth /data/true_annotated_testset.csv -solution /data/working/models/v17/AOI_2_Vegas_poly.csv -no-gui -band-triplets /root/visualizer-2.0/data/band-triplets.txt -image-dir pass

Author: martun
"""

from shapely import wkt
from shapely.geometry import Polygon # For writing polygons in WKT format
import numpy as np
import json
import csv
from os import listdir
from os.path import isfile, join
import warnings
from logging import getLogger, Formatter, StreamHandler, INFO, FileHandler
import click

# Logger
warnings.simplefilter("ignore", UserWarning)
handler = StreamHandler()
handler.setLevel(INFO)
handler.setFormatter(Formatter('%(asctime)s %(levelname)s %(message)s'))

logger = getLogger('Convert_to_spacenet_csv')
logger.setLevel(INFO)


@click.group()
def cli():
    pass


# Gets filenames like this: RGB-PanSharpen_AOI_2_Vegas_img108_8_bit.tif___objects.json, 
# must return AOI_2_Vegas_img108
def image_id_from_filename(filename):
    result = filename.replace('RGB-PanSharpen_', '')
    result = result.replace('_8_bit.tif___objects.json', '')
    return result
 

def to_sa_polygon(polygon_wkt_string):
    """ Converts wkt polygon to SA polygon format (list of x, y coordinates)
    """
    shape_obj = wkt.loads(polygon_wkt_string)
    coords = list(shape_obj.exterior.coords) 
    x = [round(float(pp[0])) for pp in coords]
    y = [round(float(pp[1])) for pp in coords]
    result = [None] * len(x) * 2
    result[0::2] = x
    result[1::2] = y
    return result
    

def to_3d_points(points_array):
    result = []
    # Z coordinate must always be 0.
    for x,y in zip(points_array[0::2], points_array[1::2]):
        result.append((x, y, 0))
    return result


@cli.command()
@click.argument('input_dir_path', type=str)
@click.argument('output_csv_path', type=str)
def sa_vectors_folder_to_spacenet_csv(input_dir_path, output_csv_path):
    annotation_files = [f for f in listdir(input_dir_path) if isfile(join(input_dir_path, f))]    
    f = open(output_csv_path, 'w')
    with f:
        writer = csv.DictWriter(
            f, fieldnames=['ImageId', 'BuildingId', 'PolygonWKT_Pix', 'Confidence'])
        writer.writeheader()
        for annotation_file in annotation_files:
            ImageId = image_id_from_filename(annotation_file)
            logger.info("Loading file {}".format(input_dir_path + '/' + annotation_file))
            sa_ann_json = json.load(open(input_dir_path + '/' + annotation_file))
            for idx, annotation in enumerate(sa_ann_json):
                if annotation['type'] != 'polygon':
                    continue
                points_3d = to_3d_points(annotation['points'])
                polygon_str = wkt.dumps(Polygon(points_3d))
                row = {'ImageId':ImageId, 'BuildingId': idx + 1, 
                       'PolygonWKT_Pix': polygon_str, 'Confidence': '1.000000'}
                writer.writerow(row);


def parse_spacenet_csv_file(input_csv_path):
    # Read csv file with polygons.
    image_id_to_polygons = {}
    with open(input_csv_path, newline='') as input_csv_file:
        reader = csv.DictReader(input_csv_file)
        for row in reader:
            if not row['ImageId'] in image_id_to_polygons:
                image_id_to_polygons[row['ImageId']] = []
            # If this is a ground truth file, there is no confidence field, if it's generated
            # by a classifier, there is one.
            confidence = 1
            if 'Confidence' in row:
                confidence = row['Confidence']
            image_id_to_polygons[row['ImageId']].append(
                (confidence, to_sa_polygon(row['PolygonWKT_Pix'])))
    for image_id, polygons in image_id_to_polygons.items():
        yield image_id, polygons


def polygons_to_annotation(polygons):
    annotations = []
    for polygon in polygons:
        annotation = {}
        annotation['type'] = 'polygon'
        annotation['classId'] = 3990 # This means "building"
        annotation['probability'] = int(float(polygon[0]) * 100)
        annotation['points'] = polygon[1]
        annotations.append(annotation)
    return annotations


def spacenet_csv_to_sa_vector_file(input_csv_path, output_json_path):
    for image_id, polygons in parse_spacenet_csv_file(input_csv_path):
        with open(output_json_path, 'w') as outfile:
            json.dump(polygons_to_annotation(polygons), outfile)


@cli.command()
@click.argument('input_csv_path', type=str)
@click.argument('output_dir_path', type=str)
def spacenet_csv_to_sa_vectors_folder(input_csv_path, output_dir_path):
    # Write a bunch of json files in SA format. 
    for image_id, polygons in parse_spacenet_csv_file(input_csv_path):
        json_path = "{}/RGB-PanSharpen_{}_8_bit.jpg___objects.json".format(output_dir_path, image_id)
        with open(json_path, 'w') as outfile:
            json.dump(annotations, outfile)

    # Write the classes.json file, this is a fixed file for now.
    classes_file = open(output_dir_path + "/classes.json", "w")
    classes_file.write('[{"id":3990,"project_id":953,"name":"Buildings","color":"#62c2b2","createdAt":"2020-03-13T09:27:10.000Z","updatedAt":"2020-03-13T09:27:26.000Z","attribute_groups":[]},{"id":4381,"project_id":953,"name":"Drawing","color":"#e30101","createdAt":"2020-03-19T22:54:39.000Z","updatedAt":"2020-03-19T22:54:58.000Z","attribute_groups":[]}]')
    classes_file.close()


if __name__ == '__main__':
    logger.addHandler(handler)
    cli()

