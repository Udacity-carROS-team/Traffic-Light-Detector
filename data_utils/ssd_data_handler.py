#!/usr/bin/env python


import glob
import os
from shutil import copyfile
import pandas as pd
import xml.etree.ElementTree as ET
"""
This module is used to divide human labeled data to training and test sets.
Before this module is used you should have xml files with the same name as the
image files. They should be in different folders under the project directory.
Corresponding images should be copied from image folder to xml folder.
Then they should be split to training and test folders. After that they should
be converted to the format that ssd model can recognize.
"""


def collect_image(xml_folder, image_folder, format='jpg'):
    """
    copy corresponding images to xml folder
    """
    xml_fns = []

    file_pattern = '../' + xml_folder + '/*.xml'
    xml_fns.extend(glob.glob(file_pattern))

    image_fns = [os.path.basename(fn)[:-3] + format for fn in xml_fns]

    src = '../' + image_folder + '/'
    dst = '../' + xml_folder + '/'

    for image_fn in image_fns:
        copyfile(src + image_fn, dst + image_fn)

    return image_fns


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def save_xml_to_csv(src, dst):
    """
    Input
    ------
    src: path to xml files, it should be a folder
    dst: path to csv file to be stored, it should be a file name
    """
    xml_df = xml_to_csv(src)
    xml_df.to_csv(dst, index=None)
    print('Successfully converted xml to csv.')


def split_data(ratio=0.1):
    pass


if __name__ == '__main__':
    # image_fns = collect_image('labeled_site_images', 'site_images')
