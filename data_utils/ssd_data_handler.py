#!/usr/bin/env python


import glob
import os
import errno
from shutil import copyfile
import pandas as pd
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
"""
This module is used to divide human labeled data to training and test sets.
Before this module is used you should have xml files with the same name as the
image files. They should be in different folders under the project directory.
Corresponding images should be copied from image folder to xml folder.
Then they should be split to training and test folders. After that they should
be converted to the format that ssd model can recognize.
"""


def image_names(xml_path):
    xml_fns = []

    file_pattern = xml_path + '/*.xml'
    xml_fns.extend(glob.glob(file_pattern))

    return xml_fns


def collect_image(xml_fns, dst, image_path, format='jpg'):
    """
    copy xml files and corresponding images to targeted folder

    Input
    ------
    xml_fns: a list of xml file names
    dst: absolute path that the xml and image files are stored to
    image_path: absolute path or relative path to project directory
    format: image format
    """
    try:
        os.makedirs(dst)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    for fn in xml_fns:
        base_xml_fn = os.path.basename(fn)
        base_img_fn = base_xml_fn[:-3] + format
        copyfile(fn, dst + '/' + base_xml_fn)
        copyfile(image_path + '/' + base_img_fn, dst + '/' + base_img_fn)


def split_data(xml_fns, ratio=.1):
    """
    split data to train and test
    """
    shuffle(xml_fns)
    train, test = train_test_split(xml_fns, test_size=ratio)
    return train, test


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


if __name__ == '__main__':
    print('say hello to ssd_data_handler.py')
