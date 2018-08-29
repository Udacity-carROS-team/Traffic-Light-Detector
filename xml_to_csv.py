#!/usr/bin/env python
import data_utils.ssd_data_handler as data_handler

if __name__ == '__main__':
    xml_file_names = data_handler.image_names('./labeled_site_images')
    train, test = data_handler.split_data(xml_file_names)
    data_handler.collect_image(train, './data/train', './site_images')
    data_handler.collect_image(test, './data/test', './site_images')
    data_handler.save_xml_to_csv('./data/train', './ssd_train.csv')
    data_handler.save_xml_to_csv('./data/test', './ssd_test.csv')
