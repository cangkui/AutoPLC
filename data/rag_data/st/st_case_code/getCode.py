import xml.etree.ElementTree as ET
import os
def read_pou_names_from_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    filecount = 0

    pous = []

    for pou in root.findall('pous/pou'):
        name = pou.get('name') 
        content = ET.tostring(pou, encoding='unicode')
        print(content)
        pous.append(content)
        filecount += 1

    return pous, filecount          


file_path = 'oscat.xml'
pou_names, filecount = read_pou_names_from_xml(file_path)
print(filecount)