import traceback

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver import Firefox
from selenium.webdriver import PhantomJS
from selenium.webdriver.firefox.options import Options

from PIL import Image
from io import BytesIO
import warnings
import json
import time

from bs4 import BeautifulSoup

def warn(*args, **kwargs):
    pass

warnings.warn = warn

def html_to_img(driver,html_content,id_count):
    '''Converts html to image.

    Args:
        driver: Selenium web driver
        html_content: String containing html code for table
        id_count: number of text entities in table

    Returns:
        im: cropped table image
        bboxes: list of lists containing the text, text length and bbox positions for each table entity
    '''
    counter=1                #This counter is to keep track of the exceptions and stop execution after 10 exceptions have occurred
    while(True):
        try:
            driver.get("data:text/html;charset=utf-8," + html_content)
            driver.set_window_size(3000,3000)
            window_size=driver.get_window_size()
            max_height,max_width=window_size['height'],window_size['width']
            element = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.ID, '0')))

            bboxes=[]
            for id in range(id_count):
                #e = driver.find_element_by_id(str(id))
                e = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.ID, str(id))))
                txt=e.text.strip()
                lentext=len(txt)
                loc = e.location
                size_ = e.size
                xmin = loc['x']
                ymin = loc['y']
                xmax = int(size_['width'] + xmin)
                ymax = int(size_['height'] + ymin)
                bboxes.append([lentext,txt,xmin,ymin,xmax,ymax])
                # cv2.rectangle(im,(xmin,ymin),(xmax,ymax),(0,0,255),2)

            png = driver.get_screenshot_as_png()
            im = Image.open(BytesIO(png))
            im = im.crop((0,0, max_width, max_height))

            return im,bboxes

        except Exception as e:
            counter+=1
            if(counter==10):
                raise e

            continue

def html_to_csv(html_content):
    '''Converts HTML file to .csv.

    Args:
        html_content: string containing contents of .html files.

    Returns:
        output_rows: list of lists containing contents of each row.
    '''

    soup = BeautifulSoup(html_content)
    table = soup.find("table")
    output_rows = []

    for table_row in table.findAll('tr'):
        headers = table_row.findAll('th')
        columns = table_row.findAll('td')
        output_row = []
        for header in headers:
            output_row.append(header.text)
        for column in columns:
            output_row.append(column.text)
        output_rows.append(output_row)

    return(output_rows)
