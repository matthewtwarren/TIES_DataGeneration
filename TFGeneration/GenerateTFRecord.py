import warnings
warnings.filterwarnings("ignore")

#import tensorflow as tf
import numpy as np
import traceback
import cv2
import os
import string
import pickle
from multiprocessing import Process,Lock
from TableGeneration.Table import Table
from multiprocessing import Process,Pool,cpu_count
import random
import argparse
from TableGeneration.tools import *
import numpy as np
from selenium.webdriver import Firefox
from selenium.webdriver import PhantomJS
import warnings
from TableGeneration.Transformation import *

def warn(*args,**kwargs):
    pass

class Logger:
    def __init__(self):
        pass
        #self.file=open('logtxt.txt','a+')

    def write(self,txt):
        file = open('logfile.txt', 'a+')
        file.write(txt)
        file.close()

class GenerateTFRecord:
    def __init__(self, outpath,filesize,unlvimagespath,unlvocrpath,unlvtablepath,visualizeimgs,visualizebboxes,distributionfilepath):
        self.outtfpath = outpath                        #directory to store tfrecords
        self.filesize=filesize                          #number of images in each tfrecord
        self.unlvocrpath=unlvocrpath                    #unlv ocr ground truth files
        self.unlvimagespath=unlvimagespath              #unlv images
        self.unlvtablepath=unlvtablepath                #unlv ground truth of tabls
        self.visualizeimgs=visualizeimgs                #wheter to store images separately or not
        self.distributionfile=distributionfilepath      #pickle file containing UNLV distribution
        self.logger=Logger()                            #if we want to use logger and store output to file
        #self.logdir = 'logdir/'
        #self.create_dir(self.logdir)
        #logging.basicConfig(filename=os.path.join(self.logdir,'Log.log'), filemode='a+', format='%(name)s - %(levelname)s - %(message)s')
        self.num_of_max_vertices=100                    #number of vertices (maximum number of words in any table)
        self.max_length_of_word=30                      #max possible length of each word
        self.row_min=2                                  #minimum number of rows in a table (includes headers)
        self.row_max=5                                #maximum number of rows in a table
        self.col_min=2                                  #minimum number of columns in a table
        self.col_max=5                                  #maximum number of columns in a table
        self.minshearval=-0.1                           #minimum value of shear to apply to images
        self.maxshearval=0.1                            #maxmimum value of shear to apply to images
        self.minrotval=-0.01                            #minimum rotation applied to images
        self.maxrotval=0.01                             #maximum rotation applied to images
        self.num_data_dims=5                            #data dimensions to store in tfrecord
        self.max_height=768                           #max image height
        self.max_width=1366                           #max image width
        self.tables_cat_dist = self.get_category_distribution(self.filesize)
        self.visualizebboxes=visualizebboxes

    def get_category_distribution(self,filesize): # Remove - replace with numbers of each category as an input
        '''This function determines the number of images from each category.
        If filesize is a multiple of 4, the categories will be equally split
        '''
        tables_cat_dist=[0,0,0,0]
        firstdiv=filesize//2
        tables_cat_dist[0]=firstdiv//2
        tables_cat_dist[1]=firstdiv-tables_cat_dist[0]

        seconddiv=filesize-firstdiv
        tables_cat_dist[2]=seconddiv//2
        tables_cat_dist[3]=seconddiv-tables_cat_dist[2]
        return tables_cat_dist

    def create_dir(self,fpath):                         #creates directory fpath if it does not exist
        if(not os.path.exists(fpath)):
            os.mkdir(fpath)

    def str_to_int(self,str):                           #converts each character in a word to equivalent int
        intsarr=np.array([ord(chr) for chr in str])
        padded_arr=np.zeros(shape=(self.max_length_of_word),dtype=np.int64)
        padded_arr[:len(intsarr)]=intsarr
        return padded_arr

    def convert_to_int(self, arr):                      #simply converts array to a string
        return [int(val) for val in arr]

    def pad_with_zeros(self,arr,shape):                 #will pad the input array with zeros to make it equal to 'shape'
        dummy=np.zeros(shape,dtype=np.int64)
        dummy[:arr.shape[0],:arr.shape[1]]=arr
        return dummy

    def generate_tables(self,driver,N_imgs,output_file_names):
        '''Creates tables (empty/filled?). Number of rows and columns are chosen (randomly) first.'''
        row_col_min=[self.row_min,self.col_min]                 #to randomly select number of rows
        row_col_max=[self.row_max,self.col_max]                 #to randomly select number of columns
        rc_arr = np.random.uniform(low=row_col_min, high=row_col_max, size=(N_imgs, 2))        #random row and col selection for N images
        table_categories=[0,0,0,0]                         #These 4 values will count the number of images for each of the category
        rc_arr[:,0]=rc_arr[:,0]+2                                     #increasing the number of rows by a fix 2. (We can comment out this line. Does not affect much)
        data_arr=[]
        exceptioncount=0

        rc_count=0                                              #for iterating through row and col array
        for assigned_category,cat_count in enumerate(self.tables_cat_dist):
            for _ in range(cat_count):
                rows = int(round(rc_arr[rc_count][0])) # Needed as np.random.uniform generates floating point values
                cols = int(round(rc_arr[rc_count][1]))

                exceptcount=0
                while(True):
                    #This loop is to repeat and retry generating image if some an exception is encountered.
                    try:
                        #initialize table class
                        table = Table(rows,cols,self.unlvimagespath,self.unlvocrpath,self.unlvtablepath,assigned_category+1,self.distributionfile)
                        #get table of rows and cols based on unlv distribution and get features of this table
                        #(same row, col and cell matrices, total unique ids, html conversion of table and its category)
                        same_cell_matrix,same_col_matrix,same_row_matrix, id_count, html_content,tablecategory= table.create()

                        #convert this html code to image using selenium webdriver. Get equivalent bounding boxes
                        #for each word in the table. This will generate ground truth for our problem
                        im,bboxes = html_to_img(driver, html_content, id_count)


                        # apply_shear: bool - True: Apply Transformation, False: No Transformation | probability weight for shearing to be 25%
                        #apply_shear = random.choices([True, False],weights=[0.25,0.75])[0]

                        #if(apply_shear==True):
                        if(assigned_category+1==4):
                            #randomly select shear and rotation levels
                            #while(True):
                            shearval = np.random.uniform(self.minshearval, self.maxshearval)
                            rotval = np.random.uniform(self.minrotval, self.maxrotval)
                            #if(shearval!=0.0 or rotval!=0.0):
                                #break
                            #If the image is transformed, then its categorycategory is 4

                            #transform image and bounding boxes of the words
                            im, bboxes = Transform(im, bboxes, shearval, rotval, self.max_width, self.max_height)
                            tablecategory=4

                        if(self.visualizeimgs):
                            #if the image and equivalent html is need to be stored
                            dirname=os.path.join('visualizeimgs/','category'+str(tablecategory))
                            f=open(os.path.join(dirname,'html',output_file_names[assigned_category]+'.html'),'w')
                            f.write(html_content)
                            f.close()
                            im.save(os.path.join(dirname,'img',output_file_names[assigned_category]+'.png'), dpi=(600, 600))

                        # driver.quit()
                        # 0/0
                        data_arr.append([[same_row_matrix, same_col_matrix, same_cell_matrix, bboxes,[tablecategory]],[im]])
                        table_categories[tablecategory-1]+=1
                        #print('Assigned category: ',assigned_category+1,', generated category: ',tablecategory)
                        break
                    except Exception as e:
                        #traceback.print_exc()
                        exceptcount+=1
                        if(exceptioncount>10):
                            print('More than 10 exceptions occured for file: ',output_file_names)
                            #if there are more than 10 exceptions, then return None
                            return None
                        #traceback.print_exc()
                        #print('\nException No.', exceptioncount, ' File: ', str(output_file_names))
                        #logging.error("Exception Occured "+str(output_file_names),exc_info=True)
                rc_count+=1
        if(len(data_arr)!=N_imgs):
            #If total number of images are not generated, then return None.
            print('Images not equal to the required size.')
            return None
        return data_arr,table_categories

    def draw_matrices(self,img,arr,matrices,imgindex,output_file_name):
        '''Call this fucntion to draw visualizations of a matrix on image'''
        no_of_words=len(arr)
        colors = np.random.randint(0, 255, (no_of_words, 3))
        arr = arr[:, 2:]

        img=img.astype(np.uint8)
        img=np.dstack((img,img,img))

        mat_names=['row','col','cell']

        for matname,matrix in zip(mat_names,matrices):
            im=img.copy()

            # x=1
            # indices = np.argwhere(matrix[x] == 1)
            # for index in indices:
            #     cv2.rectangle(im, (int(arr[index, 0])-3, int(arr[index, 1])-3),
            #                   (int(arr[index, 2])+3, int(arr[index, 3])+3),
            #                   (0,255,0), 1)
            #
            # x = 4
            # indices = np.argwhere(matrix[x] == 1)
            # for index in indices:
            #     cv2.rectangle(im, (int(arr[index, 0])-3, int(arr[index, 1])-3),
            #                   (int(arr[index, 2])+3, int(arr[index, 3])+3),
            #                   (0, 0, 255), 1)

            for x in range(no_of_words):
                indices = np.argwhere(matrix[x] == 1)
                c1 = np.random.randint(0, 255) ; c2 = np.random.randint(0, 255) ; c3 = np.random.randint(0, 255)
                for index in indices:
                    cv2.rectangle(im, (int(arr[index, 0])-3, int(arr[index, 1])-3),
                                  (int(arr[index, 2])+3, int(arr[index, 3])+3),
                                  (c1,c2,c3), 1)

            img_name=os.path.join('bboxes','category'+str(imgindex+1),output_file_name+'_'+matname+'.jpg')
            cv2.imwrite(img_name,im)


    def generate_table_set(self,filesize,threadnum):
        '''Generates a set of of four tables, one for each category.

        Args:
            filesize: number of images in one tfrecord [obsolete]
            threadnum: thread_id
        '''

        # For opening a browser session
        opts = Options()
        opts.set_headless()
        assert opts.headless
        driver = Firefox(options=opts)

        while(True):

            starttime = time.time()
            output_file_names = []
            for i in range(1,5):
                output_file_names.append(''.join(random.choices(string.ascii_lowercase + string.digits, k=20)))
            print('\nThread: ',threadnum,' Started:', output_file_names)

            #data_arr contains the images of generated tables and table_categories contains the table category of each of the table
            data_arr,table_categories = self.generate_tables(driver, filesize, output_file_names)
            if(data_arr is not None):
                if(len(data_arr)==filesize):

                    try:
                        for imgindex,subarr in enumerate(data_arr):

                            arr=subarr[0]
                            tablecategory=arr[4][0]
                            table_id = output_file_names[imgindex]

                            img=np.asarray(subarr[1][0],np.int64)[:,:,0]
                            colmatrix = np.array(arr[1],dtype=np.int64)
                            cellmatrix = np.array(arr[2],dtype=np.int64)
                            rowmatrix = np.array(arr[0],dtype=np.int64)
                            bboxes = np.array(arr[3])

                            # Output files are generated here
                            self.write_bboxes(bboxes,table_id,tablecategory)
                            self.write_matrices(colmatrix,rowmatrix,cellmatrix,table_id,tablecategory)

                            if(self.visualizebboxes):
                                cellmatrix = self.pad_with_zeros(cellmatrix,(self.num_of_max_vertices,self.num_of_max_vertices))
                                colmatrix = self.pad_with_zeros(colmatrix,(self.num_of_max_vertices,self.num_of_max_vertices))
                                rowmatrix = self.pad_with_zeros(rowmatrix,(self.num_of_max_vertices,self.num_of_max_vertices))
                                img=img.astype(np.int64)
                                self.draw_matrices(img,bboxes,[rowmatrix,colmatrix,cellmatrix],imgindex,table_id)

                        print('\nThread :',threadnum,' Completed in ',time.time()-starttime,' ' ,output_file_names,'with len:',(len(data_arr)))
                        print('category 1: ',table_categories[0],', category 2: ',table_categories[1],', category 3: ',table_categories[2],', category 4: ',table_categories[3])

                    except Exception as e:
                        print('Exception occurred in generate_table_set function for file: ',output_file_names)
                        traceback.print_exc()
                        self.logger.write(traceback.format_exc())

        driver.stop_client()
        driver.quit()


    def start_generation(self,max_threads):
        '''Starts table generation using specified number of threads, where each thread is used to generate a set of
        four tables (one from each category). It also creates various directories to store table images and annotations.

        Args:
            max_threads: the number of threads to use to generate tables.
        '''

        if(not os.path.exists(self.distributionfile)):
            if((not os.path.exists(self.unlvtablepath)) or (not os.path.exists(self.unlvimagespath)) or (not os.path.exists(self.unlvocrpath))):
                print('Dataset folders do not exist.')
                return

        #create all directories here
        if(self.visualizeimgs):
            self.create_dir('visualizeimgs')
            for tablecategory in range(1,5):
                dirname=os.path.join('visualizeimgs/','category'+str(tablecategory))
                self.create_dir(dirname)
                self.create_dir(os.path.join(dirname,'html'))
                self.create_dir(os.path.join(dirname, 'img'))

        if(self.visualizebboxes):
            self.create_dir('bboxes')
            for tablecategory in range(1,5):
                dirname=os.path.join('bboxes','category'+str(tablecategory))
                self.create_dir(dirname)

        self.create_dir(self.outtfpath)              #create output directory if it does not exist
        for tablecategory in range(1,5):
            dirname=os.path.join(self.outtfpath,'category'+str(tablecategory))
            self.create_dir(dirname)

        starttime=time.time()
        threads=[]
        for threadnum in range(max_threads):
            proc = Process(target=self.generate_table_set, args=(self.filesize, threadnum,))
            proc.start()
            threads.append(proc)

        for proc in threads:
            proc.join()
        print(time.time()-starttime)

    def write_bboxes(self,bboxes,table_id,table_category):
        ''' New function written by MTW.  Saves cell bboxes array as pickled array.'''

        output_file_name_bbox=table_id+'.bboxes'
        pickle.dump(bboxes,open(os.path.join(self.outtfpath,'category'+str(table_category),output_file_name_bbox),"wb"))

    def write_matrices(self,col_mat,row_mat,cell_mat,table_id,table_category):
        ''' New function written by MTW.  Saves adjacency matrices as pickled dictionary.'''
        all_mat = {}
        all_mat['row'] = row_mat ; all_mat['col'] = col_mat ; all_mat['cell'] = cell_mat ;
        output_file_name_mat=table_id+'.matrices'
        pickle.dump(all_mat,open(os.path.join(self.outtfpath,'category'+str(table_category),output_file_name_mat),"wb"))
