from src.main import *
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--tablesets',type=int,default=1)             # Number of table sets to generate, where one set contains four tables, one from each category
parser.add_argument('--threads',type=int,default=1)               # One thread will generate one set of tables
parser.add_argument('--outpath',default='output/')             # Directory to bbox data and adjacency matrices

parser.add_argument('--minrows',type=int,default=1)
parser.add_argument('--maxrows',type=int,default=10)
parser.add_argument('--mincols',type=int,default=1)
parser.add_argument('--maxcols',type=int,default=10)

#imagespath,
parser.add_argument('--imagespath',default='../Table_Detection_Dataset/unlv/train/images')
parser.add_argument('--ocrpath',default='../Table_Detection_Dataset/unlv/unlv_xml_ocr')
parser.add_argument('--tablepath',default='../Table_Detection_Dataset/unlv/unlv _xml_gt')

parser.add_argument('--visualizeimgs',type=int,default=0)           # If 1, will store the images along with tfrecords
parser.add_argument('--visualizebboxes',type=int,default=0)			# If 1, will store the bbox visualizations in visualizations folder
args=parser.parse_args()

visualizeimgs=False
if(args.visualizeimgs==1):
    visualizeimgs=True

visualizebboxes=False
if(args.visualizebboxes==1):
	visualizebboxes=True

distributionfile='unlv_distribution'

if args.threads > 1 and args.tablesets%args.threads != 0:
    print("You are using trying to use {} threads to generate {} sets of tables.\nPlease ensure that the number of table sets is divisible by the number of threads.".format(args.threads,args.tablesets))
    exit(130)
else:
    number_of_generations=args.tablesets//args.threads

t = TableGenerator(args.outpath,number_of_generations,args.imagespath,args.ocrpath,args.tablepath,visualizeimgs,visualizebboxes,distributionfile,args.minrows,args.maxrows,args.mincols,args.maxcols)
t.start_generation(args.threads)
