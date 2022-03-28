from src.main import *
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--tablesets',type=int,default=1,
                    help="Number of table sets to generate, where one set contains four tables, one from each category. Must be divisible by --threads.")
parser.add_argument('--threads',type=int,default=1,
                    help="Number of threads to use, where one thread generates one set of tables.")
parser.add_argument('--outpath',default='output/',
                    help="Path to directory where all output files will be saved.")
parser.add_argument('--minrows',type=int,default=1,
                    help="Minimum number of rows in the generated tables.")
parser.add_argument('--maxrows',type=int,default=10,
                    help="Maximum number of rows in the generated tables.")
parser.add_argument('--mincols',type=int,default=1,
                    help="Minimum number of columns in the generated tables.")
parser.add_argument('--maxcols',type=int,default=10,
                    help="Maximum number of columns in the generated tables.")
parser.add_argument('--visualizeimgs',type=int,default=0,
                    help="Whether to save image files: yes {1} or no {0}.")
parser.add_argument('--visualizebboxes',type=int,default=0,
                    help="Whether to save image files annotated with cell bboxes: yes {1} or no {0}.")

# Need to update these and work out if they are all required
parser.add_argument('--imagespath',default='../Table_Detection_Dataset/unlv/train/images')
parser.add_argument('--ocrpath',default='../Table_Detection_Dataset/unlv/unlv_xml_ocr')
parser.add_argument('--tablepath',default='../Table_Detection_Dataset/unlv/unlv _xml_gt')

args=parser.parse_args()

visualizeimgs=False
if(args.visualizeimgs==1):
    visualizeimgs=True

visualizebboxes=False
if(args.visualizebboxes==1):
	visualizebboxes=True

distributionfile='unlv_distribution' # Hard-coded

if args.threads > 1 and args.tablesets%args.threads != 0:
    print("You are using trying to use {} threads to generate {} sets of tables.\nPlease ensure that the number of table sets is divisible by the number of threads.".format(args.threads,args.tablesets))
    exit(130)
else:
    number_of_generations=args.tablesets//args.threads

t = TableGenerator(args.outpath,number_of_generations,args.imagespath,args.ocrpath,args.tablepath,visualizeimgs,visualizebboxes,distributionfile,args.minrows,args.maxrows,args.mincols,args.maxcols)
t.start_generation(args.threads)
