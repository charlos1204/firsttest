import sys
import process_blast
import make_XY
import train_deepARG

fname = sys.argv[1] #file name dataset

#print("procesando",fname)

# load the alignment sequences (samples) from where the model is trained.
alignments1 = process_blast.make_alignments_json(fname, iden=30, eval=1, len=25, BitScore=True)


data = make_XY.make_xy2(alignments1)

deepL = train_deepARG.main(data)

#print(data)