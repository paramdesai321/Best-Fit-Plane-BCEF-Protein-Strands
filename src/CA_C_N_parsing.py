import parsingcoord
x_coord=[]
y_coord=[]
z_coord = []
def DesiredAtoms(line):

        if(((line[12:16].strip())=="CA")or((line[12:16].strip())=="C")or((line[12:16].strip())=="N")):
            parsingcoord.striping_coords(line,x_coord,y_coord,z_coord)

            print(line)
            return True

with open(f'ParsedAtoms1CD8.txt','w') as wf:

    with open(f'./ATOMlines1CD8.txt', 'r') as rf:

        for line in rf:

            #DesiredAtoms(line)


            #print(line)
            if(DesiredAtoms(line)==True):
                print("---------------------------")
                wf.write(line)





def x_coordinates():
    return x_coord;

def y_coordinates():
    return y_coord;

def z_coordinates():
    return z_coord
