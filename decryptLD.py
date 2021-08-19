import os

def crypt(addtxt,basedir):
    for root,dirs,files in os.walk(basedir): 
        for file in files: 
            name_o = os.path.join(root,file).replace("\\","/")
            if addtxt:
                os.rename(name_o,name_o +".txt")
            elif name_o.endswith(".txt"):
                os.rename(name_o,name_o[0:-4])

if __name__ == '__main__':
    crypt(0,"D:/TestData/testdecrypt")