# -*- coding: utf-8 -*-
"""
Created on Wed May 11 08:43:50 2016

@author: Xu Gang
"""

import os

def readPDB(filename):
    f = open(filename,'r')
    atomsData = []
    for line in f.readlines():   
        if (line.strip() == ""):
            break
        else:
            if (line[:4] == 'ATOM'):
                name1 = line[11:16].strip()                    
                if(name1 in ["N","CA","C","O"]):
                    atomsData.append(line)
            # elif (line[:6] == 'HETATM'):
            #     atomsData.append(line)
    f.close()
    return atomsData

if __name__ == "__main__":

    pdb_lists = []
    f = open(r'./list_casp15', 'r')
    for i in f.readlines():
        pdb_lists.append(i.strip())
    f.close()
    print (len(pdb_lists))
    
    path = "./casp15_raw"    
    path2 = "./casp15"    
    for filename in pdb_lists:
        atomsData = readPDB(os.path.join(path, filename + ".pdb")) 
        f = open(os.path.join(path2, filename +'.pdb'), 'w')
        for i in atomsData:
            f.writelines(i)
        f.close()

                
                


    