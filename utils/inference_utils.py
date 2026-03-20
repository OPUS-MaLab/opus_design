# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""

import os
import tensorflow as tf
import numpy as np
from utils.mkinputs import PDBreader, structure, vector, Geometry, getPhiPsiOmega
    
ss8 = "CSTHGIEB"
ss8_dict = {}
for k,v in enumerate(ss8):
    ss8_dict[v] = k

def cleanPDB(file_path, file_path2):
    f = open(file_path,'r')
    atomsData = []
    for line in f.readlines():   
        if (line.strip() == ""):
            break
        else:
            if (line[:4] == 'ATOM'):
                atomsData.append(line)
    f.close()
    f = open(file_path2, 'w')
    for i in atomsData:
        f.writelines(i)
    f.close()    
    
def mk_input1d(file_path, filename, preparation_config):

    atomsData = PDBreader.readPDB(file_path) 
    residuesData = structure.getResidueData(atomsData) 
    dihedralsData = getPhiPsiOmega.getDihedrals(residuesData)
    
    length = len(dihedralsData)
    
    pps = []
    for i in dihedralsData:
        pps.append([np.sin(np.deg2rad(i.pp[0])), np.cos(np.deg2rad(i.pp[0])), 
                    np.sin(np.deg2rad(i.pp[1])), np.cos(np.deg2rad(i.pp[1])),
                    np.sin(np.deg2rad(i.pp[2])), np.cos(np.deg2rad(i.pp[2]))])

    cmd = preparation_config["mkdssp_path"] + ' ' + file_path
    print (cmd) 
    output = os.popen(cmd).read()
        
    ss = ["-"]*length
    resid = -1
    bad = 0
    for i in output.split("\n"):
        if len(i) > 13 and i[13] == "!":
            bad += 1
        if i.strip() != "" and i.strip()[0] != '#' and i.strip()[-1] != '.' and i[13] != "!":
            resid = int(i.split()[0])
            if i[16] == " ":
                ss8 = "C"
            else:
                ss8 = i[16]
            ss[resid-1-bad] = ss8
    assert (resid - bad) == length == len(pps)

    inputs_1d = np.zeros((length, 17))
    # 8ss(one-hot) + 3ss(one-hot) + 2*(phi+psi+omega)
    # 8 + 3 + 6 = 17
    ss_content = ss
    for i in range(length):
        if ss_content[i] != "-":
            inputs_1d[i][ss8_dict[ss_content[i]]] = 1

    inputs_1d[:,8] = np.sum(inputs_1d[:,:3],-1)
    inputs_1d[:,9] = np.sum(inputs_1d[:,3:6],-1)
    inputs_1d[:,10] = np.sum(inputs_1d[:,6:8],-1)
    
    pps = np.array(pps)
    inputs_1d[:,11:] = pps
    
    np.savez_compressed(os.path.join(preparation_config["tmp_files_path"], filename+".input1d"), f=inputs_1d)  

def mk_input2d(file_path, filename, preparation_config):

    atomsData = PDBreader.readPDB(file_path) 
    residuesData = structure.getResidueData(atomsData) 
    length = len(residuesData)
    
    dist_ref, omega_ref, theta_ref, phi_ref = \
        np.zeros((length, length)), np.zeros((length, length)), np.zeros((length, length)), np.zeros((length, length))
    
    for res in residuesData:
        res.resname = "A"
        
    for i in range(length):
        residue_a = residuesData[i]
        a_ca = residue_a.atoms["CA"].position
        a_n = residue_a.atoms["N"].position
        a_geo = Geometry.geometry(residue_a.resname)
        a_cb = vector.calculateCoordinates(
                residue_a.atoms["C"], residue_a.atoms["N"], residue_a.atoms["CA"], 
                a_geo.CA_CB_length, a_geo.C_CA_CB_angle, a_geo.N_C_CA_CB_diangle)
                
        for j in range(length):
            if i == j:
                continue
            residue_b = residuesData[j]
            b_ca = residue_b.atoms["CA"].position
            b_geo = Geometry.geometry(residue_b.resname)
            b_geo = Geometry.geometry("A")
    
            b_cb = vector.calculateCoordinates(
                    residue_b.atoms["C"], residue_b.atoms["N"], residue_b.atoms["CA"], 
                    b_geo.CA_CB_length, b_geo.C_CA_CB_angle, b_geo.N_C_CA_CB_diangle)
        
            dist_ref[i][j] = np.linalg.norm(a_cb - b_cb)
            omega_ref[i][j] = np.deg2rad(vector.calc_dihedral(a_ca, a_cb, b_cb, b_ca))
            theta_ref[i][j] = np.deg2rad(vector.calc_dihedral(a_n, a_ca, a_cb, b_cb))
            phi_ref[i][j] = np.deg2rad(vector.calc_angle(a_ca, a_cb, b_cb))

    p_dist  = mtx2bins(dist_ref,     2.0,  20.0, 37, mask=(dist_ref > 20))
    p_omega = mtx2bins(omega_ref, -np.pi, np.pi, 37, mask=(p_dist[...,0]==1))
    p_theta = mtx2bins(theta_ref, -np.pi, np.pi, 37, mask=(p_dist[...,0]==1))
    p_phi   = mtx2bins(phi_ref,      0.0, np.pi, 19, mask=(p_dist[...,0]==1))
    feat    = np.concatenate([p_theta, p_phi, p_dist, p_omega],-1)
    
    assert feat.shape == (length, length, 130)
    
    feat = np.array(feat, dtype=np.int8)
    np.savez_compressed(os.path.join(preparation_config["tmp_files_path"], filename+".input2d"), f=feat)  

def mtx2bins(x_ref, start, end, nbins, mask):
    bins = np.linspace(start, end, nbins)
    x_true = np.digitize(x_ref, bins).astype(np.uint8)
    x_true[mask] = 0
    return np.eye(nbins+1)[x_true][...,:-1]

def read_inputs(filenames, preparation_config):
    """
    8ss(one-hot) + 3ss(one-hot) + 2*(phi+psi+omega)
    8 + 3 + 6 = 17
    """
    inputs_1ds = []
    inputs_2ds = []
    assert len(filenames) == 1
    for filename in filenames:
        inputs_1d = np.load(os.path.join(preparation_config["tmp_files_path"], filename + ".input1d.npz"))["f"]
        seq_len = inputs_1d.shape[0]

        # 3dcnn
        feat_3dcnn = np.load(os.path.join(preparation_config["tmp_files_path"], filename + ".3dcnn.npz"))["f"]
        assert feat_3dcnn.shape == (seq_len, 15*15*15*5+20)

        inputs_1d = np.concatenate((inputs_1d, feat_3dcnn[:,15*15*15*5:], feat_3dcnn[:,:15*15*15*5]),axis=1)
        assert inputs_1d.shape == (seq_len, 17 + 20 + 15*15*15*5)
        
        # 2d
        inputs_2d = np.load(os.path.join(preparation_config["tmp_files_path"], filename + ".input2d.npz"))["f"]
        assert inputs_2d.shape == (seq_len, seq_len, 130)
        
        inputs_1ds.append(inputs_1d)
        inputs_2ds.append(inputs_2d)
        
        inputs_total_len = seq_len
        
    inputs_1ds = np.array(inputs_1ds)
    inputs_2ds = np.array(inputs_2ds)
            
    return inputs_1ds, inputs_2ds, inputs_total_len


class InputReader(object):

    def __init__(self, data_list, preparation_config):

        self.data_list = data_list
        self.preparation_config = preparation_config
        self.dataset = tf.data.Dataset.from_tensor_slices(self.data_list).batch(1)          
        
        print ("Data Size:", len(self.data_list)) 
    
    def read_file_from_disk(self, filenames_batch):
        
        filenames_batch = [bytes.decode(i) for i in filenames_batch.numpy()]
        inputs_1ds_batch, inputs_2ds_batch, inputs_total_len = \
            read_inputs(filenames_batch, self.preparation_config)
        
        inputs_1ds_batch = tf.convert_to_tensor(inputs_1ds_batch, dtype=tf.float32)
        inputs_2ds_batch = tf.convert_to_tensor(inputs_2ds_batch, dtype=tf.float32)
        
        return filenames_batch, inputs_1ds_batch, inputs_2ds_batch, inputs_total_len
            