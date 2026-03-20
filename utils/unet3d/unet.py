##################################################################################
# This code is a part from DLPacker.

# @article {Misiura2021.05.23.445347,
#     author = {Misiura, Mikita and Shroff, Raghav and Thyer, Ross and Kolomeisky, Anatoly},
#     title = {DLPacker: Deep Learning for Prediction of Amino Acid Side Chain Conformations in Proteins},
#     elocation-id = {2021.05.23.445347},
#     year = {2021},
#     doi = {10.1101/2021.05.23.445347},
#     publisher = {Cold Spring Harbor Laboratory},
#     URL = {https://www.biorxiv.org/content/early/2021/05/25/2021.05.23.445347},
#     eprint = {https://www.biorxiv.org/content/early/2021/05/25/2021.05.23.445347.full.pdf},
#     journal = {bioRxiv}
#
#If you find use, please cite it.                                         
##################################################################################
import numpy as np
import torch

from Bio.PDB import PDBParser, Selection, Superimposer, PDBIO, Residue, Structure
from utils.unet3d.utils3d import InputBoxReader, THE20, THE20_dict, BB_ATOMS, BOX_SIZE, THE20_single, THE20_single_dict, THE20_a2aaa, THE20_aaa2a
    
class U3DEng():
    def __init__(self, str_pdb:str, models = None):
        self.parser = PDBParser(PERMISSIVE = 1)
        self.sup = Superimposer()
        self.io = PDBIO()
        
        self.box_size = BOX_SIZE
        
        self.str_pdb = str_pdb
        self.ref_pdb = './utils/unet3d/reference.pdb'
        self._read_structures()
        self.reconstructed = None
        
        self.models = models
        self.input_reader = InputBoxReader()

    def _read_structures(self):
        self.structure = self.parser.get_structure('structure', self.str_pdb)
        self.reference = self.parser.get_structure('reference', self.ref_pdb)
        
        self._remove_hydrogens(self.structure) # we never use hydrogens
        self._remove_water(self.structure)     # waters are not used anyway
    
    def _remove_hydrogens(self, structure:Structure):
        for residue in Selection.unfold_entities(structure, 'R'):
            remove = []
            for atom in residue:
                if atom.element == 'H': remove.append(atom.get_id())
            for i in remove: residue.detach_child(i)
    
    def _remove_water(self, structure:Structure):
        residues_to_remove = []
        for residue in Selection.unfold_entities(structure, 'R'):
            if residue.get_resname() == 'HOH':
                residues_to_remove.append(residue)
        for r in residues_to_remove:
            r.get_parent().detach_child(r.get_id())
            
    def _get_residue_tuple(self, residue:Residue):
        r = residue.get_id()[1]
        s = residue.get_full_id()[2]
        n = residue.get_resname()
        return (r, s, n)
    
    def _get_parent_structure(self, residue:Residue):
        return residue.get_parent().get_parent().get_parent()
    
    def _align_residue(self, residue:Residue):
        if not residue.has_id('N') or not residue.has_id('C') or not residue.has_id('CA'):
            print('Missing backbone atoms: residue', self._get_residue_tuple(residue))
            return False
        r = list(self.reference.get_atoms())
        s = [residue['N'], residue['CA'], residue['C']]
        self.sup.set_atoms(r, s)
        self.sup.apply(self._get_parent_structure(residue))
        return True
    
    def _get_box_atoms(self, residue:Residue):
        aligned = self._align_residue(residue)
        if not aligned: return []
        atoms = []
        b = self.box_size + 1
        for a in self._get_parent_structure(residue).get_atoms():
            xyz = a.coord
            if xyz[0] < b and xyz[0] > -b and\
               xyz[1] < b and xyz[1] > -b and\
               xyz[2] < b and xyz[2] > -b:
                atoms.append(a)
        return atoms
    
    def _genetare_input_box(self, residue:Residue):
        atoms = self._get_box_atoms(residue)
        if not atoms: return None
        
        r, s, n = self._get_residue_tuple(residue)
        
        types, resnames = [], []
        segids, positions, names = [], [], []
        resids = []
        for i, a in enumerate(atoms):
            types.append(a.element)
            resnames.append(a.get_parent().get_resname())
            segids.append(a.get_parent().get_full_id()[2])
            positions.append(a.coord)
            names.append(a.get_name())
            resids.append(a.get_parent().get_id()[1])
            
        d = {'target': {'id': int(r), 'segid': s, 'name': n},\
             'types': np.array(types),\
             'resnames': np.array(resnames),\
             'segids': np.array(segids),\
             'positions': np.array(positions, dtype = np.float16),\
             'names': np.array(names),\
             'resids': np.array(resids)}
        
        return d
    
    def _get_sorted_residues(self, structure:Structure):
        out = []
        for residue in Selection.unfold_entities(structure, 'R'):
            if residue.get_resname() in THE20 or residue.get_resname() == 'UNK':
                out.append(residue)
        return out
        
    def _remove_sidechains(self, structure:Structure):
        for residue in Selection.unfold_entities(structure, 'R'):
            if residue.get_resname() in THE20:
                self._remove_sidechain(residue)
    
    def _remove_sidechain(self, residue:Residue):
        l = []
        for atom in residue:
            if atom.get_name() not in BB_ATOMS:
                l.append(atom.get_id())
        for d in l: residue.detach_child(d)

    def _get_prediction(self, box:dict):
        i = self.input_reader(box)
        
        o11, o12 = self.models[0].model(i[None, ...], training=False)
        o21, o22 = self.models[1].model(i[None, ...], training=False)
        o31, o32 = self.models[2].model(i[None, ...], training=False)
        o41, o42 = self.models[3].model(i[None, ...], training=False)
        o51, o52 = self.models[4].model(i[None, ...], training=False)
        
        output_label = (o11 + o21 + o31 + o41 + o51)/5.0
        
        o12[o12 < 0] = 0
        o22[o22 < 0] = 0
        o32[o32 < 0] = 0
        o42[o42 < 0] = 0
        o52[o52 < 0] = 0
        density = (o12 + o22 + o32 + o42 + o52)/5.0
        pred = np.concatenate([np.sum(i[..., :4], -1, keepdims=True), density], -1)

        pred = pred[5:-5, 5:-5, 5:-5, :]
        
        dpred_ori = np.zeros((15, 15, 15, 5))
        for i in range(0, 30, 2):
            for j in range(0, 30, 2):
                for k in range(0, 30, 2):
                    v = np.mean(pred[i:i+2, j:j+2, k:k+2, :], axis = (0, 1, 2))
                    dpred_ori[i//2, j//2, k//2, :] = v
                    
        pred_label = np.argmax(output_label)
        prob_label = output_label[pred_label]
        aa_new = THE20_dict[int(pred_label)]
                    
        return prob_label, aa_new, output_label, dpred_ori
    
    def reconstruct_residue(self, residue:Residue, sm_esm, final:bool = False):
        r, s, n = self._get_residue_tuple(residue)
        box = self._genetare_input_box(residue)
        
        if not box:
            print("Skipping residue:", (r, s, n), end = '\n')
            return
        
        aa_prob, aa_new, output_label, pred_ori = self._get_prediction(box)

        if aa_prob >= 0.5:
            residue.resname = aa_new
        elif np.max(sm_esm) < 0.3 and not final:
            residue.resname = "UNK"
        
        if final:
            if aa_prob >= 0.5:
                output_label = output_label
            elif np.max(sm_esm) >= 0.3:
                output_label = sm_esm
            else:
                output_label = output_label
                
        return output_label, pred_ori
    
    def esm_refine(self, esm_contents, sorted_residues):
        fasta = ""
        for residue in sorted_residues:
            if residue.resname in THE20_aaa2a:
                fasta += THE20_aaa2a[residue.resname]      
            else:
                fasta += '<mask>'
        data = [
            ("tmp", fasta),
        ]
        batch_labels, batch_strs, batch_tokens = esm_contents["batch_converter"](data)
        with torch.no_grad():
            results = esm_contents["esm_model"](batch_tokens, repr_layers=[33], return_contacts=False)
        result = np.argmax(results['logits'][0][1:-1], -1)

        sm = torch.nn.functional.softmax(results['logits'][0][1:-1], -1).numpy()
        sm = tranferSM(sm)

        assert result.shape[0] == len(sorted_residues) == sm.shape[0]
        
        for i in range(len(sorted_residues)):
            tok = esm_contents["alphabet"].get_tok(result[i])
            if tok in THE20_single:
                sorted_residues[i].resname = THE20_a2aaa[tok]
        return sm
       
    def reconstruct_protein(self, seq_len:int, output_path:str = '', output_path2:str = '',
                            esm_contents=None):
        if not self.reconstructed: self.reconstructed = self.structure.copy()
        else: print('Reconstructed structure already exists, something might be wrong!')
        self._remove_sidechains(self.reconstructed)

        feat_3dcnn = np.zeros((seq_len, 15*15*15*5+20))
        
        fasta_array = []
        sorted_residues = self._get_sorted_residues(self.reconstructed)
        for i, residue in enumerate(sorted_residues):
            name = self._get_residue_tuple(residue)
            print("Working on residue:", i, name, end = '\r')
            self.reconstruct_residue(residue, np.zeros(20))
        sm_esm = self.esm_refine(esm_contents, sorted_residues)  
        
        for i, residue in enumerate(sorted_residues):
            name = self._get_residue_tuple(residue)
            print("Working on residue:", i, name, end = '\r')
            self.reconstruct_residue(residue, sm_esm[i])
        sm_esm = self.esm_refine(esm_contents, sorted_residues)  

        for i, residue in enumerate(sorted_residues):
            name = self._get_residue_tuple(residue)
            print("Working on residue:", i, name, end = '\r')
            output_label, pred_ori = self.reconstruct_residue(residue, sm_esm[i], final=True)
            feat1 = pred_ori.flatten()
            feat2 = output_label.flatten()
            fasta_array.append(np.argmax(feat2))
            feat = np.concatenate([feat1, feat2], -1)
            assert feat.shape == (15*15*15*5+20,)
            feat_3dcnn[i] = feat                
        assert seq_len == i + 1
        assert len(fasta_array) == i + 1
        
        feat_3dcnn = np.array(feat_3dcnn, dtype=np.float16)
        np.savez_compressed(output_path, f=feat_3dcnn)   
        
        fasta = ""
        for i in fasta_array:
            fasta += THE20_single_dict[i]
        
        fw = open(output_path2, 'w')
        filename = output_path2.split('/')[-1].split('.')[0]
        fw.writelines(">" + filename + "\n")
        fw.writelines(fasta)
        fw.close()

def tranferSM(emb):
    embs = [
    emb[:,5],
    emb[:,10],
    emb[:,17],
    emb[:,13],
    emb[:,23],
    emb[:,16],
    emb[:,9],
    emb[:,6],
    emb[:,21],
    emb[:,12],
    emb[:,4],
    emb[:,15],
    emb[:,20],
    emb[:,18],
    emb[:,14],
    emb[:,8],
    emb[:,11],
    emb[:,22],
    emb[:,19],
    emb[:,7],
    ]
    return np.array(embs).T
