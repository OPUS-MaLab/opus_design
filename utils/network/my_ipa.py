""" Invariant point attention module adapted from AlphaFold2 """
import tensorflow as tf
from tensorflow import keras
import numpy as np
import utils.network.pre_trained_embedding.model.AttentionModule as PAttention
import utils.network.pre_trained_embedding.model.LinearModule as PLinear
DTYPE = tf.float32

QUAT_TO_ROT = np.zeros((4, 4, 3, 3), dtype=np.float32)

QUAT_TO_ROT[0, 0] = [[ 1, 0, 0], [ 0, 1, 0], [ 0, 0, 1]]
QUAT_TO_ROT[1, 1] = [[ 1, 0, 0], [ 0,-1, 0], [ 0, 0,-1]]
QUAT_TO_ROT[2, 2] = [[-1, 0, 0], [ 0, 1, 0], [ 0, 0,-1]]
QUAT_TO_ROT[3, 3] = [[-1, 0, 0], [ 0,-1, 0], [ 0, 0, 1]]

QUAT_TO_ROT[1, 2] = [[ 0, 2, 0], [ 2, 0, 0], [ 0, 0, 0]]
QUAT_TO_ROT[1, 3] = [[ 0, 0, 2], [ 0, 0, 0], [ 2, 0, 0]]
QUAT_TO_ROT[2, 3] = [[ 0, 0, 0], [ 0, 0, 2], [ 0, 2, 0]]

QUAT_TO_ROT[0, 1] = [[ 0, 0, 0], [ 0, 0,-2], [ 0, 2, 0]]
QUAT_TO_ROT[0, 2] = [[ 0, 0, 2], [ 0, 0, 0], [-2, 0, 0]]
QUAT_TO_ROT[0, 3] = [[ 0,-2, 0], [ 2, 0, 0], [ 0, 0, 0]]

class QuatAffine:
    """
    Adapted from AlphaFold source code:
    alphafold/alphafold/model/quat_affine.py starting from line 181
    """
    def __init__(self, quaternion, translation, rotation=None, normalize=True):
        if normalize and quaternion is not None:
            quaternion = quaternion / tf.linalg.norm(quaternion, axis=-1,
                                                     keepdims=True)
        if rotation is None:
            rotation = self.quat_to_rot(quaternion)

        self.quaternion = quaternion
        self.rotation = [list(row) for row in rotation]
        self.translation = list(translation)

    @classmethod
    def generate_new_affine(cls, bsz, n_res):
        quaternion = tf.tile(
            tf.Variable([[[1., 0., 0., 0.]]]),
            (bsz, n_res, 1))

        translation = tf.zeros((3, bsz, n_res))

        return QuatAffine(quaternion, translation)

    def expand_dims_rot_trans(self, extra_dims=0):
        """ expand rotation and translation along the last dimensions """
        rotation = self.rotation
        translation = self.translation
        for _ in range(extra_dims):
            rotation = [
                [tf.expand_dims(col, axis=-1) for col in row]
                for row in rotation ]

            translation = [
                tf.expand_dims(x, axis=-1)
                for x in translation ]

        return rotation, translation

    def apply_to_point(self, point, extra_dims=0):
        rotation, translation = self.expand_dims_rot_trans(extra_dims)
        rot_point = self.apply_rot_to_vec(rotation, point)
        transformed_point = [
            rot_point[0] + translation[0],
            rot_point[1] + translation[1],
            rot_point[2] + translation[2]]
        return transformed_point

    def invert_point(self, transformed_point, extra_dims=0):
        rotation, translation = self.expand_dims_rot_trans(extra_dims)
        rot_point = [
            transformed_point[0] - translation[0],
            transformed_point[1] - translation[1],
            transformed_point[2] - translation[2]]
        point = self.apply_inverse_rot_to_vec(rotation, rot_point)
        return point

    @classmethod
    def apply_rot_to_vec(cls, rot, vec):
        x, y, z = vec
        return [rot[0][0] * x + rot[0][1] * y + rot[0][2] * z,
                rot[1][0] * x + rot[1][1] * y + rot[1][2] * z,
                rot[2][0] * x + rot[2][1] * y + rot[2][2] * z]

    @classmethod
    def apply_inverse_rot_to_vec(cls, rot, vec):
        return [rot[0][0] * vec[0] + rot[1][0] * vec[1] + rot[2][0] * vec[2],
                rot[0][1] * vec[0] + rot[1][1] * vec[1] + rot[2][1] * vec[2],
                rot[0][2] * vec[0] + rot[1][2] * vec[1] + rot[2][2] * vec[2]]

    @classmethod
    def quat_to_rot(cls, normalized_quat):
        rot_tensor = tf.reduce_sum(
            tf.reshape(QUAT_TO_ROT, (4,4,9)) *
            normalized_quat[..., :, None, None] *
            normalized_quat[..., None, :, None],
            axis=(-3, -2))
        rot = tf.einsum("...i->i...", rot_tensor)
        return [[rot[0], rot[1], rot[2]],
                [rot[3], rot[4], rot[5]],
                [rot[6], rot[7], rot[8]]]

class InvariantPointAttention(tf.Module):
    """
    Algorithm 22 Invariant point attention (IPA) SI Page 28
    Adapted from AlphaFold source code:
    alphafold/alphafold/model/folding.py
    """
    def __init__(self, name, n_msa_feat, n_pair_feat, n_head,
                 n_scalar_qk, n_scalar_v,
                 n_point_qk, n_point_v,
                 use_3d, # enable 3D point attention
                 use_struct,
                 use_nanometer,
                 dist_epsilon=1e-8):
        super(InvariantPointAttention, self).__init__(name=name)

        self.n_msa_feat = n_msa_feat
        self.n_pair_feat = n_pair_feat

        self.n_head = n_head
        self.n_scalar_qk = n_scalar_qk
        self.n_scalar_v = n_scalar_v
        self.n_point_qk = n_point_qk
        self.n_point_v = n_point_v

        self.use_3d = use_3d
        self.use_struct = use_struct
        self.use_nanometer = use_nanometer
        self.dist_epsilon = dist_epsilon

        self.DTYPE = DTYPE

        with self.name_scope:
            self.fc_scalar_q = PLinear.Linear(n_head*n_scalar_qk, num_input=self.n_msa_feat, name='q_scalar', dtype=self.DTYPE, load=True)
            self.fc_scalar_kv = PLinear.Linear(n_head*(n_scalar_v + n_scalar_qk), num_input=self.n_msa_feat, name='kv_scalar', dtype=self.DTYPE, load=True)

            self.fc_point_q = PLinear.Linear(n_head*3*n_point_qk, num_input=self.n_msa_feat, name='q_point_local', dtype=self.DTYPE, load=True)
            self.fc_point_kv = PLinear.Linear(n_head*3*(n_point_qk+n_point_v), num_input=self.n_msa_feat, name='kv_point_local', dtype=self.DTYPE, load=True)

            self.fc_pair = PLinear.Linear(n_head, num_input=self.n_pair_feat, name='attention_2d', dtype=self.DTYPE, load=True)

            self.trainable_point_weights = tf.Variable(tf.ones([n_head])*0.235, name='trainable_point_weights')
            self.fc_output = PLinear.Linear(n_msa_feat, num_input=2112, name='output_projection', dtype=self.DTYPE, load=True)
    
    def __call__(self, msa, pair, L, affine, pred_backbone, return_att):
        """
        msa: 1d sequence vector (1, L, n_msa_feat)  c_s = 384
        pair: 2d pair matrix (1, L, L, n_pair_feat)  c_z = 128
        affine: 3d structure, local frames per residue
                .rotation: (3, (3, (1, L)))
                .translation: (3, (1, L))
        """

        q_scalar = self.fc_scalar_q(msa)
        q_scalar = tf.reshape(q_scalar, [1, L, self.n_head, self.n_scalar_qk])

        kv_scalar = self.fc_scalar_kv(msa)
        kv_scalar = tf.reshape(kv_scalar, [1, L, self.n_head, self.n_scalar_v + self.n_scalar_qk])
        
        k_scalar = kv_scalar[:, :, :, :self.n_scalar_qk]
        v_scalar = kv_scalar[:, :, :, self.n_scalar_qk:]

        q_point_local = self.fc_point_q(msa)
        q_point_local = tf.split(q_point_local, 3, axis=-1)
        q_point_global = affine.apply_to_point(q_point_local, extra_dims=1)
        q_point = [
            tf.reshape(x, [1, L, self.n_head, self.n_point_qk])
            for x in q_point_global
        ]

        kv_point_local = self.fc_point_kv(msa)
        kv_point_local = tf.split(kv_point_local, 3, axis=-1)
        kv_point_global = affine.apply_to_point(kv_point_local, extra_dims=1)
        kv_point_global = [
            tf.reshape(x, [1, L, self.n_head, self.n_point_qk + self.n_point_v])
            for x in kv_point_global]
        k_point = [x[:, :, :, :self.n_point_qk] for x in kv_point_global]

        v_point = [x[:, :, :, self.n_point_qk:] for x in kv_point_global]

        n_logit_terms = 3
        scalar_variance = max(self.n_scalar_qk, 1) * 1.0
        scalar_weights = (1.0/(n_logit_terms * scalar_variance))**0.5

        q = tf.transpose(scalar_weights * q_scalar, (0, 2, 1, 3))
        k = tf.transpose(k_scalar, (0, 2, 1, 3))
        v = tf.transpose(v_scalar, (0, 2, 1, 3))

        attn_qk_scalar = tf.einsum("bhic,bhjc->bhij", q, k)
        attn_logits = attn_qk_scalar

        attention_2d_weights = (1.0/n_logit_terms)**0.5

        attention_2d = self.fc_pair(pair)
        attention_2d = tf.transpose(attention_2d, (0, 3, 1, 2))
        attention_2d = attention_2d_weights * attention_2d
        attn_logits += attention_2d

        point_variance = max(self.n_point_qk, 1) * 9.0/2
        point_weights = (1.0/(n_logit_terms * point_variance))**0.5
        trainable_point_weights = tf.math.softplus(self.trainable_point_weights)
        point_weights *= tf.reshape(trainable_point_weights, shape=[self.n_head, 1, 1, 1])

        q_point = [tf.transpose(x, (0, 2, 1, 3)) for x in q_point] # (3, (1, head, L, n_point_qk))
        k_point = [tf.transpose(x, (0, 2, 1, 3)) for x in k_point]
        v_point = [tf.transpose(x, (0, 2, 1, 3)) for x in v_point]

        dist2 = [
            (qx[:, :, :, None, :] - kx[:, :, None, :, :])**2
            for qx, kx in zip(q_point, k_point)
        ]

        dist2 = sum(dist2)
        attn_qk_point = -0.5 * tf.reduce_sum(
            point_weights * dist2, axis=-1)
        attn_logits += attn_qk_point

        attn = tf.nn.softmax(attn_logits)

        output_features = []

        result_scalar = tf.matmul(attn, v)
        result_scalar = tf.transpose(result_scalar, (0, 2, 1, 3))
        result_scalar = tf.reshape(
            result_scalar, (1, L, self.n_head*self.n_scalar_v))

        output_features.append(result_scalar)

        result_point_global = [tf.matmul(attn, vx)
                               for vx in v_point]
        result_point_global = [
            tf.transpose(x, (0, 2, 1, 3))
            for x in result_point_global]
        result_point_global = [
            tf.reshape(r, (1, L, self.n_head*self.n_point_v))
            for r in result_point_global]
        result_point_local = affine.invert_point(result_point_global, extra_dims=1)

        output_features.extend(result_point_local)

        output_features.append(tf.sqrt(
            self.dist_epsilon
            + result_point_local[0]**2
            + result_point_local[1]**2
            + result_point_local[2]**2
        ))

        result_attention_over_2d = tf.einsum("bhij,bijc->bihc", attn, pair)
        output_features.append(
            tf.reshape(result_attention_over_2d,
                      [1, L, self.n_head*self.n_pair_feat]))

        final_act = tf.concat(output_features, axis=-1)
        output = self.fc_output(final_act)

        if return_att:
            return output, tf.transpose(attn, [0,2,3,1])
        else:
            return output

class StructureModule(tf.Module):
    def __init__(self, name, config, n_att_head, dropout, r_ff):
        super(StructureModule, self).__init__(name=name)

        self.n_msa_feat = config['n_1d_feat']
        self.n_structure_msa_feat = config['n_structure_msa_feat']
        self.n_pair_feat = config['n_2d_feat']
        self.n_layers = config['n_str_layers']
        
        self.DTYPE = DTYPE
        self.single_activations = PLinear.Linear(self.n_structure_msa_feat, num_input=self.n_msa_feat, name='single_activations', load=True)
            
        with self.name_scope:
            self.single_layer_norm = PAttention.LayerNorm(input_dim=self.n_structure_msa_feat, name='single_layer_norm', load=True)
            self.attention_layer_norm = PAttention.LayerNorm(input_dim=self.n_structure_msa_feat, name='attention_layer_norm', load=True)
            self.transition_layer_norm = PAttention.LayerNorm(input_dim=self.n_structure_msa_feat, name='transition_layer_norm', load=True)
            self.norm_msa4 = keras.layers.LayerNormalization()
    
            self.pair_layer_norm = PAttention.LayerNorm(input_dim=self.n_pair_feat, name='pair_layer_norm', load=True)
    
            self.dropout1 = keras.layers.Dropout(dropout)
            self.dropout2 = keras.layers.Dropout(dropout)
    
            self.ipa = InvariantPointAttention(name='invariant_point_attention',
                                               n_msa_feat=self.n_structure_msa_feat,
                                               n_pair_feat=self.n_pair_feat,
                                               n_head=n_att_head,
                                               n_scalar_qk=16, n_scalar_v=16,
                                               n_point_qk=4, n_point_v=8,
                                               use_3d=True,
                                               use_struct=False,
                                               use_nanometer=True)
    
            self.transition_1 = PLinear.Linear(self.n_structure_msa_feat, num_input=self.n_structure_msa_feat, name='transition', dtype=self.DTYPE, load=True)
            self.transition_2 = PLinear.Linear(self.n_structure_msa_feat, num_input=self.n_structure_msa_feat, name='transition_1', dtype=self.DTYPE, load=True)
            self.transition_3 = PLinear.Linear(self.n_structure_msa_feat, num_input=self.n_structure_msa_feat, name='transition_2', dtype=self.DTYPE, load=True)
            
            self.affine_update = PLinear.Linear(6, num_input=self.n_structure_msa_feat, name='affine_update', dtype=self.DTYPE, load=True)
            self.initial_projection = PLinear.Linear(self.n_structure_msa_feat, num_input=self.n_structure_msa_feat, name='initial_projection', dtype=self.DTYPE, load=True)

    def __call__(self, msa, pair, training):
        L = msa.shape[1]

        msa = self.single_activations(msa)
        
        msa_init = self.single_layer_norm(msa)
        pair = self.pair_layer_norm(pair)
        msa = self.initial_projection(msa_init)

        affine = QuatAffine.generate_new_affine(1, L)

        rots = affine.rotation
        rots = tf.convert_to_tensor(rots)
        trans = affine.translation
        trans = tf.convert_to_tensor(trans)

        lit_coords = tf.tile(tf.constant([[[[-0.525, 1.363, 0.],
                                            [0., 0., 0.],
                                            [1.526, -0., -0.],
                                            [-0.529, -0.774, -1.205]]]]), (1, L, 1, 1))

        pred_backbone = tf.einsum("ijbl,blkj->ikbl", rots, lit_coords) # (3, 3, 1, L) 0.coor 1.atom
        trans_expand = tf.expand_dims(trans, axis=1) # (3, 1, 1, L)
        pred_backbone += trans_expand

        for i in range(self.n_layers):
            rots = tf.stop_gradient(rots)

            affine.rotation = rots
            affine.translation = trans

            msa2, attn_3d = self.ipa(msa, pair, L, affine, pred_backbone=pred_backbone, return_att=True)
            msa += msa2
            msa = self.attention_layer_norm(msa)

            msa += self.transition_3(tf.nn.relu(self.transition_2(tf.nn.relu(self.transition_1(msa)))))
            msa = self.transition_layer_norm(msa)

            new_quat_transl = self.affine_update(msa)

            transl_update = new_quat_transl[:, :, 3:]
            transl_update = tf.einsum("ijbl,blj->ibl", rots, transl_update)
            trans += transl_update

            quat = tf.concat([tf.ones([1, L, 1]), new_quat_transl[:, :, :3]], axis=-1)
            quat /= tf.linalg.norm(quat, axis=-1, keepdims=True)

            rots_update = QuatAffine.quat_to_rot(quat)
            rots_update = tf.convert_to_tensor(rots_update)
            rots = tf.einsum("ijbl,jkbl->ikbl", rots, rots_update)

        return msa
