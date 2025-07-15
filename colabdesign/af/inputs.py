import jax
import jax.numpy as jnp
import numpy as np

from colabdesign.shared.utils import copy_dict
from colabdesign.shared.model import soft_seq
from colabdesign.af.alphafold.common import residue_constants
from colabdesign.af.alphafold.model import model, config

############################################################################
# AF_INPUTS - functions for modifying inputs before passing to alphafold
############################################################################

def _fix_pos(seq, cfg, return_p: bool = False):
    if "fix_pos" not in cfg["opt"]: return seq
    if "pos" in cfg["opt"]:
        p = opt["pos"][opt["fix_pos"]]
        seq_ref = jax.nn.one_hot(cfg["wt_aatype_sub"], cfg["alphabet_size"])
        rewrite = lambda x: x.at[..., p, :].set(seq_ref)
    else:
        p = opt["fix_pos"]
        seq_ref = jax.nn.one_hot(cfg["wt_aatype"], cfg["alphabet_size"])
        rewrite = lambda x: x.at[..., p, :].set(seq_ref[..., p, :])
    seq = jax.tree_util.tree_map(rewrite, seq)
    return (seq, p) if return_p else seq

def _get_seq(inputs, cfg, aux, key=None):
    params, bias, opt = inputs["params"], inputs["bias"], inputs["opt"]
    seq = soft_seq(params["seq"], bias, opt, key, num_seq = cfg["num"], shuffle_first = cfg["shuffle_first"])
    seq = _fix_pos(seq, cfg)
    aux.update({"seq": seq, "seq_pseudo": seq["pseudo"]})
    if cfg["protocol"] == "binder":
        tL = cfg["target_len"]
        seq_target = jax.nn.one_hot(inputs["batch"]["aatype"][:tL], cfg["alphabet_size"])
        seq_target = jnp.broadcast_to(seq_target, (cfg["num"], *seq_target.shape))
        seq = jax.tree_util.tree_map(lambda x: jnp.concatenate([seq_target, x], 1), seq)
    if cfg["protocol"] in {"fixbb", "hallucination", "partial"} and cfg["copies"] > 1:
        seq = jax.tree_util.tree_map(
            lambda x: expand_copies(x, cfg["copies"], cfg["block_diag"]),
            seq,
        )
    return seq

def _update_template(inputs, cfg, key=None):
    if "batch" not in inputs: return inputs
    out = inputs
    batch, opt = out["batch"], out["opt"]
    out["template_mask"] = out["template_mask"].at[0].set(1)
    L = batch["aatype"].shape[0]
    rm = jnp.broadcast_to(out.get("rm_template", False), L)
    rm_seq  = jnp.where(rm, True, jnp.broadcast_to(out.get("rm_template_seq", True), L))
    rm_sc   = jnp.where(rm_seq, True, jnp.broadcast_to(out.get("rm_template_sc", True), L))
    tmpl = {"template_aatype": jnp.where(rm_seq, 21, batch["aatype"])}
    if "dgram" in batch:
        tmpl["template_dgram"] = batch["dgram"]
        nT, nL = out["template_aatype"].shape
        out["template_dgram"] = jnp.zeros((nT, nL, nL, 39))
    if "all_atom_positions" in batch:
        cb, cb_mask = model.modules.pseudo_beta_fn(
            jnp.where(rm_seq, 0, batch["aatype"]),
            batch["all_atom_positions"],
            batch["all_atom_mask"],
        )
        tmpl.update({
            "template_pseudo_beta"      : cb,
            "template_pseudo_beta_mask" : cb_mask,
            "template_all_atom_positions": batch["all_atom_positions"],
            "template_all_atom_mask"     : batch["all_atom_mask"],
        })
    prot = cfg["protocol"]
    if prot == "partial":
        pos = opt["pos"]
        if cfg["repeat"] or cfg["homooligomer"]:
            C, L0 = cfg["copies"], cfg["length"]
            pos   = (jnp.repeat(pos, C).reshape(-1, C) + jnp.arange(C) * L0).T.flatten()
    for k, v in tmpl.items():
        if prot == "partial":
            if k == "template_dgram":
                out[k] = out[k].at[0, pos[:, None], pos[None, :]].set(v)
            else:
                out[k] = out[k].at[0, pos].set(v)
        else:
            out[k] = out[k].at[0].set(v)

    if "template_all_atom_mask" in tmpl:
        if prot == "partial":
            out["template_all_atom_mask"] = (
                out["template_all_atom_mask"]
                .at[:, pos, 5:]
                .set(jnp.where(rm_sc[:, None], 0, out["template_all_atom_mask"][:, pos, 5:]))
            )
            out["template_all_atom_mask"] = (
                out["template_all_atom_mask"]
                .at[:, pos]
                .set(jnp.where(rm[:, None], 0, out["template_all_atom_mask"][:, pos]))
            )
        else:
            out["template_all_atom_mask"] = (
                out["template_all_atom_mask"]
                .at[..., 5:]
                .set(jnp.where(rm_sc[:, None], 0, out["template_all_atom_mask"][..., 5:]))
            )
            out["template_all_atom_mask"] = jnp.where(
                rm[:, None], 0, out["template_all_atom_mask"]
            )

    return out

def update_seq(seq, inputs, seq_1hot=None, seq_pssm=None, mlm=None):
  '''update the sequence features'''

  if seq_1hot is None: seq_1hot = seq
  if seq_pssm is None: seq_pssm = seq
  target_feat = seq_1hot[0,:,:20]

  seq_1hot = jnp.pad(seq_1hot,[[0,0],[0,0],[0,22-seq_1hot.shape[-1]]])
  seq_pssm = jnp.pad(seq_pssm,[[0,0],[0,0],[0,22-seq_pssm.shape[-1]]])
  msa_feat = jnp.zeros_like(inputs["msa_feat"]).at[...,0:22].set(seq_1hot).at[...,25:47].set(seq_pssm)

  # masked language modeling (randomly mask positions)
  if mlm is not None:
    X = jax.nn.one_hot(22,23)
    X = jnp.zeros(msa_feat.shape[-1]).at[...,:23].set(X).at[...,25:48].set(X)
    msa_feat = jnp.where(mlm[...,None],X,msa_feat)

  inputs.update({"msa_feat":msa_feat, "target_feat":target_feat})

def update_aatype(aatype, inputs):
  r = residue_constants
  a = {"atom14_atom_exists":r.restype_atom14_mask,
       "atom37_atom_exists":r.restype_atom37_mask,
       "residx_atom14_to_atom37":r.restype_atom14_to_atom37,
       "residx_atom37_to_atom14":r.restype_atom37_to_atom14}
  mask = inputs["seq_mask"][:,None]
  inputs.update(jax.tree_util.tree_map(lambda x:jnp.where(mask,jnp.asarray(x)[aatype],0),a))
  inputs["aatype"] = aatype

def expand_copies(x, copies, block_diag=True):
  '''
  given msa (N,L,20) expand to (1+N*copies,L*copies,22) if block_diag else (N,L*copies,22)
  '''
  if x.shape[-1] < 22:
    x = jnp.pad(x,[[0,0],[0,0],[0,22-x.shape[-1]]])
  x = jnp.tile(x,[1,copies,1])
  if copies > 1 and block_diag:
    L = x.shape[1]
    sub_L = L // copies
    y = x.reshape((-1,1,copies,sub_L,22))
    block_diag_mask = jnp.expand_dims(jnp.eye(copies),(0,3,4))
    seq = block_diag_mask * y
    gap_seq = (1-block_diag_mask) * jax.nn.one_hot(jnp.repeat(21,sub_L),22)
    y = (seq + gap_seq).swapaxes(0,1).reshape(-1,L,22)
    return jnp.concatenate([x[:1],y],0)
  else:
    return x


class _af_inputs:

   def _make_input_cfg(self, inputs):
      return {
          "opt"            : inputs["opt"],
          "num"            : self._num,
          "shuffle_first"  : self._args["shuffle_first"],
          "protocol"       : self.protocol,
          "target_len"     : getattr(self, "_target_len", None),
          "copies"         : self._args["copies"],
          "block_diag"     : self._args.get("block_diag", None),
          "alphabet_size"  : self._args["alphabet_size"],
          "wt_aatype_sub"  : getattr(self, "_wt_aatype_sub", None),
          "wt_aatype"      : getattr(self, "_wt_aatype", None),
      }

   def _get_seq(self, inputs, aux, key=None):
       cfg = self._make_input_cfg(inputs)
       return _get_seq(inputs, cfg, aux, key=key)

   def _fix_pos(self, seq, return_p=False):
       return _fix_pos(seq, self._make_input_cfg({"opt": self.opt}), return_p=return_p)

   def _update_template(self, inputs, key):
       return _update_template(inputs, self._make_input_cfg(inputs), key=key)

