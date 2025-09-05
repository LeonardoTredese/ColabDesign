import os
import jax
import jax.numpy as jnp
import numpy as np
from inspect import signature

from colabdesign.af.alphafold.model import data, config, model, all_atom

from colabdesign.shared.model import design_model
from colabdesign.shared.utils import Key

from colabdesign.af.prep   import _af_prep
from colabdesign.af.loss   import _af_loss, get_plddt, get_pae, get_ptm
from colabdesign.af.loss   import get_contact_map, get_seq_ent_loss, get_mlm_loss
from colabdesign.af.utils  import _af_utils
from colabdesign.af.design import _af_design
from colabdesign.af.inputs import _af_inputs, update_seq, update_aatype
from colabdesign.af.loss import (_af_loss, get_plddt, get_pae, get_ptm,
                                 get_contact_map, get_seq_ent_loss,
                                 get_mlm_loss, get_losses)
from colabdesign.af.inputs import (_af_inputs,
                                   _get_seq,
                                   _update_template,
                                   update_seq, update_aatype)
from ml_collections.config_dict import FrozenConfigDict
from functools import partial

################################################################
# MK_DESIGN_MODEL - initialize model, and put it all together
################################################################

def _af_model_kernel(params, model_params, inputs, key, cfg, runner):
    args = cfg["args"]
    callbacks = cfg["callbacks"]
    protocol  = cfg["protocol"]
    keyseq = Key(key=key).get
    inputs["params"] = params
    opt   = inputs["opt"]
    aux   = {}
    seq  = _get_seq(inputs, {"num": args["num"], "shuffle_first": args["shuffle_first"], "copies": args["copies"], "opt": opt, "target_len": cfg["target_len"], "protocol": protocol, **cfg["seq_cfg"]}, aux, key=keyseq())
    pssm = jnp.where(opt["pssm_hard"], seq["hard"], seq["pseudo"])
    if args["use_mlm"]:
        mlm_mask = jax.random.bernoulli(keyseq(), opt["mlm_dropout"], seq["pseudo"].shape[:2])
        update_seq(seq["pseudo"], inputs, seq_pssm=pssm, mlm=mlm_mask)
    else:
        update_seq(seq["pseudo"], inputs, seq_pssm=pssm)
    update_aatype(seq["pseudo"][0].argmax(-1), inputs)
    inputs["msa_mask"] = jnp.where(inputs["seq_mask"], inputs["msa_mask"], 0)
    inputs["seq"] = aux["seq"]
    inputs["mask_template_interchain"] = opt["template"]["rm_ic"]
    if args["use_templates"]:
        inputs = _update_template(inputs, cfg["tmpl_cfg"])
    inputs["use_dropout"] = opt["dropout"]
    if "batch" not in inputs:
        inputs["batch"] = None
    for fn in callbacks["pre"]:
        fn_args  = {"inputs": inputs, "opt": opt, "aux": aux,
                    "seq": seq, "key": keyseq(), "params": params}
        fn(**{k: fn_args[k] for k in signature(fn).parameters})
    outputs = runner.apply(model_params, keyseq(), inputs)
    aux.update({
        "atom_positions": outputs["structure_module"]["final_atom_positions"],
        "atom_mask": outputs["structure_module"]["final_atom_mask"],
        "residue_index": inputs["residue_index"],
        "aatype": inputs["aatype"],
        "plddt": get_plddt(outputs),
        "pae": get_pae(outputs),
        "ptm": get_ptm(inputs, outputs),
        "i_ptm": get_ptm(inputs, outputs, interface=True),
        "cmap": get_contact_map(outputs, opt["con"]["cutoff"]),
        "i_cmap": get_contact_map(outputs, opt["i_con"]["cutoff"]),
        "prev": outputs["prev"]
    })

    aux["losses"] = {}
    get_losses(protocol, FrozenConfigDict(inputs), FrozenConfigDict(outputs), cfg["loss_cfg"], aux)
    aux["losses"].update(get_seq_ent_loss(inputs))
    if args["use_mlm"]:
        aux["mlm"] = outputs["masked_msa"]["logits"]
        mask = jnp.where(inputs["seq_mask"], mlm_mask, 0)
        aux["losses"].update(get_mlm_loss(outputs, mask=mask, truth=seq["pssm"]))
    for fn in callbacks["loss"]:
        aux["losses"].update(fn(inputs=inputs, outputs=outputs,
                                cfg=cfg["loss_cfg"],
                                opt=opt, aux=aux, seq=seq,
                                key=keyseq(), params=params))
    for fn in callbacks["post"]:
        fn(inputs=inputs, outputs=outputs,
           opt=opt, aux=aux, seq=seq,
           key=keyseq(), params=params)
    if args["debug"]: aux["debug"] = {"inputs": inputs, "outputs": outputs}
    w = opt["weights"]
    loss = sum(v * w.get(k, 1.0) for k, v in aux["losses"].items())
    return loss, aux



class mk_af_model(design_model, _af_inputs, _af_loss, _af_prep, _af_design, _af_utils):
  runners = dict()
  _kernel_grad = jax.jit(jax.value_and_grad(_af_model_kernel, has_aux=True, argnums=0), static_argnums=(4, 5))
  _kernel = jax.jit(_af_model_kernel, static_argnums=(4, 5))
  def __init__(self,
               protocol="fixbb",
               use_multimer=False,
               use_templates=False,
               debug=False,
               data_dir=".",
               **kwargs):
    assert protocol in ["fixbb","hallucination","binder","partial"]
    self.protocol = protocol
    self._num = kwargs.pop("num_seq",1)
    self._args = {"use_templates":use_templates, "use_multimer":use_multimer, "use_bfloat16":True,
                  "recycle_mode":"last", "use_mlm": False, "realign": True,
                  "debug":debug, "repeat":False, "homooligomer":False, "copies":1,
                  "optimizer":"sgd", "best_metric":"loss",
                  "traj_iter":1, "traj_max":10000,
                  "clear_prev": True, "use_dgram":False,
                  "shuffle_first":True, "use_remat":True,
                  "alphabet_size":20, "num": self._num,
                  "use_initial_guess":False, "use_initial_atom_pos":False}

    if self.protocol == "binder": self._args["use_templates"] = True

    self.opt = {"dropout":True, "pssm_hard":False, "learning_rate":0.1, "norm_seq_grad":True,
                "num_recycles":0, "num_models":1, "sample_models":True,
                "temp":1.0, "soft":0.0, "hard":0.0, "alpha":2.0,
                "con":      {"num":2, "cutoff":14.0, "binary":False, "seqsep":9, "num_pos":float("inf")},
                "i_con":    {"num":1, "cutoff":21.6875, "binary":False, "num_pos":float("inf")},
                "template": {"rm_ic":False},
                "weights":  {"seq_ent":0.0, "plddt":0.0, "pae":0.0, "exp_res":0.0, "helix":0.0},
                "fape_cutoff":10.0}

    self._params = {}
    self._inputs = {}
    self._tmp = {"traj":{"seq":[],"xyz":[],"plddt":[],"pae":[]},
                 "log":[],"best":{}}
    if "initial_guess" in kwargs: kwargs["use_initial_guess"] = kwargs.pop("initial_guess")
    model_names = kwargs.pop("model_names",None)
    keys = list(kwargs.keys())
    for k in keys:
      if k in self._args: self._args[k] = kwargs.pop(k)
      if k in self.opt: self.opt[k] = kwargs.pop(k)
    self._callbacks = {"model": {"pre": kwargs.pop("pre_callback",None),
                                 "post":kwargs.pop("post_callback",None),
                                 "loss":kwargs.pop("loss_callback",None)},
                       "design":{"pre": kwargs.pop("pre_design_callback",None),
                                 "post":kwargs.pop("post_design_callback",None)}}

    for m,n in self._callbacks.items():
      for k,v in n.items():
        if v is None: v = []
        if not isinstance(v,list): v = [v]
        self._callbacks[m][k] = v

    if self._args["use_mlm"]:
      self.opt["mlm_dropout"] = 0.15
      self.opt["weights"]["mlm"] = 0.1

    assert len(kwargs) == 0, f"ERROR: the following inputs were not set: {kwargs}"

    #############################
    # configure AlphaFold
    #############################
    if self._args["use_multimer"]:
      self._cfg = config.model_config("model_1_multimer")
      # TODO
      self.opt["pssm_hard"] = True
    else:
      self._cfg = config.model_config("model_1_ptm" if self._args["use_templates"] else "model_3_ptm")

    if self._args["recycle_mode"] in ["average","first","last","sample"]:
      num_recycles = 0
    else:
      num_recycles = self.opt["num_recycles"]
    self._cfg.model.num_recycle = num_recycles
    self._cfg.model.global_config.use_remat = self._args["use_remat"]
    self._cfg.model.global_config.use_dgram = self._args["use_dgram"]
    self._cfg.model.global_config.bfloat16  = self._args["use_bfloat16"]

    # load model_params
    if model_names is None:
      model_names = []
      if self._args["use_multimer"]:
        model_names += [f"model_{k}_multimer_v3" for k in [1,2,3,4,5]]
      else:
        if self._args["use_templates"]:
          model_names += [f"model_{k}_ptm" for k in [1,2]]
        else:
          model_names += [f"model_{k}_ptm" for k in [1,2,3,4,5]]

    self._model_params, self._model_names = [],[]
    for model_name in model_names:
      params = data.get_model_haiku_params(model_name=model_name, data_dir=data_dir, fuse=True)
      if params is not None:
        if not self._args["use_multimer"] and not self._args["use_templates"]:
          params = {k:v for k,v in params.items() if "template" not in k}
        self._model_params.append(params)
        self._model_names.append(model_name)
      else:
        print(f"WARNING: '{model_name}' not found")

    #####################################
    # set protocol specific functions
    #####################################
    idx = ["fixbb","hallucination","binder","partial"].index(self.protocol)
    self.prep_inputs = [self._prep_fixbb, self._prep_hallucination, self._prep_binder, self._prep_partial][idx]
    self._get_loss = [self._loss_fixbb, self._loss_hallucination, self._loss_binder, self._loss_partial][idx]

  def _get_model(self, cfg, callback=None):
    static_cfg = FrozenConfigDict({
        "args": FrozenConfigDict(self._args),
        "protocol": self.protocol,
        "target_len": getattr(self, "_target_len", None),
        "callbacks": FrozenConfigDict({
            "pre": tuple(self._callbacks["model"]["pre"]),
            "loss": tuple(self._callbacks["model"]["loss"]),
            "post": tuple(self._callbacks["model"]["post"]),
        }),
        "seq_cfg" : FrozenConfigDict({
            "alphabet_size": self._args["alphabet_size"],
        }),
        "tmpl_cfg": FrozenConfigDict({
            "protocol"      : self.protocol,
            "repeat"        : self._args["repeat"],
            "homooligomer"  : self._args["homooligomer"],
            "copies"        : self._args["copies"],
            "length"        : getattr(self, "_len", None),
        }),
        "loss_cfg": self._make_loss_cfg()
    })
    rargs = FrozenConfigDict({'cfg': cfg, 'recycle_mode': self._args["recycle_mode"], 'use_multimer': self._args["use_multimer"]})
    if rargs not in mk_af_model.runners: mk_af_model.runners[rargs] = model.RunModel(rargs['cfg'], recycle_mode=rargs["recycle_mode"], use_multimer=rargs["use_multimer"])
    runner = mk_af_model.runners[rargs]
    return {
        "grad_fn": partial(mk_af_model._kernel_grad, runner=runner, cfg=static_cfg),
        "fn"    : partial(mk_af_model._kernel, runner=runner, cfg=static_cfg),
        "runner": runner,
    }

