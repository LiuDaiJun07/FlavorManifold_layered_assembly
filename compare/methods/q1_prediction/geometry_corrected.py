"""Curvature-consistent read-only clone. Original implementation remains untouched."""
import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('compare_corrected_hyperbolic',Path(__file__).resolve().parents[3]/'src/q1/04_hyperbolic.py')
H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
legacy_distance=H.poincare_dist


def log_map0(z,c=1.):
    r=np.linalg.norm(z,axis=1,keepdims=True);sr=np.sqrt(c)*r
    factor=np.divide(2*np.arctanh(np.minimum(sr,H.RAD_CAP)),sr,out=np.full_like(sr,2.),where=sr>1e-15)
    return factor*z


def distance(a,b,c=1.,same=False):
    aa=np.sum(a*a,axis=1)[:,None];bb=np.sum(b*b,axis=1)[None,:]
    diff=np.maximum(aa+bb-2*a@b.T,0)
    den=np.maximum((1-c*aa)*(1-c*bb),1e-30)
    x=2*c*diff/den
    out=np.log1p(x+np.sqrt(x*(x+2)))/np.sqrt(c)
    if same:np.fill_diagonal(out,0)
    return out


def stress_loss_grad(z,target,c=1.,mask=None):
    n=len(z);aa=np.sum(z*z,axis=1);pu=np.maximum(1-c*aa,1e-12)
    s=np.maximum(aa[:,None]+aa[None,:]-2*z@z.T,0)
    x=2*c*s/(pu[:,None]*pu[None,:]);arg=1+x
    dist=np.log1p(x+np.sqrt(x*(x+2)))/np.sqrt(c)
    w=1/(target*target+1e-6)
    if mask is not None:w*=mask
    np.fill_diagonal(w,0)
    residual=(dist-target)*w;loss=float(np.sum(residual*(dist-target)))/(n*n)
    gd=2*residual/(n*n)
    coef=(gd+gd.T)/(np.sqrt(np.maximum(x*(x+2),1e-30))*np.sqrt(c))
    np.fill_diagonal(coef,0)
    diff=z[:,None,:]-z[None,:,:]
    num=4*c*(diff*pu[:,None,None]+c*s[:,:,None]*z[:,None,:])
    grad=np.sum(coef[:,:,None]*num/(pu[:,None]**2*pu[None,:])[:,:,None],axis=1)
    return loss,grad


H.log_map0=log_map0
H.poincare_dist=distance
H.stress_loss_grad=stress_loss_grad
