"""Task 56. Fit a periodic transverse profile and a regularized axial field."""
import cadquery as cq
import numpy as np
from scipy.interpolate import make_smoothing_spline

from ring_fit import correspond


def periodic_wall(contours, z, *, count=96, modes=28, axial_penalty=.008):
    """Use common rays and smooth the radial field in both directions."""
    z = np.asarray(z,dtype=float)
    rings = correspond(contours,z,count=count,smooth=0.)
    if rings is None:
        return None
    center = rings.mean(axis=(0,1))
    coefficients = np.fft.rfft(rings-center,axis=1)
    frequency = np.arange(coefficients.shape[1])
    weights = 1/(1+(frequency/max(1,modes))**8)
    coefficients *= weights[None,:,None]
    filtered = np.fft.irfft(coefficients,n=count,axis=1)+center
    shape = filtered.shape
    flat = filtered.reshape(len(z),-1)
    filtered = np.column_stack([make_smoothing_spline(z,col,lam=axial_penalty)(z)
                                for col in flat.T]).reshape(shape)
    take = np.unique(np.linspace(0,len(z)-1,min(9,len(z))).round().astype(int))
    wires = []
    for i in take:
        edge = cq.Edge.makeSpline([cq.Vector(float(x),float(y),float(z[i])) for x,y in filtered[i]],
                                  periodic=True)
        wires.append(cq.Wire.assembleEdges([edge]))
    try:
        body = cq.Solid.makeLoft(wires)
        if body.isValid() and len(body.Solids())==1:
            return body,filtered
    except (ValueError,RuntimeError):
        pass
    return None


def curvatures(rings,z):
    """Curvature of the sampled longitudinal and periodic transverse curves."""
    velocity = np.gradient(rings,z,axis=0)
    speed = np.concatenate((velocity,np.ones((*velocity.shape[:2],1))),axis=2)
    acc = np.gradient(speed,z,axis=0)
    axial = np.linalg.norm(np.cross(speed,acc),axis=2)/np.linalg.norm(speed,axis=2)**3
    a = rings-np.roll(rings,1,axis=1)
    b = np.roll(rings,-1,axis=1)-rings
    num = 2*abs(a[:,:,0]*b[:,:,1]-a[:,:,1]*b[:,:,0])
    den = np.linalg.norm(a,axis=2)*np.linalg.norm(b,axis=2)*np.linalg.norm(a+b,axis=2)
    return axial,num/np.maximum(den,1e-12)
