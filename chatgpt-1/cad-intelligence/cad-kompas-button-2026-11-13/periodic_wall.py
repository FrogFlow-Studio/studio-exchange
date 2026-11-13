"""Task 56. Fit a periodic transverse profile and a regularized axial field."""
import cadquery as cq
import numpy as np
from wall_profiles import fit_profiles


def periodic_wall(contours, z, *, count=96, modes=28, axial_penalty=.008, degree=None):
    """Use common rays and smooth the radial field in both directions."""
    z = np.asarray(z,dtype=float)
    filtered = fit_profiles(contours,z,count=count,modes=modes,axial_penalty=axial_penalty)
    if filtered is None:
        return None
    take = np.unique(np.linspace(0,len(z)-1,min(9,len(z))).round().astype(int))
    wires = []
    for i in take:
        edge = cq.Edge.makeSpline([cq.Vector(float(x),float(y),float(z[i])) for x,y in filtered[i]],
                                  periodic=True)
        wires.append(cq.Wire.assembleEdges([edge]))
    try:
        if degree is None:
            body = cq.Solid.makeLoft(wires)
        else:
            from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections

            tool = BRepOffsetAPI_ThruSections(True,False,1e-6)
            tool.SetMaxDegree(degree)
            tool.CheckCompatibility(False)
            for wire in wires:
                tool.AddWire(wire.wrapped)
            tool.Build()
            body = cq.Solid(tool.Shape())
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
