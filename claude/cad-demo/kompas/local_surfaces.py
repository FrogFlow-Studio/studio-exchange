"""Scan-supported changes to the upper wall and pocket cutters."""
import numpy as np
import shapely
import trimesh
from scipy.spatial import cKDTree
from shapely.geometry import Polygon


def upper_wall(scan, levels, ring, anchor_z, end_z):
    """Extend the outer wall from side-facing samples on incomplete sections."""
    if end_z-anchor_z<.5:
        return None
    ring = np.asarray(ring)
    xy = ring-np.roll(ring,1,axis=0)
    tangent = np.roll(ring,-1,axis=0)-np.roll(ring,1,axis=0)
    normals = tangent[:,::-1]*[1,-1]
    area = np.sum(ring[:,0]*xy[:,1]-ring[:,1]*xy[:,0])
    normals *= np.sign(area)
    length = np.linalg.norm(normals,axis=1)
    if np.min(length)<1e-6:
        return None
    normals /= length[:,None]
    p, ids = trimesh.sample.sample_surface(scan,40000,seed=64)
    _, nearest = cKDTree(ring).query(p[:,:2])
    # Project onto the two incident segments rather than their shared
    # vertex; sparse rings otherwise bias samples between stations.
    starts = np.column_stack(((nearest-1)%len(ring),nearest))
    ends = (starts+1)%len(ring)
    a = ring[starts]
    ab = ring[ends]-a
    fraction = np.clip(np.einsum("ijk,ijk->ij",p[:,None,:2]-a,ab)/
                       np.maximum(np.sum(ab*ab,axis=2),1e-12),0.,1.)
    delta = p[:,None,:2]-(a+fraction[:,:,None]*ab)
    length2 = np.sum(delta*delta,axis=2)
    choice = np.argmin(length2,axis=1)
    ii = np.arange(len(p))
    distance = np.sqrt(length2[ii,choice])
    n = scan.face_normals[ids]
    heights = p[:,2]-anchor_z
    offsets = np.einsum("ij,ij->i",delta[ii,choice],normals[nearest])
    mask = ((heights>-.65)&(heights<end_z-anchor_z)&(distance<1.35)
            &(abs(offsets)<.65)&(abs(n[:,2])<.65)
            &(np.einsum("ij,ij->i",n[:,:2],normals[nearest])>.6))
    slopes = np.zeros(len(ring))
    supported = np.zeros(len(ring),dtype=bool)
    for i in range(len(ring)):
        j = abs(nearest-i)
        use = mask & (np.minimum(j,len(ring)-j)<=1)
        low = use & (heights<=.1)
        high = use & (heights>=.23)
        if low.sum()<6 or high.sum()<6:
            continue
        dz = np.median(heights[high])-np.median(heights[low])
        slope = (np.median(offsets[high])-np.median(offsets[low]))/dz
        slopes[i] = np.clip(slope,-.5,.25)
        supported[i] = True
    if supported.sum()<3:
        return None
    # Smooth cyclically; unsupported parts retain the stock extension.
    for _ in range(4):
        slopes = .5*slopes+.25*(np.roll(slopes,1)+np.roll(slopes,-1))
    shift = np.clip(slopes*(end_z-anchor_z),-.55,.35)
    if max(abs(shift))<.045:
        return None
    extended = ring+shift[:,None]*normals
    if not Polygon(extended).is_valid:
        return None
    return [*[(p,z) for p,z in levels if z<anchor_z-.25],(ring,anchor_z),(extended,end_z)]


def floor_samples(scan):
    points, ids = trimesh.sample.sample_surface(scan,50000,seed=65)
    return points[scan.face_normals[ids,2]>.68]


def pocket_floor(scan, tool, *, samples=None):
    """Fit a shallow height field only with support across the cutter floor."""
    import kompas_build as stock

    caps = [f for f in tool.Faces() if stock._nz(f)<-.65]
    if not caps:
        return None
    cap = min(caps,key=lambda f:f.Center().z)
    points = cap.outerWire().sample(96)[0]
    poly = Polygon([(p.x,p.y) for p in points])
    if not poly.is_valid or poly.area<12:
        return None
    region = poly.buffer(-.18)
    if region.is_empty:
        return None
    z0 = cap.Center().z
    p = floor_samples(scan) if samples is None else samples
    p = p[(p[:,2]>z0-.35)&(p[:,2]<z0+2.2)]
    p = p[shapely.contains_xy(region,p[:,0],p[:,1])]
    if len(p)<70:
        return None
    from shapely.geometry import MultiPoint

    if MultiPoint(p[:,:2]).convex_hull.area<.65*region.area:
        return None
    x0,y0,x1,y1 = poly.bounds
    origin = np.array([(x0+x1)/2,(y0+y1)/2])
    scale = np.array([(x1-x0)/2,(y1-y0)/2])
    xy = (p[:,:2]-origin)/scale
    design = stock._basis(*xy.T,6)
    if np.linalg.cond(design)>30:
        return None
    weights = np.ones(len(p))
    for _ in range(4):
        coef = np.linalg.lstsq(design*weights[:,None],p[:,2]*weights,rcond=None)[0]
        residual = p[:,2]-design@coef
        weights = np.sqrt(np.minimum(1.,.07/np.maximum(abs(residual),1e-9)))
    if np.quantile(abs(residual),.9)>.16:
        return None

    def evaluate(x,y):
        return stock._basis((x-origin[0])/scale[0],(y-origin[1])/scale[1],6)@coef

    grid = evaluate(np.array([x0,x1,x1,x0]),np.array([y0,y0,y1,y1]))
    if np.min(grid)<z0-.25 or np.max(grid)>z0+2.3 or np.ptp(design@coef)<.09:
        return None
    return evaluate


def trim_pocket(scan, tool, *, samples=None):
    """Trim the lower end of a cutter to a scan-supported smooth surface."""
    import kompas_build as stock

    floor = pocket_floor(scan,tool,samples=samples)
    if floor is None:
        return None
    box = tool.BoundingBox()
    trimmed = stock.cut_above(tool,floor,(box.xmin,box.ymin,box.xmax,box.ymax),below=True)
    if trimmed.isValid() and len(trimmed.Solids())==1 and .5*tool.Volume()<trimmed.Volume()<tool.Volume():
        return trimmed
    return None
