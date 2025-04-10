from vpython import *
import numpy as np
import time
G = -9.8
fps = 30
resolutionMultiplier = 30
dt = (1/fps) / resolutionMultiplier

def alignVectors(a, b):
    b = b / np.linalg.norm(b) # normalize a
    a = a / np.linalg.norm(a) # normalize b
    v = np.cross(a, b)
    # s = np.linalg.norm(v)
    c = np.dot(a, b)
    if np.isclose(c, -1.0):
        return -np.eye(3, dtype=np.float64)

    v1, v2, v3 = v
    h = 1 / (1 + c)

    Vmat = np.array([[0, -v3, v2],
                  [v3, 0, -v1],
                  [-v2, v1, 0]])

    R = np.eye(3, dtype=np.float64) + Vmat + (Vmat.dot(Vmat) * h)
    return R

# Abstract
class Object:
    def __init__(self, pos=np.zeros(3)):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.zeros(3)
    
    def show(self):
        self.visual.pos.x = self.pos[0]
        self.visual.pos.y = self.pos[1]
        self.visual.pos.z = self.pos[2]
        
class Sphere(Object):
    def __init__(self, pos=np.zeros(3), radius=1, color=vec(0.4, 0.4, 0.4), make_trail=False):
        super().__init__(pos=pos)
        self.radius = radius
        self.visual = sphere(make_trail = make_trail, pos=vec(0,0,0), radius=radius, color=color)
        self.show()
        
    def collideSphere(self, other):
        displacement = other.pos - self.pos
        if np.linalg.norm(displacement) < self.radius + other.radius:
            collisionNormal = displacement/np.linalg.norm(displacement)
            return collisionNormal, collisionNormal*self.radius
        return None, None
        
class Box(Object):
    def __init__(self, pos=np.zeros(3), size=np.ones(3), axis=[1,0,0], angle=0, color=vec(0.4, 0.4, 0.4), showNormals=False):
        super().__init__(pos=pos)
        self.size = np.array(size, dtype=float)
        self.axis = np.array(axis, dtype=float)
        self.axis /= np.linalg.norm(self.axis)
        self.rotation = alignVectors(np.array([1,0,0]), self.axis)
        self.normals = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=float)
        for i in range(6):
            self.normals[i] = np.dot(self.rotation, self.normals[i])
        if showNormals:
            for normal in self.normals:
                alignedDiagonal = np.dot(self.rotation, self.size)
                rahPos = self.pos+alignedDiagonal/2
                arrow(pos=vec(rahPos[0],rahPos[1],rahPos[2]), axis=vec(normal[0],normal[1],normal[2]))
        self.angle = angle
        
        self.visual = box(pos=vec(0,0,0), size=vec(size[0],size[1],size[2]), axis=vec(axis[0],axis[1],axis[2]), color=color)
        self.visual.rotate(angle)
        self.show()
        
    def collideSphere(self, other):
        # the dot product is a neat way of finding minimum distance between point and plane
        minDistances = [np.abs(np.dot(other.pos - (self.pos+self.size[int(i/2)]*self.normals[i]*(0 if i%2==1 else 1)), self.normals[i])) for i in range(6)]
        for i in range(6):
            if minDistances[i] < other.radius:
                wallsToCheck = [j for j in range(6) if j != i and j != i+(1 if i%2==0 else -1)]
                outsideWalls = False
                for j in wallsToCheck:
                    if minDistances[j] > self.size[int(j/2)]:
                        # print(str(j) + " | " + str(minDistances))
                        outsideWalls = True
                if not outsideWalls:
                    collisionPoint = other.pos - self.normals[i]*minDistances[i]
                    return self.normals[i], collisionPoint
        return None, None
        
    def update(self):
        pass
    
    def show(self):
        alignedDiagonal = np.dot(self.rotation, self.size)
        self.pos += alignedDiagonal/2
        super().show()
        self.pos -= alignedDiagonal/2
        
        
class Ball(Sphere):
    def __init__(self, pos=np.zeros(3), make_trail=False):
        super().__init__(pos=pos, radius=0.02, color=vec(1,1,1), make_trail=make_trail)
        self.restitution = 0.8
        
    def update(self, collideables=[]):
        self.vel += np.array([0,1,0])*G*dt
        self.pos += self.vel*dt
        
        
        for obj in collideables:
            collisionNormal, collisionPoint = obj.collideSphere(self)
            if type(collisionNormal)==list:
                for i in range(len(collisionNormal)):
                    if np.all(collisionNormal[i] != None):
                        rotMat = alignVectors(self.vel*np.array([-1,-1,-1]), collisionNormal[i])
                        self.vel = np.linalg.norm(self.vel)*np.dot(rotMat, collisionNormal[i])
                        restitutionAdjust = self.vel*collisionNormal[i]*(1-self.restitution)
                        if np.dot(self.vel, restitutionAdjust) > 0:
                            self.vel -= restitutionAdjust
                        else:
                            self.vel += restitutionAdjust
                        self.pos = collisionPoint[i] + self.radius*collisionNormal[i]
            elif np.all(collisionNormal != None):
                rotMat = alignVectors(self.vel*np.array([-1,-1,-1]), collisionNormal)
                self.vel = np.linalg.norm(self.vel)*np.dot(rotMat, collisionNormal)
                restitutionAdjust = self.vel*collisionNormal*(1-self.restitution)
                if np.dot(self.vel, restitutionAdjust) > 0:
                    self.vel -= restitutionAdjust
                else:
                    self.vel += restitutionAdjust
                self.pos = collisionPoint + self.radius*collisionNormal
                
        if self.pos[1] < 0:
            self.pos[1] = 0
            self.vel[1] *= -self.restitution
        self.show()
                
        
class Table(Box):
    def __init__(self, pos=np.zeros(3)):
        super().__init__(pos=pos, size=[2.743, 0.752, 1.524], color=vec(0.3, 0.3, 0.6))

class Net(Box):
    def __init__(self, pos=np.zeros(3)):
        super().__init__(pos=pos, size=[0.005, 0.1525, 1.83], color=vec(0.7, 0.7, 0.7))

class PingPongTable():
    def __init__(self, pos=np.zeros(3)):
        self.table = Table(pos=pos)
        self.net = Net(pos=[pos[0]+self.table.size[0]/2,pos[1]+self.table.size[1],pos[2]-0.15])
    def collideSphere(self, other):
        tableNormal, tablePoint = self.table.collideSphere(other)
        netNormal, netPoint = self.net.collideSphere(other)
        return [tableNormal, netNormal], [tablePoint, netPoint]
    def show(self):
        self.table.show()
        self.net.show()

scene = canvas()
arrow(pos=vec(0,0,0), axis=vec(1,0,0), color=vec(1,0,0))
arrow(pos=vec(0,0,0), axis=vec(0,1,0), color=vec(0,1,0))
arrow(pos=vec(0,0,0), axis=vec(0,0,1), color=vec(0,0,1))
# rah = Sphere(radius=1, pos=[0.5, 0, 0.5])
ball = Ball(pos=[0.2, 5, 0.4], make_trail=True)
ball.vel[0]=5.0
ball.vel[2]=0.07
collideables = []

table = Table(pos =[0,0,0.2])
box1 = Box(pos = [0,0,0], size=[4, 0.4, 4], axis=[-1,1,0], angle=0, showNormals=False)
box2 = Box(pos = [0,0,0], size=[4, 0.4, 4], axis=[1,1,-0.05], angle=0, showNormals=False, color=vec(0.6, 0.4, 0.4))
ball.vel[0]=0.
ball.vel[2]=0.
collideables.append(box1)
collideables.append(box2)

# table = PingPongTable()
# collideables.append(table)
# collideables.append(Box(pos=[-1,0,0], size=[1,6,3]))
# collideables.append(Box(pos=[table.table.size[0],0,0], size=[1,6,3]))


t = 0
while t < 15:
    # print("ALIVE")
    startTime = time.time()
    for _ in range(resolutionMultiplier):
        ball.update(collideables=collideables)
    endTime = time.time()
    # print("UPDATE TIME: " + str(endTime-startTime))
    # print("STILL ALIVE")
    
    t += dt*resolutionMultiplier
    rate(fps)
