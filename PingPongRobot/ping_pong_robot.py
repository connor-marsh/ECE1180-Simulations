from vpython import *
import numpy as np
import time
import math
import xml.etree.ElementTree as ET
import ikpy.chain
G = -9.8
floorActive = True
fps = 30
speedMultiplier = 1
dt = 0.0005 * speedMultiplier
resolutionMultiplier = int(1.0/(dt*fps)*speedMultiplier)
print(resolutionMultiplier)
preSimulateThenPlayback = False





# Returns a rotation matrix R that will rotate vector a to point in the same direction as vector b
def alignVectorsRotationMatrix(a, b):
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
def axisAngleRotationMatrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis /= np.linalg.norm(axis)
    axis = np.asarray(axis)
    
    a = cos(theta / 2.0)
    b, c, d = -axis * sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
def yawRotationMatrix(theta):
    return np.array([[cos(theta), 0, sin(theta)],
                     [0, 1, 0],
                     [-sin(theta), 0, cos(theta)]])
    
def pitchRotationMatrix(theta):
    return np.array([[cos(theta), -sin(theta), 0],
                     [sin(theta), cos(theta), 0],
                     [0, 0, 1]])
def rollRotationMatrix(theta):
    return np.array([[1, 0, 0],
                     [0, cos(theta), -sin(theta)],
                     [0, sin(theta), cos(theta)]])

# if m2 is negative, it assumes m2 has infinite mass
# so (m1*v1+m2*v2)/(m1+m2) = m2*v2/m2 = v2 as m2 approaches infinity
def collision(m1, v1, m2, v2, alpha):
    vp = (1+alpha)*((m1*v1+m2*v2)/(m1+m2) if m2 > 0 else v2)
    v1p = vp - alpha*v1
    v2p = vp - alpha*v2
    ## These were for testing
    # kei = 0.5*m1*v1**2 + 0.5*m2*v2**2
    # kef = 0.5*m1*v1p**2 + 0.5*m2*v2p**2
    # foundAlpha = kef / kei
    return v1p, v2p#, kei, kef, foundAlpha



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
        
    def move(self, pos=None):
        if np.all(pos != None):
            self.pos = pos
        
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
        self.angle = 0
        self.axisRotation = None
        self.angleRotation = None
        self.visual = box() # make default box for now, it will get redefined in the move
        self.visual.color = color
        self.showNormals = showNormals
        self.arrows = []
        self.move(axis=axis, angle=angle, showNormals=showNormals)
        self.show()
        
    def move(self, pos=None, axis=None, upAxis=None, angle=None, showNormals=None, axisRotation=None, angleRotation=None):
        prevAxis = self.axis
        prevAngle = self.angle
        useUpAxis = False
        if np.all(pos != None):
            self.pos = pos
        if np.all(axis != None):
            self.axis = np.array(axis, dtype=float)
            self.axis /= np.linalg.norm(self.axis)
        elif np.all(upAxis != None):
            useUpAxis = True
        if angle != None:
            self.angle = angle
        if showNormals != None:
            self.showNormals = showNormals
        
        if np.any(self.axis != prevAxis) or np.any(self.angle != prevAngle) or np.all(self.axisRotation == None) or useUpAxis:
            if np.all(axisRotation != None) and np.all(angleRotation != None):
                self.axisRotation = axisRotation
                self.angleRotation = angleRotation
            elif useUpAxis:
                self.axisRotation = alignVectorsRotationMatrix(np.array([0,1,0]), upAxis)
                self.angleRotation = axisAngleRotationMatrix(upAxis, self.angle)
            else:
                self.axisRotation = alignVectorsRotationMatrix(np.array([1,0,0]), self.axis)
                self.angleRotation = axisAngleRotationMatrix(self.axis, self.angle)
            # maybe rework to adjust existing np array instead of creating new one each move
            self.normals = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=float)
            for i in range(6):
                self.normals[i] = np.dot(self.axisRotation, self.normals[i])
                self.normals[i] = np.dot(self.angleRotation, self.normals[i])
            if useUpAxis:
                self.axis = self.normals[0]

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
        
        self.visual.axis.x = self.axis[0]
        self.visual.axis.y = self.axis[1]
        self.visual.axis.z = self.axis[2]
        
        self.visual.up.x = self.normals[2][0]
        self.visual.up.y = self.normals[2][1]
        self.visual.up.z = self.normals[2][2]
        
        self.visual.size.x = self.size[0]
        self.visual.size.y = self.size[1]
        self.visual.size.z = self.size[2]
        
        # alignedDiagonal = self.size
        alignedDiagonal = np.dot(self.axisRotation, self.size)
        alignedDiagonal = np.dot(self.angleRotation, alignedDiagonal)
        self.pos += alignedDiagonal/2
        super().show()
        if self.showNormals:
            if len(self.arrows)==0:
                for normal in self.normals:
                    shortenedNormal = normal/(np.linalg.norm(normal)*4)
                    self.arrows.append(arrow(pos=vec(self.pos[0],self.pos[1],self.pos[2]), axis=vec(shortenedNormal[0],shortenedNormal[1],shortenedNormal[2])))
            else:
                for i in range(len(self.arrows)):
                    shortenedNormal = self.normals[i]/(np.linalg.norm(self.normals[i])*4)
                    self.arrows[i].axis = vec(shortenedNormal[0],shortenedNormal[1],shortenedNormal[2])
                    self.arrows[i].pos = vec(self.pos[0],self.pos[1],self.pos[2])
                    
        self.pos -= alignedDiagonal/2
        
class Ball(Sphere):
    def __init__(self, pos=np.zeros(3), make_trail=False):
        super().__init__(pos=pos, radius=0.02, color=vec(1,1,1), make_trail=make_trail)
        self.restitution = 0.8
        self.mass = 0.0027 # This value is currently irrelevant, because we are assuming all other objects have infinite mass
        
    def update(self, collideables=[]):
        self.vel += np.array([0,1,0])*G*dt
        self.pos += self.vel*dt
        
        for obj in collideables:
            collisionNormal, collisionPoint = obj.collideSphere(self)
            if type(collisionNormal)!=list:
                collisionNormal = [collisionNormal]
                collisionPoint = [collisionPoint]
            for i in range(len(collisionNormal)):
                if np.all(collisionNormal[i] != None):
                    # First get the new velocity if the collision was perfectly elastic
                    rotMat = alignVectorsRotationMatrix(self.vel*np.array([-1,-1,-1]), collisionNormal[i])
                    self.vel = np.linalg.norm(self.vel)*np.dot(rotMat, collisionNormal[i])
                    
                    # Then find how to adjust it based off of inelasticity
                    # We do this by looking at the magnitude of the velocity projected onto the collision normal
                    projectedVelocityMag = np.dot(self.vel, collisionNormal[i])
                    # And then do a 1d collision equation between both objects projected velocities
                    # Which gives us a new velocity in this 1 dimension.
                    if hasattr(obj, "vel"):
                        otherVel = np.dot(obj.vel, collisionNormal[i])
                    else:
                        otherVel = 0 # Stationary
                    # angular vel in collisions WIP
                    if hasattr(obj, "angularVel"):
                        if np.linalg.norm(obj.angularVel) != 0:
                            # we need to find the instantaneous velocity of the collision point
                            # which is equal to the cross product of the vector from object origin to collision point
                            # and the angular velocity of the object
                            instantaneousRotatingVelocity = np.cross(obj.angularVel, collisionPoint[i])

                            # lastly we project the instantenousRotatingVelocity onto the collision normal
                            rotatingCollisionVelocity = np.dot(instantaneousRotatingVelocity, collisionNormal[i])
                            # and superimpose it with the translational velocity
                            otherVel += rotatingCollisionVelocity
                    # we use negative projectedVelocityMag so the collision function thinks they are colliding
                    newProjectedVelocityMag, _ = collision(self.mass, -projectedVelocityMag, -1, otherVel, self.restitution)
                    # print(f"NEWPROJECTVELMAG {newProjectedVelocityMag}")
                    # We take the difference in magnitude from before and after the collision
                    collisionAdjustMag = projectedVelocityMag - newProjectedVelocityMag
                    # Then add the difference to the new velocity along the direction of the collision normal
                    collisionAdjust = collisionAdjustMag * collisionNormal[i]
                    if np.dot(self.vel, collisionNormal[i]) > 0:
                        self.vel -= collisionAdjust
                    else:
                        self.vel += collisionAdjust
                    # print(f"POST COLLISION VEL {self.vel} and UNIT {self.vel/np.linalg.norm(self.vel)}")
                        
                    
                    # Adjust position to snap to collision point
                    self.pos = collisionPoint[i] + self.radius*collisionNormal[i]
        if self.pos[1] < 0 and floorActive:
            self.pos[1] = 0
            self.vel[1] *= -self.restitution
    def findInterception(self, desiredInterceptionHeight):
        timeOfInterception = (self.vel[1] + sqrt(self.vel[1]**2 + 2*G*(desiredInterceptionHeight - self.pos[1]))) / -G
        velocityOfInterception = np.array([self.vel[0], self.vel[1] + G*timeOfInterception, self.vel[2]])
        positionOfInterception = np.array([self.pos[0] + self.vel[0]*timeOfInterception, self.pos[1] + self.vel[1]*timeOfInterception + 0.5*G*timeOfInterception**2, self.pos[2] + self.vel[2]*timeOfInterception])
        return timeOfInterception, positionOfInterception, velocityOfInterception

    # @staticmethod
    # def findInterceptionStatic(pos, vel, desiredInterceptionHeight):
    #     timeOfInterception = (vel[1] + sqrt(vel[1]**2 + 2*G*(desiredInterceptionHeight - pos[1]))) / -G
    #     velocityOfInterception = np.array([vel[0], vel[1] + G*timeOfInterception, vel[2]])
    #     positionOfInterception = np.array([pos[0] + vel[0]*timeOfInterception, pos[1] + vel[1]*timeOfInterception + 0.5*G*timeOfInterception**2, pos[2] + vel[2]*timeOfInterception])
    #     return timeOfInterception, positionOfInterception, velocityOfInterception
    
    def findVelocityToReachPosition(self, desiredPosition, startingPosition=None):
        if np.all(startingPosition != None):
            pos = startingPosition
        else:
            pos = self.pos
        distanceXZ = (desiredPosition - pos)*np.array([1, 0, 1])
        distanceMag = np.linalg.norm(distanceXZ)

        # Just use theta=45 degrees upwards
        # trajectory = np.array([distanceXZ[0], distanceMag, distanceXZ[2]])
        # theta = acos(np.dot(distanceXZ, trajectory)/(np.linalg.norm(trajectory)*np.linalg.norm(distanceXZ)))
        # initialMagnitude = sqrt(abs(0.5*G*(np.linalg.norm(distanceXZ)/cos(theta))**2/(np.linalg.norm(distanceXZ)*tan(theta)+pos[1]-desiredPosition[1])))    
        # return trajectory/np.linalg.norm(trajectory)*initialMagnitude
        
        # Sweep a few angles to find the best theta
        thetas = np.linspace(np.pi/4, np.pi/2, 30)
        numerator = 0.5 * G * (distanceMag / np.cos(thetas)) ** 2
        denominator = distanceMag * np.tan(thetas) + pos[1] - desiredPosition[1]
        mags = np.sqrt(np.abs(numerator / denominator))
        yvels = mags * np.sin(thetas)
        times = (yvels + np.sqrt(yvels ** 2 + 2 * G * (desiredPosition[1] - pos[1]))) / -G
        yvels = mags + G*times
        lowestVel = 99999
        lowestIndex = None
        for i in range(len(yvels)):
            if yvels[i] < 0 and mags[i] < lowestVel:
                lowestVel = mags[i]
                lowestIndex = i
        trajectory = distanceXZ
        rotAxis = np.cross(distanceXZ, np.array([0, 1, 0]))
        rotAxis /= np.linalg.norm(rotAxis)
        rotMat = axisAngleRotationMatrix(rotAxis, thetas[lowestIndex])
        trajectory = np.dot(rotMat, trajectory)


        return trajectory/np.linalg.norm(trajectory)*lowestVel
                
        
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

class PingPongPaddle():
    def __init__(self, pos=np.zeros(3)):
        
        # Object stuff
        handleSize=[0.1, 0.02, 0.03]
        paddleSize=[0.2, 0.02, 0.15]
        # self.handle = Box(pos=[pos[0], pos[1]-handleSize[1]/2, pos[2]-handleSize[2]/2], size=handleSize, showNormals=False)
        # self.paddle = Box(pos=[pos[0]+self.handle.size[0], pos[1]-paddleSize[1]/2, pos[2]-paddleSize[2]/2], size=paddleSize, showNormals=False)
        self.handle = Box(pos=[pos[0], pos[1], pos[2]], size=handleSize, showNormals=False)
        self.paddle = Box(pos=[pos[0]+self.handle.size[0], pos[1], pos[2]-paddleSize[2]/2+self.handle.size[2]/2], size=paddleSize, showNormals=False)
        self.handle.visual.color = vec(.76, .55, .22)
        self.paddle.visual.color = vec(.21, .21, .8)
        
        # Physics stuff
        self.pos = np.array(pos, dtype=float)
        self.axis = self.handle.axis
        self.angle = self.handle.angle
        self.vel = np.zeros(3)
        self.angularVel = np.zeros(3)
        self.metaVel = np.zeros(3)
        self.axisRotation = None
        self.angleRotation = None
        
        # Controls stuff
        self.hitBall = False
        self.ballHit = None
        self.disableVelocity = False
        # self.moveTarget()
        self.targetVisual = sphere(pos=vec(0, 0, 0), radius=0.05, color=vec(1, 0.8, 0))
    
    # deprecated
    # def update(self):
    #     self.pos += self.vel*dt
    #     self.rotation += self.angularVel*dt
    #     self.axis = np.dot(yawRotationMatrix(self.rotation[0]), np.dot(pitchRotationMatrix(self.rotation[1]), np.array([1,0,0])))
    #     self.angle += self.angularVel[2]*dt
    #     self.move()
    def update(self):
        self.vel = self.vel*0
        self.angularVel = self.angularVel*0
        
        if self.hitBall and self.ballHit:
            if self.ballHit.vel[1] > 0.8:
                print("CONTROL")
                t, pos, incomingVel = self.ballHit.findInterception(self.ballHit.pos[1])
                outgoingVel = self.ballHit.findVelocityToReachPosition(self.targetPosition, startingPosition=pos)
                print(f"TARGETVEL {outgoingVel} and UNIT {outgoingVel/np.linalg.norm(outgoingVel)}")
                
                #### Make ball go to target position
                # arrow(pos=vec(pos[0], pos[1], pos[2]), axis=vec(incomingVelUnit[0], incomingVelUnit[1], incomingVelUnit[2]), color=vec(1, 0, 0))
                # arrow(pos=vec(pos[0], pos[1], pos[2]), axis=vec(outgoingVelUnit[0], outgoingVelUnit[1], outgoingVelUnit[2]), color=vec(0, 1, 0))
                
                print("ATTEMPTING NUMERICAL SOLUTION")
                ###### Guess a normal (upAxis), then find paddleMagnitude
                incomingVelUnit = -1*incomingVel / np.linalg.norm(incomingVel)
                outgoingVelUnit = outgoingVel / np.linalg.norm(outgoingVel)
                error = 99999
                threshold = 0.01
                attempts = 0
                maxAttempts = 200
                lowestError = 99999
                bestPaddleVel = 0
                prevPaddleVel = 0
                prevPrevPaddleVel = 0
                bestNormal = np.array([0, 1, 0])
                moreOutgoingNormal = outgoingVelUnit
                moreCurrentNormal = incomingVelUnit
                prevNormal = bestNormal.copy()
                
                # if True:
                while error > threshold and attempts < maxAttempts:
                    # for starting guess, assume paddle is stationary and use that correct normal
                    if attempts == 0 or True:
                        upAxis = outgoingVelUnit + incomingVelUnit # bisect target and current velocity
                        upAxis /= np.linalg.norm(upAxis)
                    # for future guesses, binary search over angle range between current and target velocity
                    # if the paddle velocity was greater than previous attempt, move normal towards target half way
                    # if paddle velocity was less than previous attempt, move normal towards last normal half way
                    # else:
                    #     if prevPaddleVel > prevPrevPaddleVel:
                    #         print("GOING OUTGOING")
                    #         upAxis = moreOutgoingNormal + prevNormal
                    #         upAxis /= np.linalg.norm(upAxis)
                    #         moreCurrentNormal = prevNormal.copy()
                    #     else:
                    #         print("GOING CURRENT")
                    #         upAxis = moreCurrentNormal + prevNormal
                    #         upAxis /= np.linalg.norm(upAxis)
                    #         moreOutgoingNormal = prevNormal.copy()
                    #     prevNormal = upAxis.copy()
                    
                    ## randomly choose normal along line of rotation between incoming and outgoing vels
                    ## slightly less dirty than below
                    upAxis = np.random.rand(1)[0]*incomingVelUnit + 0.5*outgoingVelUnit
                    upAxis /= np.linalg.norm(upAxis)
                    
                    # # randomly disturb the normal, then renormalize, use this as our normal guess
                    # ### this is super dirty and we should try to find a better way
                    # upAxis += np.random.rand(3)-0.5
                    # upAxis /= np.linalg.norm(upAxis)

                    initialMagnitude = np.linalg.norm(np.dot(outgoingVel, upAxis))
                    incomingMagnitude = np.linalg.norm(np.dot(incomingVel, upAxis))
                    # print(f"INITMAG {initialMagnitude}")
                    # comes from collision equation see below
                    paddleMagnitude = (initialMagnitude - self.ballHit.restitution*incomingMagnitude)/(1+self.ballHit.restitution)
                    
                    ###### then based on that, see what the output would be, (this code is copied from ball update, check there for comments/more descriptions)
                    rotMat = alignVectorsRotationMatrix(incomingVel*np.array([-1,-1,-1]), upAxis)
                    reflectedVel = np.linalg.norm(incomingVel)*np.dot(rotMat, upAxis)
                    projectedVelocityMag = np.dot(reflectedVel, upAxis)
                    newProjectedVelocityMag, _ = collision(self.ballHit.mass, -projectedVelocityMag, -1, paddleMagnitude, self.ballHit.restitution)
                    
                    collisionAdjustMag = projectedVelocityMag - newProjectedVelocityMag
                    
                    collisionAdjust = collisionAdjustMag * upAxis
                    if np.dot(reflectedVel, upAxis) > 0:
                        finalVel = reflectedVel - collisionAdjust
                    else:
                        finalVel = reflectedVel + collisionAdjust
                    # print(f"PREDICTED VEL {finalVel}")
                    error = np.linalg.norm(outgoingVel - finalVel)
                    # print(f"ERROR {error}")
                    
                    # Save the best result to use later
                    if error < lowestError:
                        # print("NEW BEST SOLUTION FOUND ON ATTEMPT", attempts)
                        prevPrevPaddleVel = prevPaddleVel
                        prevPaddleVel = paddleMagnitude
                        bestPaddleVel = paddleMagnitude
                        bestNormal = upAxis.copy()
                        lowestError = error
                        
                    attempts += 1
                    
                
                if attempts == maxAttempts:
                    print("MAX ATTEMPTS OF", maxAttempts, "REACHED")
                    print("BEST SOLUTION HAS ERROR", lowestError)
                else:
                    print("NUMERICAL SOLUTION TOOK", attempts, "ATTEMPTS")
                    print("FINAL SOLUTION HAS ERROR", lowestError)
                
                upAxis = bestNormal
                paddleMagnitude = bestPaddleVel
                self.metaVel = upAxis * paddleMagnitude
                self.move(pos=pos-self.metaVel*t, upAxis=upAxis, angle=0, alignCenter=True)
                
                ####### make ball return to where it came from
#                 t, pos, vel = self.ballHit.findInterception(self.ballHit.pos[1])
#                 upAxis = -1*vel
#                 defaultUpAxis = np.array([0, 1, 0])
#                 rotMat = alignVectorsRotationMatrix(defaultUpAxis, upAxis)
#                 defaultAxis = np.array([1,0,0])
#                 defaultAxis = np.dot(rotMat, defaultAxis)
                
#                 distance = (pos - self.ballHit.pos)*np.array([1, 0, 1])
                
#                 theta = acos(np.dot(distance, upAxis)/(np.linalg.norm(upAxis)*np.linalg.norm(distance)))
#                 initialMagnitude = sqrt(abs(0.5*G*(np.linalg.norm(distance)/cos(theta))**2/(np.linalg.norm(distance)*tan(theta)+self.ballHit.pos[1]-pos[1])))
                
#                 # we want ball to leave collision with velocity magnitude of initialMagnitude
#                 # it is coming in with self.ballHit.vel velocity
#                 # we need to find metaVel which would be v2 to make the collision output initialMagnitude
#                 # vp is an intermediate, v1 is ball vel, v2 is paddle vel. v1p is output vel
#                 # vp = (1+alpha)*((m1*v1+m2*v2)/(m1+m2) if m2 > 0 else v2)
#                 # above simplifies to vp = (1+alpha)*v2 because of infinite mass paddle
#                 # v1p = vp - alpha*v1
#                 # find v2 such that v1p = initialMagnitude
#                 # initialMagnitude = vp - alpha*v1 = (1+alpha)*v2 - alpha*v1
#                 # v2 = (initialMagnitude + alpha*v1)/(1+alpha)
                
#                 paddleMagnitude = (initialMagnitude - self.ballHit.restitution*np.linalg.norm(vel))/(1+self.ballHit.restitution)
                
#                 self.metaVel = upAxis/np.linalg.norm(upAxis) * paddleMagnitude
                
                
#                 self.move(pos=pos-self.metaVel*t, axis=defaultAxis, angle=0, alignCenter=True)
                
                
                
                ###### Find the paddle velocity needed to keep the ball at the same height
                # t, pos, vel = self.ballHit.findInterception(self.ballHit.pos[1])
                # ball is coming in with velocity v1, if we have velocity v2 then:
                # vp = (1+alpha)*((m1*v1+m2*v2)/(m1+m2) if m2 > 0 else v2)
                # v1p = vp - alpha*v1
                # where vp is an intermediate, and v1p is the output velocity
                # we want to find v2 such that v1p is equal to -v1
                # but we know that m2 will be considered infinite, so the vp expression simplifies to:
                # vp = (1+alpha)*v2
                # which gives: v1p = (1+alpha)*v2 - alpha*v1
                # solving for v2 to find v1p=-v1 gives: v2=(alpha*v1-v1)/(1+alpha)=v1*(alpha-1)/(1+alpha)
                # self.metaVel[1] = self.ballHit.vel[1]*(1-self.ballHit.restitution)/(1+self.ballHit.restitution)
                # this code puts the ball straight back up at same velocity
                # self.move(pos=pos-self.metaVel*t, axis = [1,0,0], alignCenter=True)
                # # self.move(pos=pos, axis = [1,0,0], alignCenter=True)
                
            self.hitBall = False
            
        if np.any(self.metaVel != 0):
            self.move(pos=self.pos+self.metaVel*dt, alignCenter=False)
        
    def move(self, pos=None, axis=None, upAxis=None, angle=None, alignCenter=False):
        prevPos = self.pos
        prevAxis = self.axis
        prevAngle = self.angle
        prevUpVector = self.handle.normals[2]
        useUpAxis=False
        if np.all(axis != None):
            self.axis = np.array(axis, dtype=float)
        elif np.all(upAxis != None):
            useUpAxis=True
        if angle != None:
            self.angle = angle
        
        if useUpAxis:
            if np.any(prevUpVector != upAxis):
                self.axisRotation = alignVectorsRotationMatrix(np.array([0,1,0]), upAxis)
                self.angleRotation = axisAngleRotationMatrix(upAxis, self.angle)
        elif np.any(self.axis != prevAxis) or np.any(self.angle != prevAngle) or np.all(self.axisRotation == None):
            self.axisRotation = alignVectorsRotationMatrix(np.array([1,0,0]), self.axis)
            self.angleRotation = axisAngleRotationMatrix(self.axis, self.angle)
            
        if np.all(pos != None):    
            self.pos = np.array(pos, dtype=float)
        if alignCenter:
            centerAdjust = np.array([self.handle.size[0]+self.paddle.size[0]/2, 0, 0]) # bottom back left corner aligned coordinates
            # account for rotation
            centerAdjust = np.dot(self.axisRotation, centerAdjust)
            centerAdjust = np.dot(self.angleRotation, centerAdjust)
            self.pos = self.pos-centerAdjust
        self.vel = (self.pos - prevPos)/dt
        if useUpAxis:
            self.handle.move(pos=self.pos, upAxis=upAxis, angle=self.angle, axisRotation=self.axisRotation, angleRotation=self.angleRotation)
            self.axis = self.handle.normals[0]
        else:
            self.handle.move(pos=self.pos, axis=self.axis, angle=self.angle, axisRotation=self.axisRotation, angleRotation=self.angleRotation)
        
        paddleAdjust = np.array([self.handle.size[0], 0, self.handle.size[2]/2-self.paddle.size[2]/2]) # bottom back left corner aligned coordinates
        # paddleAdjust = np.array([self.handle.size[0], -self.paddle.size[1]/2, -self.paddle.size[2]/2]) # principle axis aligned coordinates
        # paddleAdjust = np.array([self.handle.size[0]/2+self.paddle.size[0]/2, 0, 0]) # center points aligned coordinates
        
        # account for rotation when placing paddle relative to handle
        paddleAdjust = np.dot(self.axisRotation, paddleAdjust)
        paddleAdjust = np.dot(self.angleRotation, paddleAdjust)
        if useUpAxis:
            self.paddle.move(pos=self.pos+paddleAdjust, upAxis=upAxis, angle=self.angle, axisRotation=self.axisRotation, angleRotation=self.angleRotation)
        else:
            self.paddle.move(pos=self.pos+paddleAdjust, axis=self.axis, angle=self.angle, axisRotation=self.axisRotation, angleRotation=self.angleRotation)
        
        if np.any(self.axis != prevAxis) or np.any(self.angle != prevAngle):
            # Find angular velocity based on change in axis and up vector
            dAxis = (self.axis - prevAxis) / dt
            dUp = (self.handle.normals[2] - prevUpVector) / dt
            # angularVel = (x X xdot)/(x . x) + x*[(y X ydot)/(y . y) - (x X xdot)/(x . x)]_n / (x-y)_n
            unconstrainedVector = np.cross(prevAxis, dAxis)/np.dot(prevAxis, prevAxis)
            constraintVector = np.cross(prevUpVector, dUp)/np.dot(prevUpVector, prevUpVector) - unconstrainedVector
            constraintScalar = constraintVector[0] / (prevAxis - prevUpVector)[0]
            self.angularVel = unconstrainedVector + prevAxis*constraintScalar
        
        if self.disableVelocity:
            self.vel = np.zeros(3)
            self.angularVel = np.zeros(3)
        
        
        
    def collideSphere(self, other):
        handleNormal, handlePoint = self.handle.collideSphere(other)
        paddleNormal, paddlePoint = self.paddle.collideSphere(other)
        if np.all(handleNormal != None) or np.all(paddleNormal != None):
            self.moveTarget()
            self.hitBall = True
            self.ballHit = other
        return [handleNormal, paddleNormal], [handlePoint, paddlePoint]
    
    def moveTarget(self):
        if self.ballHit != None:
            self.targetVisual.pos = vec(*self.targetPosition)
        self.targetPosition = (np.random.rand(3)*2-1)*0.5
        if self.targetPosition[1] < 0 and floorActive:
            self.targetPosition[1] *= -1
        # self.targetPosition[1] = 10 ## set to test hitting straight up or close to it
        self.targetPosition[1] = 1.5
        if self.ballHit == None:
            self.targetVisual.pos = vec(*self.targetPosition)
        
        
    
    def show(self):
        self.handle.show()
        self.paddle.show()




scene = canvas()
scene.autoscale = True
arrow(pos=vec(0,0,0), axis=vec(1,0,0), color=vec(1,0,0))
arrow(pos=vec(0,0,0), axis=vec(0,1,0), color=vec(0,1,0))
arrow(pos=vec(0,0,0), axis=vec(0,0,1), color=vec(0,0,1))

# give ball height 5 for test scenarios
ball = Ball(pos = [0.2, 3, 0.4], make_trail=False)
moveables = [ball]
collideables = []

paddle = PingPongPaddle(pos=[0.2,0.4,0.4])
# randVec = (np.random.rand(3)*2)-1
# paddle.move(axis=randVec/np.linalg.norm(randVec)*np.array([1, 0.2, 1]), alignCenter=True)
# paddle.move(axis=[1, 0.2, 0], alignCenter=True)
paddle.move(alignCenter=True)
# paddle.vel = np.array([0, 0.1, 0])
# paddle.angularVel = np.array([0, 0.3, 0.3])
collideables.append(paddle)
moveables.append(paddle)


moveablesStates = []
for _ in moveables:
    moveablesStates.append([])
t = 0
frames = 0
simLengthSeconds = 12/speedMultiplier
# simLengthTicks = simLengthSeconds/speedMultiplier
startTime = time.time()

while t < simLengthSeconds:
    moveCam = False
    for _ in range(resolutionMultiplier):
        # startTime = time.time()
        ball.update(collideables=collideables)
        
        if paddle.hitBall:
            moveCam = True
            
        paddle.update()
        t += dt
        frames += 1
        # endTime = time.time()
    if preSimulateThenPlayback:
        for i in range(len(moveables)):
            moveablesStates[i].append(np.copy(moveables[i].pos))
    if not preSimulateThenPlayback:
        for moveable in moveables:
            moveable.show()
        if moveCam and False:
            scene.camera.pos = paddle.paddle.visual.pos + vec(0, 5, 5)
            scene.camera.axis = paddle.paddle.visual.pos - scene.camera.pos
        rate(fps)
endTime = time.time()

if preSimulateThenPlayback:
    print(f"Simulation took: {endTime - startTime} seconds")
    for t in range(len(moveablesStates[0])):
        for i in range(len(moveablesStates)):
            moveables[i].move(pos=moveablesStates[i][t])
            moveables[i].show()
        rate(fps)
else:
    print(f"Simulation was expected to take: {t} seconds and it took: {endTime - startTime} seconds")