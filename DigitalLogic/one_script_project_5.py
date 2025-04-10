from vpython import *
from queue import PriorityQueue

# class PriorityQueue:
#     def __init__(self):
#         self.items=[]
#     def put(self, item):
#         for i in range(len(self.items)):
#             if item < self.items[i]:
#                 self.items.insert(i, item)
#                 return
#         self.items.append(item)
#     def get(self):
#         return self.items.pop(0)
#     def empty(self):
#         return len(self.items)==0
class Gate:
    def __init__(self, func, inputs, output, delay=10):
        self.func = func
        self.inputs = inputs
        for net in self.inputs:
            net.receivers.append(self)
        self.output = output
        self.delay = delay
    def drive(self, t):
        
        outputValue = self.func(self.inputs)
        # if t+self.delay==21:
        #     print(outputValue)
        #     print(self.output.value)
        # outputValue = None
        # for net in self.inputs:
        #     if net.value=='X':
        #         # print('X')
        #         outputValue='X'
                
        # # print(self.func(self.inputs))
        # if outputValue == None:
        #     outputValue=self.func(self.inputs)
        if self.output.value != outputValue:
            return Event(self.output.id, t+self.delay, outputValue)

def nand(inputs):
    hasX = False
    for net in inputs:
        if net.value==0:
            return 1
        elif net.value=='X':
            hasX = True
    
    return 'X' if hasX else 0
        
    
class Event:
    def __init__(self, netID, time, value):
        self.netID = netID
        self.time = time
        self.value = value
    def __lt__(self, other):
        return self.time < other.time
    def __str__(self):
        return "Net ID: " + str(self.netID) + " | Time: " + str(self.time) + " | Value: " + str(self.value)
    
class Net:
    nets = []
    def __init__(self, receivers=None, initialValue='X'):
        self.id = len(Net.nets)
        if receivers == None:
            self.receivers = []
        else:
            self.receivers = receivers
        #self.receivers = receivers[0:-1] ## ALSO WORKS
        self.value=initialValue
        Net.nets.append(self)
    def update(self, value, t):
        self.value = value
        events = []
        for receiver in self.receivers:
            event = receiver.drive(t)
            if event:
                events.append(event)
        return events
    def graphValue(self):
        return -1+self.id*3 if self.value == 'X' else self.value+self.id*3

Net.nets = []
queue = PriorityQueue()
# AND from two NANDS
# gate1 = Gate(nand, [Net(), Net()], Net())
# gate2 = Gate(nand, [gate1.output, gate1.output], Net())
# initialEvents = [Event(0, 0, 0), Event(1, 1, 0), Event(0, 2, 1), Event(1, 3, 1)]

# D Latch
# D=Net(initialValue=0)
# E=Net(initialValue=0)
# notD=Net(initialValue=1)
# S=Net(initialValue=1)
# R=Net(initialValue=1)
D=Net()
E=Net()
notD=Net()
S=Net()
R=Net()
Q=Net(initialValue=0)
QBar=Net(initialValue=1)
SGate=Gate(nand, [D, E], S)
notGate=Gate(nand, [D,D], notD)
RGate=Gate(nand, [notD,E], R)
QGate=Gate(nand, [S, QBar], Q)
QBarGate=Gate(nand, [R, Q], QBar)
inputEvents = [Event(0, 0, 0), Event(1, 0, 0), Event(0, 2, 1), Event(0, 4, 0), Event(1, 6, 1), Event(0, 8, 1), Event(0, 10, 0), Event(0, 12, 1), Event(1, 14, 0), Event(0, 16, 0)]

# [queue.put(i) for i in initialEvents]
t=0
curves = [gcurve(color=vec(0, 1.0/float(i+1), 1.0/float(len(Net.nets)-i))) for i in range(len(Net.nets))]
curves[0].plot(-1,0)

runs=0
while not len(inputEvents)==0:
    newEvent = inputEvents.pop(0)
    newEvent.time = t
    queue.put(newEvent)
    while not queue.empty():
        for i in range(len(curves)):
            curves[i].plot(t, Net.nets[i].graphValue())
        
        event = queue.get()
        print(event)
        t = event.time

        for i in range(len(curves)):
            curves[i].plot(t, Net.nets[i].graphValue())

        [queue.put(ev) for ev in Net.nets[event.netID].update(event.value, t)]

        
        runs+=1
        rate(20)
for i in range(len(curves)):
    curves[i].plot(t, Net.nets[i].graphValue())
for i in range(len(curves)):
    curves[i].plot(t+5, Net.nets[i].graphValue())
print("Done!")
print(runs)