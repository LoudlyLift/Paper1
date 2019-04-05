import math

"""

power -- power required to transmit data (watts)

power_waiting -- power required to await an incoming transmission (watts)

gain -- ???

frequency -- this UE's clock speed (hertz)

energyPerCycle -- energy consumed by each clock cycle (joules?)

timeenergy_ratio -- the linear tradeoff to make between optimizing energy
consumption and optimizing the time until completion. One means complete the
task as soon as possible without regard to energy costs. Zero means that
equipment's task is to be completed in the most energy efficient manner possible
(as long as it is still completed within sDelayMax seconds).

cbInput -- the byte count of the input cCycle

cCycle -- the count of the number of CPU cycles it will take to compute this
task

sDelayMax -- maximum tolerable delay of the task, in seconds

"""
class equipment:
    def __init__(self, power: float, power_waiting: float, gain, frequency:
                 float, energyPerCycle: float, timeenergy_ratio: float, cbInput:
                 int, cCycle: int, sDelayMax: float):
        self.power = power
        self.power_waiting = power_waiting
        self.gain = gain
        self.frequency = frequency
        self.energyPerCycle = energyPerCycle
        self.timeenergy_ratio = timeenergy_ratio
        self.cbInput = cbInput
        self.cCycle = cCycle
        self.sDelayMax = sDelayMax

    def local_processing_time(self)->float:
        #eq. (2)
        return self.cCycle / self.frequency

    def cost_local(self)->float:
        time = self.local_processing_time()
        #eq. (3)
        energy = self.cCycle * self.energyPerCycle

        #eq. after (3) but before (4)
        return time*self.timeenergy_ratio + energy*(1-self.timeenergy_ratio)

    def cost_offload(self, bandwidth: float, effectiveServerClockSpeed: float, N0: float)->float:
        """Computes the cost of offloading this equipment's task

        bandwidth -- the amount of bandwidth to use when uploading the task to the MEC server

        effectiveServerClockSpeed -- the rate at which the server will process the task after it is uploaded (CPU cycles / second)

        N0 -- ???
        """

        # eq. (1)
        numerator=self.power*self.gain
        denominator=bandwidth*N0
        quotient=numerator/denominator
        upload_rate = bandwidth * math.log(1+quotient)


        # eq. (4)
        time_offload = self.cbInput / upload_rate

        # eq. (5)
        energy_offload = self.power * time_offload

        # eq. (6)
        time_processing = self.cCycle / effectiveServerClockSpeed

        # eq. (7)
        energy_waiting = self.power_waiting * time_processing

        # eq. (9)
        time_total = time_offload + time_processing

        # eq. (10)
        energy_total = energy_offload + energy_waiting

        # eq. (11)
        return time_total*self.timeenergy_ratio + energy_total*(1-self.timeenergy_ratio)
