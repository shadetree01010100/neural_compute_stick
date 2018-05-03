from nio import Block
from nio.properties import VersionProperty, StringProperty
try:
    import mvnc.mvncapi as ncs
except:
    pass


class NCS_Inference(Block):

    version = VersionProperty('0.1.0')
    model = StringProperty(title='Model Path')

    def __init__(self):
        super().__init__()
        self.device = None
        self.graph = None

    def configure(self, context):
        super().configure(context)
        self.device = ncs.Device(ncs.EnumerateDevices()[0])
        self.device.OpenDevice()
        self.graph = self.device.AllocateGraph(self.model())

    def process_signals(self, signals):
        outgoing_signals = []
        for signal in signals:
            if self.graph.LoadTensor(signal.batch, 'userObject'):
                output, _ = graph.GetResult()
                outgoing_signals.append(Signal({'prediction': output}))
        self.notify_signals(outgoing_signals)

    def stop(self):
        self.graph.DeallocateGraph()
        self.device.CloseDevice()
