import mvnc.mvncapi as ncs
import numpy as np

from nio import Block
from nio.properties import VersionProperty, StringProperty


class NCS_Inference(Block):

    version = VersionProperty('0.1.0')
    model = StringProperty(title='Model Path')
    name = StringProperty(title='Graph Name')

    def __init__(self):
        super().__init__()
        self.device = None
        self.graph = None
        self.input_fifo = None
        self.output_fifo = None

    def configure(self, context):
        super().configure(context)
        self.device = ncs.Device(ncs.EnumerateDevices()[0])
        self.device.open()
        with open(model, mode='rb') as f:
            graph_file_buffer = f.read()
        self.graph = mvncapi.Graph(name)
        self.input_fifo, self.output_fifo = graph.allocate_with_fifos(
            device, 
            graph_file_buffer, 
            input_fifo_type=mvncapi.FifoType.HOST_WO, 
            input_fifo_data_type=mvncapi.FifoDataType.FP32,
            input_fifo_num_elem=2, 
            output_fifo_type=mvncapi.FifoType.HOST_RO, 
            output_fifo_data_type=mvncapi.FifoDataType.FP32, 
            output_fifo_num_elem=2
        )

    def process_signals(self, signals):
        outgoing_signals = []
        for signal in signals:
            if type(signal) is np.ndarray:
                graph.queue_inference_with_fifo_elem(
                    self.input_fifo, self.output_fifo, signal, 'user object')
                output, user_obj = output_fifo.read_elem()
                outgoing_signals.append(Signal({'prediction': output}))
        self.notify_signals(outgoing_signals)

    def stop(self):
        self.graph.destroy()
        self.device.close()
        self.device.destroy()
        input_fifo.destroy()
