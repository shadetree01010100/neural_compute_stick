from unittest.mock import patch
from nio.block.terminals import DEFAULT_TERMINAL
from nio.signal.base import Signal
from nio.testing.block_test_case import NIOBlockTestCase
from ..ncs_inference_block import NCS_Inference


class TestNCS_Inference(NIOBlockTestCase):

    @patch(NCS_Inference.__module__ + '.mvnc.mvncapi')
    def test_process_signals(self, mock_ncs):
        """Signals pass through block unmodified."""
        blk = NCS_Inference()
        self.configure_block(blk, {})
        blk.start()
        blk.process_signals([Signal({"hello": "nio"})])
        blk.stop()
        self.assert_num_signals_notified(1)
        self.assertDictEqual(
            self.last_notified[DEFAULT_TERMINAL][0].to_dict(),
            {"hello": "nio"})
