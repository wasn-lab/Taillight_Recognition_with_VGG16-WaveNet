import time
import threading

class DELAY_CLOSE(object):
    """
    This is a simple on/off (True/False) state managment.
    For False-->True, it's imediate change.
    For True-->False, it will be delayed for self.delay_sec,
         if there is no False-->True transient happend in the original signal.
    """
    def __init__(self, delay_sec=1.0, init_state=False):
        """
        """
        # Variables
        self.state = init_state
        # Parameters
        self.delay_sec = delay_sec
        #
        self.close_thread = None

    def __str__(self):
        return str(self.state)

    def input(self, signal_ori):
        """
        Input the original signal
        """
        if signal_ori:
            self._cancel_timer_close()
            self.state = True
        else:
            self._set_timer_close()

    def output(self):
        """
        """
        return self.state

    def _cancel_timer_close(self):
        """
        """
        if not self.close_thread is None:
            self.close_thread.cancel()

    def _set_timer_close(self):
        """
        """
        self._cancel_timer_close()
        self.close_thread = threading.Timer(self.delay_sec, self._timeout_handle)
        self.close_thread.start()

    def _timeout_handle(self):
        """
        """
        self.state = False
