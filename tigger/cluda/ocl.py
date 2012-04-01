class OclEnvironment:

    def __init__(self, device_num=0, fast_math=True):
        self.fast_math = fast_math

    def supportsDouble(self):
        return False
