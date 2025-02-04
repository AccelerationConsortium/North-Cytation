import pythoncom

class Biotek:
    def __init__(self, readerType, ComPort, appName = 'Gen5.Application', BaudRate = 38400):
        self.appName = appName
        self.readerType = readerType
        self.ComPort = ComPort
        self.BaudRate = BaudRate
        self.appDispatch = pythoncom.new(appName)
        self.ConfigureSerialReader()
        if self.TestReaderCommunication() == 1:
            print(self.appName + ' is connected')
        else:
            print("Carrier not connected")

    def TestReaderCommunication(self):
        TestReaderCommunication_id = self.appDispatch.GetIDsOfNames('TestReaderCommunication')
        LCID = 0x0
        wFlags = pythoncom.DISPATCH_METHOD
        bResultWanted = True
        return self.appDispatch.Invoke(TestReaderCommunication_id, LCID, wFlags, bResultWanted)

    def _getIDsOfNames(self, func_name):
        return self.appDispatch.GetIDsOfNames(func_name)

    def ConfigureSerialReader(self):
        func_name = 'ConfigureSerialReader'
        func_id = self._getIDsOfNames(func_name)
        LCID = 0x0
        #define the flags
        wFlags = pythoncom.DISPATCH_METHOD
        #do we want results back
        bResultWanted = True
        return self.appDispatch.Invoke(func_id, LCID, wFlags, bResultWanted, self.readerType, self.ComPort, self.BaudRate)
    
    def _simple_method_invoke(self, func_name):
        func_id = self._getIDsOfNames(func_name)
        LCID = 0x0
        #define the flags
        wFlags = pythoncom.DISPATCH_METHOD
        #do we want results back
        bResultWanted = False
        #define the parameters of our Range Property
        return self.appDispatch.Invoke(func_id, LCID, wFlags, bResultWanted)

    def CarrierOut(self):
        print("Sending out carrier...")
        func_name = 'CarrierOut'
        self._simple_method_invoke(func_name)

    def CarrierIn(self):
        print("Sending in carrier...")
        func_name = 'CarrierIn'
        self._simple_method_invoke(func_name)

    def close(self):
        self.appDispatch = None
