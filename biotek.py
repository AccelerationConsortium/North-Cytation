import pythoncom

class Biotek:
    def __init__(self, readerType=21, ComPort=4, appName = 'Gen5.Application', BaudRate = 38400):
        self.appName = appName
        self.readerType = readerType
        self.ComPort = ComPort
        self.BaudRate = BaudRate
        pythoncom.CoInitialize()
        self.appDispatch = pythoncom.new(appName)
        self.ConfigureSerialReader()
        if self.TestReaderCommunication() == 1:
            print(self.appName + ' is connected')
        else:
            input("Carrier not connected")
            

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
        pythoncom.CoUninitialize()
        self.appDispatch = None

    def load_protocol(self, protocol_file_path):
        experiment_id = self.appDispatch.GetIDsOfNames('NewExperiment')
        print("Loading experiment", experiment_id)
        LCID = 0x0
        wFlags = pythoncom.DISPATCH_METHOD
        bResultWanted = True
        NewExperiment = self.appDispatch.Invoke(experiment_id, LCID, wFlags, bResultWanted, protocol_file_path)

        ####Get access to the plates interface?
        plates_id = NewExperiment.GetIDsOfNames('Plates')
        print("Plates ID", plates_id)
        wFlags = pythoncom.DISPATCH_PROPERTYGET
        bResultWanted = True
        Plates = NewExperiment.Invoke(plates_id, LCID, wFlags, bResultWanted)

        ###Get the plate?
        plate_id = Plates.GetIDsOfNames('GetPlate')
        print("Plate id", plate_id)
        wFlags = pythoncom.DISPATCH_METHOD
        bResultWanted = True
        plate = Plates.Invoke(plate_id, LCID, wFlags, bResultWanted, 1)
        print(plate)
        return plate
    
    def run_protocol(self,plate):
        print("Running experiment")
        start_id = plate.GetIDsOfNames('StartRead')
        print("Start ID", start_id)
        LCID = 0x0
        wFlags = pythoncom.DISPATCH_METHOD
        bResultWanted = True
        monitor = plate.Invoke(start_id, LCID, wFlags, bResultWanted)
        return monitor

    def protocol_in_progress(self,monitor):
        read_id = monitor.GetIDsOfNames('ReadInProgress')
        LCID = 0x0
        wFlags = pythoncom.DISPATCH_PROPERTYGET
        bResultWanted = True
        in_progress = monitor.Invoke(read_id, LCID, wFlags, bResultWanted)
        return in_progress