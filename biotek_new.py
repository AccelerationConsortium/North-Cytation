import time
import string
import pandas as pd
import xml.etree.ElementTree as ET

class Biotek_Wrapper:
    biotek = None
    def __init__(self, ComPort=4,simulate=False):
        if not simulate:
            from biotek_driver.biotek import Biotek
            from biotek_driver.xml_builders.partial_plate_builder import build_bti_partial_plate_xml
            self.biotek = Biotek(reader_name="Cytation5",communication="serial",com_port=ComPort)
            self.build_bti_partial_plate_xml = build_bti_partial_plate_xml
        else:
            from unittest.mock import MagicMock
            self.biotek = MagicMock()
            self.build_bti_partial_plate_xml = None
        status = self.biotek.get_reader_status()
        print(f"Current reader status: {status}")
        if status == 0:
            print('Cytation is connected')
        elif not simulate:
            input("Cytation not connected... May need to restart")
            
    def CarrierIn(self,plate_type="96 WELL PLATE"):
        self.biotek.carrier_in(plate_type_name=plate_type)
    def CarrierOut(self):
        self.biotek.carrier_out()
    
    def monitor_plate_read(self, plate):
        # Start an actual read
        monitor = plate.start_read()
        if not monitor:
            print("Failed to start read.")
        else:
            print("Read started. Waiting for completion...")

            while True:
                rstatus = plate.read_status
                if rstatus == 5:
                    print("Plate read completed.")
                    break
                elif rstatus == 2:
                    print("Plate read was aborted.")
                    break
                elif rstatus == 3:
                    print("Plate read is paused (waiting).")
                    # You could decide to continue waiting or handle differently
                elif rstatus == 4:
                    print("Plate read error encountered.")
                    break

                time.sleep(2)

    def get_num_reads(self,plate):
        current_procedure = plate.get_procedure()
        root = ET.fromstring( current_procedure )
            # Find all Measurement elements
        measurement_elements = root.findall(".//Measurement")

        # Collect unique index values
        unique_indexes = {m.attrib['Index'] for m in measurement_elements}

        return unique_indexes

    def get_wavelengths_from_plate(self,plate):
        current_procedure = plate.get_procedure()

        #print(current_procedure)

        #input()

        root = ET.fromstring( current_procedure )

        # Extract wavelength values
        start_nm = int(root.find(".//WavelengthStartnm").text)
        stop_nm = int(root.find(".//WavelengthStopnm").text)
        step_nm = int(root.find(".//WavelengthStepnm").text)

        # Generate list of wavelengths
        wavelengths = list(range(start_nm, stop_nm + 1, step_nm))

        return wavelengths

    def run_plate(self,plate,wells,prot_type):
        
        plate_data = pd.DataFrame()
        plate.keep_plate_in_after_read()
        if not plate:
            print("Failed to add a plate. Possibly a multi-plate assay.")
        else:
            if prot_type=="spectra":
                wavelengths = self.get_wavelengths_from_plate(plate)
                plate_data['Wavelengths']=wavelengths

            # 2) Define a partial plate (random wells) and set it
            # random_well_xml = build_bti_partial_plate_xml(single_block=False,wells=wells)
            random_well_xml = self.build_bti_partial_plate_xml(single_block=False,wells=wells)
            plate.set_partial_plate(random_well_xml)

            #3) Monitor the plate while it runs
            self.monitor_plate_read(plate)

            #Add the data to the dataframe
            if prot_type == "spectra":
                for well in wells:
                    results = plate.get_raw_data()
                    plate_data[well]=(results[1]['value']) 
            elif prot_type == "read":
                num_reads = self.get_num_reads(plate)
                print("Measurement types: ", num_reads)
                for measurement_type in num_reads:
                    results = plate.get_raw_data()
                    i = 0
                    for well in wells:
                        plate_data.at[well,measurement_type]=(list(results[1]['value'])[i])
                        i=i+1
            return plate_data
        return None    
    
    def determine_read_type(self, plate):
        current_procedure = plate.get_procedure()
        root = ET.fromstring(current_procedure)

        # Find the <ReadStep> element
        read_step = root.find(".//ReadStep")
        if read_step is None:
            return "Unknown"

        # Get detection and read type
        detection = read_step.findtext("Detection")
        read_type = read_step.findtext("ReadType")

        # Determine based on known logic
        if read_type == "Spectrum":
            return "spectra"
        elif read_type == "EndPoint":
            return "read"
        else:
            return "Unknown"

    def run_protocol(self,protocol_path, wells=None, plate_type="96 WELL PLATE"):
        self.biotek.app.data_export_enabled = True
        experiment = self.biotek.new_experiment(protocol_path)
        protocol_data = pd.DataFrame()

        if experiment:
            print("Experiment created successfully.")
            print(f"Experiment Protocol Type: {experiment.protocol_type}")
            plates = experiment.plates
            if wells is not None:
                grouped_wells = self.group_wells(wells, plate_type)
            for well_group in grouped_wells:
                plate = plates.add()
                prot_type=self.determine_read_type(plate)
                data_group = self.run_plate(plate,well_group,prot_type=prot_type)
                if protocol_data.empty:
                    protocol_data = data_group
                elif prot_type=="spectra":
                    protocol_data = protocol_data.merge(data_group, on ="Wavelengths", how='outer')     
                elif prot_type=="read":
                    protocol_data = pd.concat([protocol_data,data_group])      
        else:
            print("Experiment creation failed.")

        return protocol_data

    def well_index_to_label(self,index,nums_per_letter):
        row = string.ascii_uppercase[index // nums_per_letter]  # Get row letter (A-H)
        col = (index % nums_per_letter) + 1  # Get column number (1-12)
        return f"{row}{col}"

    def group_wells(self,indices,plate_type):
        indices = sorted(indices)  # Ensure sorted order
        grouped = []
        current_group = []

        print(plate_type)
        if plate_type == "96 WELL PLATE":
            nums_per_letter = 12
        elif plate_type == "48 WELL PLATE":
            nums_per_letter = 8
        else:
            print("Wellplate conversion issue")
        
        for i in range(len(indices)):
            if not current_group:
                current_group.append(indices[i])
            else:
                prev = current_group[-1]
                if indices[i] == prev + 1 and (indices[i] % nums_per_letter != 0):  # Check continuity and row overflow
                    current_group.append(indices[i])
                else:
                    grouped.append([self.well_index_to_label(idx,nums_per_letter) for idx in current_group])
                    current_group = [indices[i]]
        
        if current_group:
            grouped.append([self.well_index_to_label(idx,nums_per_letter) for idx in current_group])
        
        return grouped
