from biotek_driver.biotek import Biotek
from biotek_driver.xml_builders.partial_plate_builder import build_bti_partial_plate_xml
import time
import string
import xml.etree.ElementTree as ET

class Biotek_Wrapper:
    biotek = None
    def __init__(self, ComPort=4):
        self.biotek = Biotek(reader_name="Cytation5",communication="serial",com_port=ComPort)
        status = self.biotek.get_reader_status()
        print(f"Current reader status: {status}")
    def CarrierIn(self):
        self.biotek.carrier_in()
    def CarrierOut(self):
        self.biotek.carrier_out()
    def run_protocol(self,protocol_path, wells=None):
        self.biotek.app.data_export_enabled = True
        experiment = self.biotek.new_experiment(protocol_path)
        if experiment:
            print("Experiment created successfully.")
            print(f"Experiment Protocol Type: {experiment.protocol_type}")
        
            plates = experiment.plates
            
            for group in grouped_wells:
                plate = plates.add()
                plate.keep_plate_in_after_read()
                if not plate:
                    print("Failed to add a plate. Possibly a multi-plate assay.")
                else:
                    current_procedure = plate.get_procedure()
                    root = ET.fromstring( current_procedure )

                    # Extract wavelength values
                    start_nm = int(root.find(".//WavelengthStartnm").text)
                    stop_nm = int(root.find(".//WavelengthStopnm").text)
                    step_nm = int(root.find(".//WavelengthStepnm").text)

                    # Generate list of wavelengths
                    wavelengths = list(range(start_nm, stop_nm + 1, step_nm))

                    print(wavelengths)  # Output: [400, 420, 440]

                    # 2) Define a partial plate (random wells) and set it
                    random_well_xml = build_bti_partial_plate_xml(single_block=False,wells=group)
                    plate.set_partial_plate(random_well_xml)

                    # Start an actual read
                    monitor = plate.start_read()


                    if not monitor:
                        print("Failed to start read.")
                    else:
                        print("Read started. Waiting for completion...")

                        # Wait until read completes or errors out
                        # read_status can be:
                        #   0 => Not Started
                        #   1 => In Progress
                        #   2 => Aborted
                        #   3 => Paused
                        #   4 => Error
                        #   5 => Completed
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

                        result_len = 1
                        while result_len > 0:
                            results = plate.get_raw_data()
                            result_len = len(results[1]['value'])
                            if result_len > 0:
                                print(results[1]['value'])                  
            else:
                print("Experiment creation failed.")
        else:
            print("Failed to establish USB connection. Please check the device.")

    def well_index_to_label(self,index):
        row = string.ascii_uppercase[index // 12]  # Get row letter (A-H)
        col = (index % 12) + 1  # Get column number (1-12)
        return f"{row}{col}"

    def group_wells(self,indices):
        indices = sorted(indices)  # Ensure sorted order
        grouped = []
        current_group = []
        
        for i in range(len(indices)):
            if not current_group:
                current_group.append(indices[i])
            else:
                prev = current_group[-1]
                if indices[i] == prev + 1 and (indices[i] % 12 != 0):  # Check continuity and row overflow
                    current_group.append(indices[i])
                else:
                    grouped.append([self.well_index_to_label(idx) for idx in current_group])
                    current_group = [indices[i]]
        
        if current_group:
            grouped.append([self.well_index_to_label(idx) for idx in current_group])
        
        return grouped

#Example usage:
biotek_i = Biotek_Wrapper()
indices = [0, 1, 2, 9, 10, 11, 12]
grouped_wells = biotek_i.group_wells(indices)
print(grouped_wells)  # Output: [['A1', 'A2', 'A3'], ['A10', 'A11', 'A12'], ['B1']]
protocol_path = r"C:\Protocols\Spectral_Automation.prt"
biotek_i.CarrierOut()
biotek_i.CarrierIn()
biotek_i.run_protocol(protocol_path,grouped_wells)

