"""
Biotek Cytation5 Plate Reader Interface

This module provides a Python wrapper for controlling the Biotek Cytation5 plate reader.
It handles plate reading operations, protocol execution, and data extraction for both
spectral analysis and endpoint measurements.

Key Features:
- Automated plate carrier control (in/out)
- Protocol execution with well selection
- Support for spectral scans and endpoint reads
- Data extraction and formatting to pandas DataFrames
- Simulation mode for testing without hardware

Dependencies:
- biotek_driver: Biotek's proprietary driver (required for hardware mode)
- pandas: Data manipulation and analysis
- xml.etree.ElementTree: XML parsing for protocol configuration

Author: North Robotics
Date: October 2025
"""

import time
import string
import pandas as pd
import xml.etree.ElementTree as ET

class Biotek_Wrapper:
    """
    Python wrapper for Biotek Cytation5 plate reader control.
    
    This class provides high-level methods for operating the Biotek Cytation5,
    including plate handling, protocol execution, and data retrieval.
    
    Attributes:
        biotek: Biotek driver instance (or mock in simulation mode)
        logger: Logger instance for debugging and status messages
        build_bti_partial_plate_xml: Function to build XML for partial plate reads
    """
    
    biotek = None
    
    def _log(self, level, message):
        """
        Safe logging helper that only logs if a logger is available.
        
        Args:
            level (str): Log level ('debug', 'info', 'warning', 'error')
            message (str): Message to log
        """
        if self.logger:
            getattr(self.logger, level)(message)
    
    def __init__(self, ComPort=4, simulate=False, logger=None):
        """
        Initialize the Biotek wrapper.
        
        Args:
            ComPort (int): Serial communication port number (default: 4)
            simulate (bool): If True, use mock objects instead of real hardware
            logger: Logger instance for debugging and status tracking (optional)
                   If None, logging will be disabled
            
        Raises:
            RuntimeError: If hardware connection fails in non-simulation mode
        """
        self.logger = logger

        if not simulate:
            from biotek_driver.biotek import Biotek
            from biotek_driver.xml_builders.partial_plate_builder import build_bti_partial_plate_xml
            self.biotek = Biotek(reader_name="Cytation5", communication="serial", com_port=ComPort)
            self.build_bti_partial_plate_xml = build_bti_partial_plate_xml
        else:
            from unittest.mock import MagicMock
            self.biotek = MagicMock()
            self.build_bti_partial_plate_xml = None
            
        status = self.biotek.get_reader_status()
        self._log('debug', f"Current reader status: {status}")
        if status == 0:
            self._log('info', 'Cytation5 is connected and ready')
        elif not simulate:
            self._log('warning', "Cytation5 not connected... May need to restart")
            
    def CarrierIn(self, plate_type="96 WELL PLATE"):
        """
        Move the plate carrier into the reader.
        
        Args:
            plate_type (str): Type of plate to expect ("96 WELL PLATE" or "48 WELL PLATE")
        """
        self.biotek.carrier_in(plate_type_name=plate_type)
        
    def CarrierOut(self):
        """Move the plate carrier out of the reader for plate access."""
        self.biotek.carrier_out()
    
    def monitor_plate_read(self, plate):
        """
        Monitor a plate read operation until completion.
        
        Continuously checks the read status and waits for completion.
        Handles various read states including completion, abortion, and errors.
        
        Args:
            plate: Biotek plate object with an active read operation
            
        Raises:
            RuntimeError: If read fails to start, is aborted, or encounters an error
        """
        # Start an actual read
        monitor = plate.start_read()
        if not monitor:
            self._log('error', "Failed to start read.")
            raise RuntimeError("Failed to start plate read")
        else:
            self._log('info', "Read started. Waiting for completion...")

            while True:
                rstatus = plate.read_status
                if rstatus == 5:
                    self._log('info', "Plate read completed successfully.")
                    break
                elif rstatus == 2:
                    self._log('warning', "Plate read was aborted.")
                    raise RuntimeError("Plate read was aborted")
                elif rstatus == 3:
                    self._log('warning', "Plate read is paused (waiting).")
                    # Continue waiting for read to resume
                elif rstatus == 4:
                    self._log('error', "Plate read error encountered.")
                    raise RuntimeError("Plate read error encountered")

                time.sleep(2)  # Check status every 2 seconds

    def extract_measurement_parameters(self, plate):
        """
        Extract measurement parameters from the plate's XML procedure.
        
        Parses the protocol XML to determine what measurements are being taken,
        including wavelengths for absorbance or excitation/emission pairs for fluorescence.
        
        Args:
            plate: Biotek plate object with loaded procedure
            
        Returns:
            list: List of measurement parameter strings (e.g., ['280', '485_528'])
        """
        self._log('debug', "Extracting measurement parameters from plate procedure.") 
        current_procedure = plate.get_procedure()
        root = ET.fromstring(current_procedure)

        results = []

        for measurement in root.findall(".//Measurement"):
            wavelength = measurement.find("Wavelength")
            excitation = measurement.find("Excitation")
            emission = measurement.find("Emission")

            if wavelength is not None:
                # Absorbance measurement - single wavelength
                results.append(wavelength.text.strip())
            elif excitation is not None and emission is not None:
                # Fluorescence measurement - excitation/emission pair
                ex_val = excitation.text.split('/')[0].strip()
                em_val = emission.text.split('/')[0].strip()
                results.append(f"{ex_val}_{em_val}")

        return results

    def get_wavelengths_from_plate(self, plate):
        """
        Extract wavelength range from spectral scan procedure.
        
        Parses the protocol XML to get the wavelength scan parameters
        and generates the complete list of wavelengths to be measured.
        
        Args:
            plate: Biotek plate object with spectral scan procedure
            
        Returns:
            list: List of wavelength values (e.g., [280, 285, 290, ...])
        """
        current_procedure = plate.get_procedure()
        root = ET.fromstring(current_procedure)

        # Extract wavelength scan parameters
        start_nm = int(root.find(".//WavelengthStartnm").text)
        stop_nm = int(root.find(".//WavelengthStopnm").text)
        step_nm = int(root.find(".//WavelengthStepnm").text)

        # Generate complete wavelength list
        wavelengths = list(range(start_nm, stop_nm + 1, step_nm))

        return wavelengths

    def run_plate(self, plate, wells, prot_type):
        """
        Execute a plate read operation on specified wells.
        
        Configures the plate for partial read, executes the protocol,
        and extracts the measurement data into a pandas DataFrame.
        
        Args:
            plate: Biotek plate object with loaded procedure
            wells (list): List of well labels to read (e.g., ['A1', 'A2', 'B1'])
            prot_type (str): Protocol type - either 'spectra' or 'read'
            
        Returns:
            pandas.DataFrame: Measurement data with wells as columns/rows
                - For 'spectra': Wavelengths as index, wells as columns
                - For 'read': Wells as index, measurement types as columns
                
        Returns:
            None: If plate operation fails
        """
        self._log('info', f"Running plate with wells: {wells} and protocol type: {prot_type}")
        plate_data = pd.DataFrame()
        plate.keep_plate_in_after_read()
        
        if not plate:
            self._log('error', "Failed to add a plate. Possibly a multi-plate assay.")
            return None
            
        if prot_type == "spectra":
            # For spectral scans, get the wavelength range
            wavelengths = self.get_wavelengths_from_plate(plate)
            plate_data['Wavelengths'] = wavelengths

        # Configure partial plate for selected wells only
        random_well_xml = self.build_bti_partial_plate_xml(single_block=False, wells=wells)
        plate.set_partial_plate(random_well_xml)

        # Execute the read and monitor until completion
        self.monitor_plate_read(plate)

        # Extract and organize the measurement data
        if prot_type == "spectra":
            # Spectral data: each well gets a column of intensity values
            for well in wells:
                results = plate.get_raw_data()
                plate_data[well] = (results[1]['value']) 
        elif prot_type == "read":
            # Endpoint data: extract measurement types and values
            measurement_params = self.extract_measurement_parameters(plate)
            self._log('debug', f"Measurement types: {measurement_params}")
            
            for measurement_type in measurement_params:
                results = plate.get_raw_data()
                for i, well in enumerate(wells):
                    plate_data.at[well, measurement_type] = (list(results[1]['value'])[i])
                    
        return plate_data
    
    def determine_read_type(self, plate):
        """
        Determine the type of read protocol loaded on the plate.
        
        Analyzes the XML procedure to classify the measurement type.
        
        Args:
            plate: Biotek plate object with loaded procedure
            
        Returns:
            str: Protocol type - 'spectra', 'read', or 'Unknown'
                - 'spectra': Wavelength scanning (spectral analysis)
                - 'read': Endpoint measurements (absorbance/fluorescence)
                - 'Unknown': Unrecognized protocol format
        """
        current_procedure = plate.get_procedure()
        root = ET.fromstring(current_procedure)

        # Find the ReadStep element to determine protocol type
        read_step = root.find(".//ReadStep")
        if read_step is None:
            return "Unknown"

        # Get detection and read type parameters
        detection = read_step.findtext("Detection")
        read_type = read_step.findtext("ReadType")

        # Classify based on read type
        if read_type == "Spectrum":
            return "spectra"
        elif read_type == "EndPoint":
            return "read"
        else:
            return "Unknown"

    def run_protocol(self, protocol_path, wells=None, plate_type="96 WELL PLATE"):
        """
        Execute a complete protocol file on specified wells.
        
        Loads a Biotek protocol file (.prt) and executes it on the specified wells.
        Handles well grouping for sequential reads and combines data appropriately.
        
        Args:
            protocol_path (str): Path to the Biotek protocol file (.prt)
            wells (list, optional): List of well indices to measure (0-based)
                If None, measures all wells according to protocol
            plate_type (str): Plate format - "96 WELL PLATE" or "48 WELL PLATE"
            
        Returns:
            pandas.DataFrame: Combined measurement data from all wells
                - For spectral protocols: Wavelengths vs wells
                - For endpoint protocols: Wells vs measurement parameters
                
        Raises:
            RuntimeError: If experiment creation fails
        """
        self.biotek.app.data_export_enabled = True
        experiment = self.biotek.new_experiment(protocol_path)
        protocol_data = pd.DataFrame()

        if experiment:
            self._log('debug', "Experiment created successfully.")
            self._log('debug', f"Experiment Protocol Type: {experiment.protocol_type}")
            plates = experiment.plates
            
            if wells is not None:
                # Group wells for sequential reads (handles plate layout constraints)
                grouped_wells = self.group_wells(wells, plate_type)
            else:
                # Use all wells if none specified
                if plate_type == "96 WELL PLATE":
                    grouped_wells = [self.well_index_to_label(i, 12) for i in range(96)]
                else:  # 48 WELL PLATE
                    grouped_wells = [self.well_index_to_label(i, 8) for i in range(48)]
                grouped_wells = [grouped_wells]  # Single group with all wells
                
            # Process each group of wells sequentially
            for well_group in grouped_wells:
                plate = plates.add()
                prot_type = self.determine_read_type(plate)
                data_group = self.run_plate(plate, well_group, prot_type=prot_type)
                
                # Combine data based on protocol type
                if protocol_data.empty:
                    protocol_data = data_group
                elif prot_type == "spectra":
                    # Merge spectral data on wavelength column
                    protocol_data = protocol_data.merge(data_group, on="Wavelengths", how='outer')     
                elif prot_type == "read":
                    # Concatenate endpoint data
                    protocol_data = pd.concat([protocol_data, data_group])      
        else:
            self._log('error', "Experiment creation failed.")
            raise RuntimeError("Failed to create experiment from protocol file")

        return protocol_data

    def well_index_to_label(self, index, nums_per_letter):
        """
        Convert well index to standard plate notation.
        
        Converts 0-based well indices to standard plate labels (A1, A2, B1, etc.)
        
        Args:
            index (int): 0-based well index (0 = A1, 1 = A2, 12 = B1, etc.)
            nums_per_letter (int): Wells per row (12 for 96-well, 8 for 48-well)
            
        Returns:
            str: Well label in standard format (e.g., 'A1', 'B12', 'H8')
            
        Example:
            >>> well_index_to_label(0, 12)   # First well of 96-well plate
            'A1'
            >>> well_index_to_label(13, 12)  # Second row, second well
            'B2'
        """
        row = string.ascii_uppercase[index // nums_per_letter]  # Get row letter (A-H)
        col = (index % nums_per_letter) + 1  # Get column number (1-12)
        return f"{row}{col}"

    def group_wells(self, indices, plate_type):
        """
        Group well indices for efficient sequential reading.
        
        Groups wells into contiguous sequences that can be read together,
        respecting plate layout constraints. Avoids reading across row boundaries
        which could cause issues with the plate reader's partial plate functionality.
        
        Args:
            indices (list): List of 0-based well indices to group
            plate_type (str): Plate format - "96 WELL PLATE" or "48 WELL PLATE"
            
        Returns:
            list: List of well groups, where each group is a list of well labels
                Example: [['A1', 'A2', 'A3'], ['A5', 'A6'], ['B1']]
                
        Note:
            Wells are grouped into contiguous sequences within the same row.
            This prevents reading across row boundaries which could cause
            issues with the Biotek's partial plate reading functionality.
        """
        indices = sorted(indices)  # Ensure sorted order for proper grouping
        grouped = []
        current_group = []

        self._log('debug', f"Grouping wells for plate type: {plate_type}")

        # Determine wells per row based on plate type
        if plate_type == "96 WELL PLATE":
            nums_per_letter = 12  # A1-A12, B1-B12, etc.
        elif plate_type == "48 WELL PLATE":
            nums_per_letter = 8   # A1-A8, B1-B8, etc.
        else:
            self._log('error', f"Unsupported plate type: {plate_type}")
            raise ValueError(f"Unsupported plate type: {plate_type}")
        
        # Group contiguous wells within same row
        for i in range(len(indices)):
            if not current_group:
                # Start new group
                current_group.append(indices[i])
            else:
                prev = current_group[-1]
                # Check if current well is contiguous and in same row
                if indices[i] == prev + 1 and (indices[i] % nums_per_letter != 0):
                    current_group.append(indices[i])
                else:
                    # End current group and start new one
                    grouped.append([self.well_index_to_label(idx, nums_per_letter) for idx in current_group])
                    current_group = [indices[i]]
        
        # Add final group if it exists
        if current_group:
            grouped.append([self.well_index_to_label(idx, nums_per_letter) for idx in current_group])
        
        return grouped
