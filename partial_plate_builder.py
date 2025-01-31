import xml.etree.ElementTree as ET
from typing import Optional, List
from ..utils.logger import logger  # Import the configured logger


def build_bti_partial_plate_xml(
    version: str = "1.00",
    single_block: bool = True,
    first_well: str = "B2",
    last_well: str = "F6",
    wells: Optional[List[str]] = None
) -> str:
    """
    Constructs an XML string in the BTIPartialPlate format.

    4.7 BTIPartialPlate format examples:

    Single-block example:
        <BTIPartialPlate Version="1.00">
            <SingleBlock>Yes</SingleBlock>
            <FirstWell>B2</FirstWell>
            <LastWell>F6</LastWell>
        </BTIPartialPlate>

    Random-well example:
        <BTIPartialPlate Version="1.00">
            <SingleBlock>No</SingleBlock>
            <Wells>
                <Well>A1</Well>
                <Well>B3</Well>
                <Well>H12</Well>
            </Wells>
        </BTIPartialPlate>

    :param version: The version attribute for the BTIPartialPlate root. Defaults to "1.00".
    :param single_block: Whether to generate a single-block XML (True) or a random-well XML (False).
    :param first_well: Used if single_block=True. For example, "B2".
    :param last_well:  Used if single_block=True. For example, "F6".
    :param wells:      Used if single_block=False. A list of wells, e.g. ["A1", "B3", "H12"].
    :return:           An XML string ready for Gen5's SetPartialPlate method.
    """
    try:
        logger.info("Starting to build BTIPartialPlate XML.")

        # Create the root element with an attribute "Version"
        root = ET.Element("BTIPartialPlate", attrib={"Version": version})
        logger.debug(f"Created root <BTIPartialPlate> element with Version='{version}'.")

        # <SingleBlock>Yes|No</SingleBlock>
        single_block_elem = ET.SubElement(root, "SingleBlock")
        single_block_text = "Yes" if single_block else "No"
        single_block_elem.text = single_block_text
        logger.debug(f"Set <SingleBlock> to '{single_block_text}'.")

        if single_block:
            # Single-block example
            ET.SubElement(root, "FirstWell").text = first_well
            ET.SubElement(root, "LastWell").text = last_well
            logger.debug(f"Added <FirstWell>='{first_well}' and <LastWell>='{last_well}'.")
        else:
            # Random-well example
            if wells is None:
                wells = ["A1", "B3", "H12"]  # Default set if not provided
                logger.debug("No wells provided. Using default wells ['A1', 'B3', 'H12'].")
            else:
                logger.debug(f"Wells provided: {wells}")

            wells_elem = ET.SubElement(root, "Wells")
            for well in wells:
                well_elem = ET.SubElement(wells_elem, "Well")
                well_elem.text = well
                logger.debug(f"Added <Well>='{well}'.")

        # Convert the ElementTree to an XML string with declaration
        xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        xml_str = xml_bytes.decode("utf-8")  # Convert bytes to string
        logger.info("BTIPartialPlate XML built successfully.")
        return xml_str

    except Exception as e:
        logger.exception(f"Unexpected error while building BTIPartialPlate XML: {e}")
        raise
