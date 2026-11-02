import os
import sys
from pathlib import Path

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PySide6.QtWidgets import QApplication, QWidget, QHBoxLayout
from PySide6.QtGui import QFontDatabase, QFont
from vial_manager_gui import VialWidget, VialRackWidget, CombinedRackWidget

app = QApplication([])
font_id = QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf')
if font_id >= 0:
    app.setFont(QFont(QFontDatabase.applicationFontFamilies(font_id)[0]))
window = QWidget()
layout = QHBoxLayout(window)
cases = [
    ('Empty', 0, 8, True, 'closed', '0 / 8 mL'),
    ('Quarter', 2, 8, True, 'open', '2 / 8 mL'),
    ('Half full', 4, 8, True, 'closed', '4 / 8 mL'),
    ('Large vial', 10, 20, True, 'open', '10 / 20 mL'),
    ('Full', 8, 8, False, 'open', '8 / 8 mL'),
    ('Over capacity', 9, 8, True, 'closed', '9 / 8 mL'),
    ('Unknown', 'bad', 8, True, 'open', 'Volume unknown'),
    ('No capacity', 4, None, True, 'closed', '4 mL'),
]
for name, volume, capacity, capped, cap_type, expected in cases:
    tile = VialWidget(dict(vial_name=name, vial_volume=volume, capped=capped, cap_type=cap_type),
                      capacity_ml=capacity)
    assert expected in tile.volume_label.text(), (name, tile.volume_label.text())
    assert not tile.fill_icon.pixmap().isNull()
    assert tile.cap_icon.isHidden() == (not capped)
    layout.addWidget(tile)

# Verify rack capacity reaches the actual tiles through both rendering paths.
data = dict(vial_name='Capacity check', vial_volume=10, capped=True, cap_type='open',
            location='large_vial_rack', location_index=0)
for rack in (VialRackWidget('large_vial_rack'), CombinedRackWidget()):
    rack.add_vials([data])
    tile = next(iter(rack.vials.values()))
    assert tile.capacity_ml == 20
    assert '10 / 20 mL' in tile.volume_label.text()
    tile.update_vial_data(dict(data, vial_volume=5))
    assert '5 / 20 mL' in tile.volume_label.text()

window.show()
app.processEvents()
window.grab().save(str(Path(__file__).with_name('vial_fill_preview.png')))
print('Passed: fill levels, unknown values, cap states, rack capacities, and updates')

