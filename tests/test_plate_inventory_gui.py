"""Run with a PySide6 environment; offscreen tests never connect to hardware."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from vial_manager_gui import TrackStatusWidget


@pytest.fixture
def editor(tmp_path):
    app = QApplication.instance() or QApplication([])
    path = tmp_path / "track_status.yaml"
    path.write_bytes((ROOT / "tests/fixtures/mixed_wellplate_track_status.yaml").read_bytes())
    widget = TrackStatusWidget(path, show_load_errors=False)
    yield widget, path, app
    widget.close()


def select(widget, position):
    widget.select_position(position)
    assert widget.selected_position() == position


def test_detail_buttons_follow_selection_and_stack_order(editor):
    widget, path, app = editor
    select(widget, "waste_stack")
    assert widget.add_button.isEnabled()
    for button in (widget.edit_button, widget.remove_button, widget.move_button,
                   widget.higher_button, widget.lower_button):
        assert not button.isEnabled()
    select(widget, "source_stack")
    assert widget.edit_button.isEnabled()
    assert widget.remove_button.isEnabled()
    assert widget.move_button.isEnabled()
    assert not widget.higher_button.isEnabled()
    assert widget.lower_button.isEnabled()
    widget._select_object("plate-24")
    assert widget.higher_button.isEnabled()
    assert not widget.lower_button.isEnabled()
    widget.higher_button.click()
    assert widget.inventory.contents("source_stack") == ["plate-96", "plate-24"]
    assert not widget.higher_button.isEnabled()
    assert widget.lower_button.isEnabled()


@pytest.mark.parametrize("width", [600, 1000])
def test_stack_holders_are_double_height_and_pack_without_overlap(editor, width):
    widget, path, app = editor
    widget.diagram.resize(width, 800)
    rectangles = widget.diagram.card_rectangles()
    assert rectangles["source_stack"].height() == 2 * rectangles["pipetting_area"].height()
    assert rectangles["waste_stack"].height() == 2 * rectangles["gripper"].height()
    for name, rect in rectangles.items():
        assert rect.right() <= widget.diagram.width()
        assert rect.bottom() <= widget.diagram.minimumHeight()
        assert all(not rect.intersects(other) for key, other in rectangles.items() if key != name)


def test_long_stack_grows_only_its_holder_and_all_layers_remain_visible(editor):
    widget, path, app = editor
    widget.add_item("source_stack", "24 WELL PLATE", quantity=12)
    widget._populate_form()
    show_diagram(widget, app)
    rectangles = widget.diagram.card_rectangles()
    source = rectangles["source_stack"]
    assert source.height() > rectangles["waste_stack"].height()
    assert rectangles["pipetting_area"].height() == 200
    layers = [(rect, item) for rect, location, item in widget.diagram._hits
              if location == "source_stack" and item is not None]
    assert len(layers) == 14
    for rect, item in layers:
        assert source.contains(rect)
        assert rect.top() > source.top() + 55
    # The top layer remains selectable and draggable after the holder expands.
    destination = rectangles["pipetting_area"].center()
    top = widget.inventory.top("source_stack")
    assert drop_item(widget, app, top, destination)
    assert widget.inventory.top("pipetting_area") == top


@pytest.mark.parametrize("capacity", [2, None])
def test_other_positions_use_configured_capacity_for_display_height(editor, capacity):
    widget, path, app = editor
    widget.inventory.position_config["positions"]["pipetting_area"]["max_wellplates"] = capacity
    widget._populate_form()
    rectangles = widget.diagram.card_rectangles()
    assert rectangles["pipetting_area"].height() == 400
    assert rectangles["cytation_tray"].height() == 200


def test_position_editor_reorders_mixed_source_and_preserves_pose(editor):
    widget, path, app = editor
    assert widget.state_loaded
    select(widget, "source_stack")
    assert widget.contents_tree.topLevelItemCount() == 2
    pose = dict(widget.inventory.data["track_pose"])
    widget._reorder(-1)
    assert widget.inventory.contents("source_stack") == ["plate-96", "plate-24"]
    assert widget._save_track_status(silent=True)
    assert widget._load_track_status()
    assert widget.inventory.contents("source_stack") == ["plate-96", "plate-24"]
    assert widget.inventory.data["track_pose"] == pose
    holder_names = [widget.position_selector.itemData(i)
                    for i in range(widget.position_selector.count())]
    assert "cytation_safe_area" not in holder_names


def test_gripper_four_states_and_lid_relocation(editor):
    widget, path, app = editor
    for kind, with_lid in (("lid", False), ("plate", False), ("plate", True)):
        created = widget.add_item("gripper", "96 WELL PLATE", kind, with_lid)[0]
        assert widget._save_track_status(silent=True)
        assert widget._load_track_status()
        assert widget.inventory.positions["gripper"] == created
        assert (created in widget.inventory.attachments) == with_lid
        if with_lid:
            lid = widget.inventory.attachments[created]
            widget.inventory.relocate(lid, "lid_storage_96")
            assert widget.inventory.positions["gripper"] == created
            assert widget.inventory.positions["lid_storage_96"] == lid
            widget.inventory.remove(lid)
        widget.inventory.remove(created)
    assert widget._save_track_status(silent=True)
    assert yaml.safe_load(path.read_text())["positions"]["gripper"] is None


def test_add_remove_relocate_capacity_and_no_implicit_writes(editor):
    widget, path, app = editor
    original = path.read_bytes()
    created = widget.add_item("pipetting_area", "48 WELL PLATE", with_lid=True)[0]
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="capacity"):
        widget.add_item("pipetting_area", "24 WELL PLATE")
    assert widget.inventory.contents("pipetting_area") == [created]
    widget.inventory.relocate(created, "gripper")
    assert widget.inventory.contents("pipetting_area") == []
    widget.inventory.remove(created)
    assert widget.inventory.attachments == {}
    assert widget._save_track_status(silent=True)


def test_legacy_review_save_backs_up_and_does_not_regenerate_ids(editor):
    widget, path, app = editor
    legacy = {"wellplate_type": "24 WELL PLATE", "num_in_source": 2, "num_in_waste": 0}
    path.write_text(yaml.safe_dump(legacy))
    assert widget._load_track_status()
    ids = widget.inventory.contents("source_stack")
    assert yaml.safe_load(path.read_text()) == legacy
    assert widget._save_track_status(silent=True)
    assert widget._load_track_status()
    assert widget.inventory.contents("source_stack") == ids
    assert yaml.safe_load(path.with_suffix(".yaml.legacy.bak").read_text()) == legacy


def test_diagram_click_selects_plate_without_changing_recorded_state(editor):
    from PySide6.QtTest import QTest
    widget, path, app = editor
    widget.resize(1320, 700)
    widget.show()
    app.processEvents()
    before = widget.state_snapshot()
    hits = [(rect, position, item) for rect, position, item in widget.diagram._hits if item == "plate-96"]
    assert len(hits) == 1
    rect, position, _ = hits[0]
    QTest.mouseClick(widget.diagram, Qt.LeftButton, pos=rect.center().toPoint())
    assert widget.selected_position() == "source_stack"
    assert widget.selected_object() == "plate-96"
    assert widget.state_snapshot() == before
    assert not widget.recovery_panel.isVisible()
    assert widget.edit_button.isEnabled()


def test_recovery_only_appears_for_interrupted_transfer(editor):
    widget, path, app = editor
    widget.show()
    app.processEvents()
    assert not widget.recovery_panel.isVisible()
    widget.inventory.begin("plate-96", "source_stack", "gripper")
    widget._populate_form()
    assert widget.recovery_panel.isVisible()
    assert "Check the physical setup" in widget.recovery_label.text()
    widget.confirm_positions.click()
    assert widget.inventory.data["pending_transfer"] is None
    assert not widget.recovery_panel.isVisible()


def test_repeated_reordering_keeps_the_same_plate_selected(editor):
    widget, path, app = editor
    extra = widget.add_item("source_stack", "48 WELL PLATE")[0]
    widget._populate_form()
    select(widget, "source_stack")
    widget._select_object(extra)
    widget._guard(lambda: widget._reorder(-1))
    assert widget.selected_object() == extra
    widget._guard(lambda: widget._reorder(-1))
    assert widget.inventory.contents("source_stack")[0] == extra


def show_diagram(widget, app):
    widget.resize(1320, 700)
    widget.show()
    app.processEvents()


def drop_item(widget, app, object_id, point, mime=None):
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QDragEnterEvent, QDropEvent
    view = widget.diagram
    mime = mime or view._mime_data(object_id)
    entered = QDragEnterEvent(point.toPoint(), Qt.MoveAction, mime, Qt.LeftButton, Qt.NoModifier)
    app.sendEvent(view, entered)
    if not entered.isAccepted():
        return False
    event = QDropEvent(QPointF(point), Qt.MoveAction, mime, Qt.LeftButton, Qt.NoModifier)
    app.sendEvent(view, event)
    app.processEvents()
    return event.isAccepted()


def test_drag_lidded_plate_to_holder_preserves_id_lid_and_unsaved_state(editor):
    widget, path, app = editor
    lid = widget.inventory.add("source_stack", "96 WELL PLATE", kind="lid")
    widget._populate_form()
    show_diagram(widget, app)
    before = path.read_bytes()
    destination = widget.diagram.card_rectangles()["pipetting_area"].center()
    assert drop_item(widget, app, "plate-96", destination)
    assert widget.inventory.contents("source_stack") == ["plate-24"]
    assert widget.inventory.contents("pipetting_area") == ["plate-96"]
    assert widget.inventory.attachments["plate-96"] == lid
    assert widget.selected_position() == "pipetting_area"
    assert widget.selected_object() == "plate-96"
    assert path.read_bytes() == before
    assert widget._save_track_status(silent=True)
    assert widget._load_track_status()
    assert widget.inventory.attachments["plate-96"] == lid


def test_drag_changes_stack_order_and_keeps_selection(editor):
    widget, path, app = editor
    show_diagram(widget, app)
    rect = next(rect for rect, _, item in widget.diagram._hits if item == "plate-24")
    point = rect.bottomRight()
    point.setX(rect.center().x())
    point.setY(rect.bottom()-2)
    assert drop_item(widget, app, "plate-96", point)
    assert widget.inventory.contents("source_stack") == ["plate-96", "plate-24"]
    assert widget.selected_object() == "plate-96"


def test_drag_inserts_into_middle_of_another_stack(editor):
    widget, path, app = editor
    lower, upper = widget.add_item("waste_stack", "48 WELL PLATE", quantity=2)
    widget._populate_form()
    show_diagram(widget, app)
    rect = next(rect for rect, _, item in widget.diagram._hits if item == lower)
    point = rect.topLeft()
    point.setX(rect.center().x())
    point.setY(rect.top()+2)
    assert drop_item(widget, app, "plate-96", point)
    assert widget.inventory.contents("waste_stack") == [lower, "plate-96", upper]


def test_drag_to_full_holder_is_rejected_without_mutation(editor):
    widget, path, app = editor
    widget.add_item("pipetting_area", "48 WELL PLATE")
    widget._populate_form()
    show_diagram(widget, app)
    before = widget.state_snapshot()
    point = widget.diagram.card_rectangles()["pipetting_area"].center()
    assert not drop_item(widget, app, "plate-96", point)
    assert widget.state_snapshot() == before
    assert "capacity" in widget.feedback.text()
    assert widget.diagram._drop_target[2] is False


def test_drag_attached_lid_to_storage_keeps_plate_in_place(editor):
    widget, path, app = editor
    lid = widget.inventory.add("source_stack", "96 WELL PLATE", kind="lid")
    widget._populate_form()
    show_diagram(widget, app)
    assert any(item == lid for _, _, item in widget.diagram._hits)
    point = widget.diagram.card_rectangles()["lid_storage_96"].center()
    assert drop_item(widget, app, lid, point)
    assert widget.inventory.contents("source_stack") == ["plate-24", "plate-96"]
    assert widget.inventory.attachments == {}
    assert widget.inventory.positions["lid_storage_96"] == lid
    point = widget.diagram.card_rectangles()["source_stack"].center()
    assert drop_item(widget, app, lid, point)
    assert widget.inventory.attachments["plate-96"] == lid


def test_drag_plate_into_gripper_and_back(editor):
    widget, path, app = editor
    show_diagram(widget, app)
    point = widget.diagram.card_rectangles()["gripper"].center()
    assert drop_item(widget, app, "plate-96", point)
    assert widget.inventory.positions["gripper"] == "plate-96"
    assert not drop_item(widget, app, "plate-24", point)
    assert widget.inventory.positions["gripper"] == "plate-96"
    point = widget.diagram.card_rectangles()["waste_stack"].center()
    assert drop_item(widget, app, "plate-96", point)
    assert widget.inventory.positions["gripper"] is None
    assert widget.inventory.contents("waste_stack") == ["plate-96"]


def test_cancelled_drag_and_stale_or_foreign_drag_do_not_edit_state(editor):
    from PySide6.QtGui import QDragEnterEvent, QDragLeaveEvent
    widget, path, app = editor
    show_diagram(widget, app)
    view = widget.diagram
    point = view.card_rectangles()["waste_stack"].center()
    before = widget.state_snapshot()
    mime = view._mime_data("plate-96")
    enter = QDragEnterEvent(point.toPoint(), Qt.MoveAction, mime, Qt.LeftButton, Qt.NoModifier)
    app.sendEvent(view, enter)
    assert enter.isAccepted()
    app.sendEvent(view, QDragLeaveEvent())
    assert widget.state_snapshot() == before
    assert view._drop_target is None
    widget.inventory.relocate("plate-96", "pipetting_area")
    widget._populate_form()
    before = widget.state_snapshot()
    assert not drop_item(widget, app, "plate-96", point, mime)
    assert widget.state_snapshot() == before
    import json
    from robot_state.plate_layout_view import TRACK_ITEM_MIME
    mime = view._mime_data("plate-96")
    payload = json.loads(bytes(mime.data(TRACK_ITEM_MIME)))
    payload["owner"] = "another-editor"
    mime.setData(TRACK_ITEM_MIME, json.dumps(payload).encode())
    assert not drop_item(widget, app, "plate-96", point, mime)
    assert widget.state_snapshot() == before


def test_mouse_drag_starts_only_after_drag_threshold(editor, monkeypatch):
    from PySide6.QtCore import QEvent, QPointF
    from PySide6.QtGui import QDrag, QMouseEvent
    from PySide6.QtTest import QTest
    from robot_state.plate_layout_view import TRACK_ITEM_MIME
    widget, path, app = editor
    show_diagram(widget, app)
    view = widget.diagram
    point = next(rect.center() for rect, _, item in view._hits if item == "plate-96")
    calls = []
    def execute_drag(drag, action):
        assert drag.mimeData().hasFormat(TRACK_ITEM_MIME)
        assert not drag.pixmap().isNull()
        calls.append(action)
        return Qt.IgnoreAction
    monkeypatch.setattr(QDrag, "exec", execute_drag)
    before = widget.state_snapshot()
    QTest.mousePress(view, Qt.LeftButton, pos=point.toPoint())
    near = point + QPointF(1, 0)
    app.sendEvent(view, QMouseEvent(QEvent.MouseMove, near, near, Qt.NoButton, Qt.LeftButton, Qt.NoModifier))
    assert calls == []
    far = point + QPointF(app.startDragDistance()+4, 0)
    app.sendEvent(view, QMouseEvent(QEvent.MouseMove, far, far, Qt.NoButton, Qt.LeftButton, Qt.NoModifier))
    assert calls == [Qt.MoveAction]
    assert widget.state_snapshot() == before
    QTest.mouseRelease(view, Qt.LeftButton, pos=far.toPoint())
