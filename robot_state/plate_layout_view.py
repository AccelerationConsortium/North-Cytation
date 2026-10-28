"""Clickable schematic of the recorded plate holders; no hardware dependencies."""

import json
from uuid import uuid4

from PySide6.QtCore import Qt, Signal, QRectF, QSize, QMimeData, QPoint
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QDrag
from PySide6.QtWidgets import QWidget, QApplication

TRACK_ITEM_MIME = "application/x-north-track-item"


POSITION_LABELS = {
    "source_stack": "Source stack", "waste_stack": "Waste stack",
    "pipetting_area": "Active area", "cytation_tray": "Cytation",
    "lid_storage": "Lid storage", "lid_storage_96": "96-well lid storage",
    "gripper": "Track gripper",
}


def position_label(location):
    return POSITION_LABELS.get(location, location.replace("_", " ").capitalize())


def plate_label(plate_type):
    return plate_type.replace(" WELL PLATE", "-well").replace("quartz", "Quartz")


class PlateLayoutView(QWidget):
    position_activated = Signal(str, object)
    item_dropped = Signal(str, str, object)
    drop_feedback = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.inventory = None
        self.selected_position = "source_stack"
        self.selected_object = None
        self._hits = []
        self._drag_owner = uuid4().hex
        self._pressed = None
        self._drop_target = None
        self.setAcceptDrops(True)
        self.setMinimumWidth(430)
        self.setMouseTracking(True)
        self.setAccessibleName("Plate positions diagram")
        self.setToolTip("Select a plate to edit it, or drag it to another holder or stack layer. No hardware moves.")

    def sizeHint(self):
        return QSize(850, 440)

    def set_inventory(self, inventory, selected_position, selected_object=None):
        self.inventory = inventory
        self.selected_position = selected_position
        self.selected_object = selected_object
        self._resize_canvas()
        self.update()

    def resizeEvent(self, event):
        self._resize_canvas()
        super().resizeEvent(event)

    def _locations(self):
        available = list(self.inventory.holders()) + ["gripper"] if self.inventory else []
        preferred = list(POSITION_LABELS)
        return [name for name in preferred if name in available] + [name for name in available if name not in preferred]

    def _resize_canvas(self):
        height = int(max((rect.bottom() for rect in self.card_rectangles().values()), default=200) + 12)
        if self.minimumHeight() != height:
            self.setMinimumHeight(height)

    def card_rectangles(self):
        columns = 4 if self.width() >= 740 else 2
        gap, margin = 12, 12
        width = (self.width() - 2 * margin - (columns-1)*gap) / columns
        bottoms = [margin] * columns
        rectangles = {}
        for name in self._locations():
            is_plate_holder = self.inventory.holders().get(name) == "plate"
            capacity = self.inventory.capacity(name) if is_plate_holder else 1
            stack_holder = is_plate_holder and (capacity is None or capacity > 1)
            height = max(400 if stack_holder else 200, 96 + len(self.inventory.contents(name)) * 34)
            # Pack shorter holders beside stacks; only the full holder grows.
            column = min(range(columns), key=bottoms.__getitem__)
            rectangles[name] = QRectF(margin + column * (width + gap), bottoms[column], width, height)
            bottoms[column] += height + gap
        return rectangles

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#f3f6fa"))
        self._hits = []
        if not self.inventory:
            return
        for location, rect in self.card_rectangles().items():
            selected = location == self.selected_position
            target = self._drop_target if self._drop_target and self._drop_target[0] == location else None
            border_color = ("#16a34a" if target[2] else "#dc2626") if target else ("#2563eb" if selected else "#d7dee8")
            painter.setPen(QPen(QColor(border_color), 2 if selected or target else 1))
            painter.setBrush(QColor(("#f0fdf4" if target[2] else "#fff1f2") if target else "#ffffff"))
            painter.drawRoundedRect(rect, 10, 10)
            self._hits.append((rect, location, None))
            font = QFont(self.font())
            font.setBold(True)
            font.setPointSize(10)
            painter.setFont(font)
            painter.setPen(QColor("#243247"))
            painter.drawText(rect.adjusted(12, 12, -9, -rect.height()+36), Qt.AlignLeft | Qt.AlignVCenter, position_label(location))
            items = self.inventory.contents(location)
            kind = self.inventory.holders().get(location)
            maximum = self.inventory.capacity(location) if kind == "plate" else None
            summary = (f"{len(items)} of {maximum} plates" if maximum is not None else f"{len(items)} plate{'s' if len(items) != 1 else ''}") if kind == "plate" else ("Empty" if not items else "Lid" if self.inventory.objects[items[0]]["kind"] == "lid" else "Plate held")
            font.setBold(False)
            font.setPointSize(9)
            painter.setFont(font)
            painter.setPen(QColor("#69788c"))
            painter.drawText(rect.adjusted(12, 36, -10, -rect.height()+55), Qt.AlignLeft | Qt.AlignVCenter, summary)
            base = rect.bottom()-17
            painter.setPen(QPen(QColor("#c3cedb"), 3))
            painter.drawLine(int(rect.left()+14), int(base+2), int(rect.right()-14), int(base+2))
            if not items:
                painter.setPen(QPen(QColor("#d5dde8"), 1, Qt.DashLine))
                painter.setBrush(QColor("#f8fafc"))
                ghost = QRectF(rect.left()+14, base-34, rect.width()-28, 30)
                painter.drawRoundedRect(ghost, 5, 5)
                painter.setPen(QColor("#8b98aa"))
                painter.drawText(ghost, Qt.AlignCenter, "Empty")
            for index, object_id in enumerate(items):
                plate_type = self.inventory.plate_type(object_id)
                obj = self.inventory.objects[object_id]
                slab = QRectF(rect.left()+14, base-32-index*34, rect.width()-28, 30)
                if obj["kind"] == "lid":
                    slab = QRectF(slab.left(), slab.bottom()-12, slab.width(), 12)
                colors = {"96 WELL PLATE": ("#dbeafe", "#60a5fa"), "24 WELL PLATE": ("#dcfce7", "#4ade80"),
                          "48 WELL PLATE": ("#ede9fe", "#a78bfa"), "quartz": ("#fef3c7", "#fbbf24")}
                fill, border = colors.get(plate_type, ("#e2e8f0", "#94a3b8"))
                painter.setBrush(QColor(fill))
                painter.setPen(QPen(QColor("#1d4ed8" if object_id == self.selected_object else border), 2 if object_id == self.selected_object else 1))
                painter.drawRoundedRect(slab, 5, 5)
                if object_id in self.inventory.attachments:
                    painter.setPen(QPen(QColor(border), 4))
                    painter.drawLine(int(slab.left()+4), int(slab.top()+1), int(slab.right()-4), int(slab.top()+1))
                painter.setPen(QColor("#243247"))
                label = plate_label(plate_type) + (" lid" if obj["kind"] == "lid" else "")
                if object_id in self.inventory.attachments:
                    label += " + lid"
                if obj["kind"] == "plate":
                    # Small well pattern gives each layer a recognizable plate shape.
                    painter.setBrush(QColor(border))
                    painter.setPen(Qt.NoPen)
                    for row in range(3):
                        for column in range(5):
                            painter.drawEllipse(QRectF(slab.left()+7+column*4, slab.top()+7+row*5, 2.5, 2.5))
                    painter.setPen(QColor("#243247"))
                text_rect = slab.adjusted(32 if obj["kind"] == "plate" else 5, 0, -4, 0)
                painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, label)
                self._hits.append((slab, location, object_id))
                if object_id in self.inventory.attachments:
                    lid = self.inventory.attachments[object_id]
                    cap = QRectF(slab.left()+3, slab.top()-2, slab.width()-6, 8)
                    self._hits.append((cap, location, lid))
                if index == len(items)-1 and len(items) > 1:
                    painter.setPen(QColor("#52647b"))
                    painter.drawText(QRectF(slab.left(), slab.top()-17, slab.width(), 14), Qt.AlignRight | Qt.AlignVCenter, "TOP / next")
            if target and target[2] and target[1] is not None:
                index, moving = target[1], target[3]
                remaining = [item for item in items if item != moving]
                if index == 0:
                    y = base+1
                else:
                    lower = remaining[index-1]
                    y = next(hit.top()-2 for hit, where, item in self._hits if where == location and item == lower)
                painter.setPen(QPen(QColor("#16a34a"), 3))
                painter.drawLine(int(rect.left()+10), int(y), int(rect.right()-10), int(y))

    def _hit(self, point):
        return next(((rect, location, object_id) for rect, location, object_id in reversed(self._hits)
                     if rect.contains(point)), None)

    def _mime_data(self, object_id):
        mime = QMimeData()
        mime.setData(TRACK_ITEM_MIME, json.dumps({"owner": self._drag_owner, "object_id": object_id,
                     "source": self.inventory.locate(object_id)}).encode("utf-8"))
        return mime

    def _drag_object(self, mime):
        if not self.inventory or not mime.hasFormat(TRACK_ITEM_MIME):
            return None
        try:
            payload = json.loads(bytes(mime.data(TRACK_ITEM_MIME)))
            object_id = payload["object_id"]
            if (payload["owner"] != self._drag_owner or object_id not in self.inventory.objects
                    or self.inventory.locate(object_id) != payload["source"]):
                return None
            return object_id
        except (ValueError, KeyError, TypeError, UnicodeDecodeError):
            return None

    def _proposal(self, point, object_id):
        hit = self._hit(point)
        if hit is None:
            return None
        _, location, _ = hit
        index = None
        if self.inventory.objects[object_id]["kind"] == "plate" and isinstance(self.inventory.positions[location], list):
            contents = [item for item in self.inventory.contents(location) if item != object_id]
            rectangles = {item: rect for rect, where, item in self._hits if where == location and item in contents}
            if any(item not in rectangles for item in contents):
                return None  # Wait for a repaint if the setup changed mid-drag.
            index = sum(point.y() < rectangles[item].center().y() for item in contents)
        return location, index

    def _check_drop(self, event):
        object_id = self._drag_object(event.mimeData())
        proposal = self._proposal(event.position(), object_id) if object_id else None
        if proposal is None:
            self._drop_target = None
            event.ignore()
            self.update()
            return None
        location, index = proposal
        try:
            self.inventory.check_relocation(object_id, location, index)
        except (ValueError, KeyError, TypeError) as error:
            self._drop_target = (location, index, False, object_id)
            self.drop_feedback.emit(str(error))
            event.ignore()
            self.update()
            return None
        self._drop_target = (location, index, True, object_id)
        self.drop_feedback.emit(f"Move to {position_label(location)}" + (f" · layer {index+1} from the bottom" if index is not None else ""))
        event.setDropAction(Qt.MoveAction)
        event.accept()
        self.update()
        return object_id, location, index

    def dragEnterEvent(self, event):
        self._check_drop(event)

    def dragMoveEvent(self, event):
        self._check_drop(event)

    def dragLeaveEvent(self, event):
        self._drop_target = None
        self.update()
        event.accept()

    def dropEvent(self, event):
        proposal = self._check_drop(event)
        self._drop_target = None
        self.update()
        if proposal:
            self.item_dropped.emit(*proposal)

    def _start_drag(self, object_id):
        drag = QDrag(self)
        drag.setMimeData(self._mime_data(object_id))
        rect = next(rect for rect, _, item in self._hits if item == object_id)
        pixmap = self.grab(rect.adjusted(-3, -3, 3, 3).toAlignedRect())
        drag.setPixmap(pixmap)
        drag.setHotSpot(QPoint(pixmap.width()//2, pixmap.height()//2))
        try:
            drag.exec(Qt.MoveAction)
        finally:
            self._drop_target = None
            self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            hit = self._hit(event.position())
            if hit:
                _, location, object_id = hit
                self._pressed = (object_id, event.position().toPoint()) if object_id else None
                self.position_activated.emit(location, object_id)
                return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self._pressed = None
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        if self._pressed and event.buttons() & Qt.LeftButton:
            object_id, start = self._pressed
            if (event.position().toPoint()-start).manhattanLength() >= QApplication.startDragDistance():
                self._pressed = None
                self._start_drag(object_id)
                return
        for rect, location, object_id in reversed(self._hits):
            if rect.contains(event.position()):
                self.setCursor(Qt.PointingHandCursor)
                text = position_label(location)
                if object_id:
                    text += f"\n{object_id}\n{self.inventory.plate_type(object_id)}"
                    text += "\nLid attached" if object_id in self.inventory.attachments else ""
                    text += "\nDrag to move this recorded item."
                self.setToolTip(text)
                return
        self.unsetCursor()
        super().mouseMoveEvent(event)
