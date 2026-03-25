import copy
import argparse
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import cast, Any
from dataclasses import dataclass, fields, Field

from matplotlib.widgets import Button, Slider
from PySide6 import QtCore, QtGui, QtWidgets

from icecream import ic

from .args import (
    arg_field,
    PartialParsedArgs,
    add_argument,
    add_arguments,
    parse_fields,
    parse_field,
    Action,
    MISSING_TYPE,
    ArgType,
)


class Option(ABC):
    def __init__(self, name, parent: QtWidgets.QWidget):
        self.name = name
        self.parent = parent

    @abstractmethod
    def get_parts(self) -> tuple[QtWidgets.QWidget, ...]:
        pass

    @abstractmethod
    def value(self) -> Any | None:
        pass

    @abstractmethod
    def set_value(self, value: Any):
        pass

    @abstractmethod
    def add_to_grid(
        self, grid: QtWidgets.QGridLayout, row: int, column: int
    ) -> tuple[int, int]:
        pass


class FolderOption(Option):
    def __init__(self, name, parent: QtWidgets.QWidget):
        super().__init__(name, parent)

        self.label = QtWidgets.QLabel(f"{self.name}:")

        self.entry = QtWidgets.QLineEdit()
        self.entry.setReadOnly(True)

        self.button = QtWidgets.QPushButton("Browse...")

        self.button.clicked.connect(self._browse)
        self.selected = False

    def get_parts(self) -> tuple[QtWidgets.QWidget, ...]:
        return self.label, self.entry, self.button

    def value(self):
        return Path(self.entry.text()) if self.selected else None

    def set_value(self, value):
        if value is not None:
            self.entry.setText(str(value))
            self.selected = len(str(value)) > 0

    def add_to_grid(
        self, grid: QtWidgets.QGridLayout, row: int, column: int
    ) -> tuple[int, int]:
        grid.addWidget(self.label, row, column)
        grid.addWidget(self.entry, row + 1, column, 1, 2)
        grid.addWidget(self.button, row + 1, column + 2, 1, 1)
        return (2, 3)

    def _browse(self):
        dialog = QtWidgets.QFileDialog(self.parent, self.name)
        dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        dialog.setOptions(QtWidgets.QFileDialog.ShowDirsOnly)

        if dialog.exec():
            selected = dialog.selectedFiles()[0]
            self.entry.setText(selected)
            self.selected = True

        self.parent.activateWindow()
        self.parent.raise_()


class ChoicesOption(Option):
    def __init__(self, name, choices: list, parent: QtWidgets.QWidget):
        super().__init__(name, parent)

        self.choices = choices

        self.label = QtWidgets.QLabel(f"{self.name}:")

        self.combo_box = QtWidgets.QComboBox()
        self.combo_box.addItems(choices)
        self.combo_box.setCurrentIndex(0)

    def get_parts(self) -> tuple[QtWidgets.QWidget, ...]:
        return self.label, self.combo_box

    def value(self):
        return self.choices[self.combo_box.currentIndex()]

    def set_value(self, value):
        if value is not None:
            inx = self.choices.index(value)
            self.combo_box.setCurrentIndex(inx)

    def add_to_grid(
        self, grid: QtWidgets.QGridLayout, row: int, column: int
    ) -> tuple[int, int]:
        grid.addWidget(self.label, row, column)
        grid.addWidget(self.combo_box, row, column + 1)
        return (1, 2)


class BoolOption(Option):
    def __init__(self, name, default: bool, parent: QtWidgets.QWidget):
        super().__init__(name, parent)

        self.check_box = QtWidgets.QCheckBox(f"{self.name}")
        self.check_box.setChecked(default)

    def get_parts(self) -> tuple[QtWidgets.QWidget, ...]:
        return (self.check_box,)

    def value(self):
        return self.check_box.isChecked()

    def set_value(self, value):
        self.check_box.setChecked(value)

    def add_to_grid(
        self, grid: QtWidgets.QGridLayout, row: int, column: int
    ) -> tuple[int, int]:
        grid.addWidget(self.check_box, row, column, 1, 2)
        return (1, 2)


def add_validator_for_type(typ: type, entry: QtWidgets.QLineEdit):
    if typ == float:
        validator = QtGui.QDoubleValidator(-1e12, 1e12, 12, entry)
        validator.setNotation(QtGui.QDoubleValidator.ScientificNotation)
        entry.setValidator(validator)
    elif typ == int:
        validator = QtGui.QIntValidator(entry)
        entry.setValidator(validator)
    elif typ == str:
        pass
    else:
        raise ValueError(f"Type {typ} not supported")


class InputOption(Option):
    def __init__(
        self, name, value_type: type, required: bool, parent: QtWidgets.QWidget
    ):
        super().__init__(name, parent)

        self.label = QtWidgets.QLabel(f"{self.name}:")
        self.entry = QtWidgets.QLineEdit()
        self.value_type = value_type
        self.required = required

        add_validator_for_type(value_type, self.entry)

    def get_parts(self) -> tuple[QtWidgets.QWidget, ...]:
        return (self.label, self.entry)

    def value(self):
        try:
            if self.value_type is float:
                return float(self.entry.text().strip())
            elif self.value_type is int:
                return int(self.entry.text().strip())
            else:
                return self.entry.text().strip()

        except ValueError:
            if self.required:
                raise
            else:
                return None

    def set_value(self, value):
        self.entry.setText(value)

    def add_to_grid(
        self, grid: QtWidgets.QGridLayout, row: int, column: int
    ) -> tuple[int, int]:
        grid.addWidget(self.label, row, column, 1, 1)
        grid.addWidget(self.entry, row, column + 1, 1, 1)
        return (1, 2)


class ItemWidget(QtWidgets.QWidget):
    def __init__(self, values: list, remove_callback):
        super().__init__()
        self.values = values
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(4, 2, 4, 2)
        self.labels = [QtWidgets.QLabel(text) for text in self.values]
        for label in self.labels:
            label.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
            )
            layout.addWidget(label)
        self.del_btn = QtWidgets.QPushButton("Delete")
        self.del_btn.setFixedWidth(70)
        layout.addWidget(self.del_btn)
        self.setLayout(layout)
        self.del_btn.clicked.connect(remove_callback)


class ItemsOption(Option):
    def __init__(
        self, name, types: list[type], expected_height: int, parent: QtWidgets.QWidget
    ):
        super().__init__(name, parent)

        self.types = types
        self.expected_height = expected_height
        self.input_layout = QtWidgets.QHBoxLayout()
        self.entries = [QtWidgets.QLineEdit() for v in self.types]
        for entry, t in zip(self.entries, self.types):
            entry.setPlaceholderText("Type an item and press + or Enter")
            add_validator_for_type(t, entry)
            self.input_layout.addWidget(entry)
        self.add_button = QtWidgets.QPushButton("+")
        self.add_button.setFixedWidth(30)
        self.input_layout.addWidget(self.add_button)

        self.list_widget = QtWidgets.QListWidget()

        self.add_button.clicked.connect(self._add_item_from_entries)

    def _add_item_from_entries(self):
        values = [entry.text().strip() for entry in self.entries]
        self.add_item(values, raise_on_invalid=False)

    def set_value(self, value):
        self.list_widget.clear()
        for item in value:
            self.add_item(item)

    def add_item(self, values: list, raise_on_invalid: bool = True):
        assert len(values) == len(self.types)
        valid = True
        for entry, value in zip(self.entries, values):
            if validator := entry.validator():
                valid &= (
                    validator.validate(value, 0)[0] == QtGui.QValidator.State.Acceptable
                )
        if not valid:
            if raise_on_invalid:
                raise ValueError("One of the values provided was invalid.")
            return
        item = QtWidgets.QListWidgetItem()

        def remove():
            row = self.list_widget.row(item)
            if row != -1:
                self.list_widget.takeItem(row)

        widget = ItemWidget(values, remove)
        item.setSizeHint(widget.sizeHint())
        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, widget)
        for entry in self.entries:
            entry.clear()
        self.entries[0].setFocus()

    def get_parts(self):
        return (
            self.input_layout,
            self.list_widget,
        )

    def value(self):
        values = []
        for ii in range(self.list_widget.count()):
            item = self.list_widget.item(ii)
            widget = self.list_widget.itemWidget(item)
            values.append(
                [t(v) for t, v in zip(self.types, cast(ItemWidget, widget).values)]
            )

        return values

    def add_to_grid(
        self, grid: QtWidgets.QGridLayout, row: int, column: int
    ) -> tuple[int, int]:
        grid.addLayout(self.input_layout, row, column, 1, len(self.types))
        grid.addWidget(
            self.list_widget, row + 1, column, self.expected_height, len(self.types)
        )
        return (self.expected_height + 1, len(self.types))


class MainWindow(QtWidgets.QWidget):
    def __init__(self, prog: str):
        super().__init__()
        self.setWindowTitle("Select In/Out Directories and Colormap")
        self.setMinimumSize(600, 140)

        self._layout = QtWidgets.QGridLayout(self)
        self._layout.setContentsMargins(12, 12, 12, 12)
        self._layout.setHorizontalSpacing(8)
        self._layout.setVerticalSpacing(8)

        self._actions: dict[str, tuple[Action, Option]] = dict()

        self.row = 0

        self.result_values: dict[str, Any] | None = None

    def add_argument(self, action: Action):
        name = action.get_display_name()

        assert action.dest not in self._actions

        match action.action:
            case "store":
                if action.choices is None:
                    if action.value_type is Path:
                        option = FolderOption(name, self)
                    else:
                        assert (
                            action.value_type is int
                            or action.value_type is float
                            or action.value_type is str
                        ), f"Type not recognised for store: {action.value_type}"
                        option = InputOption(
                            name, action.value_type, action.required, self
                        )
                else:
                    option = ChoicesOption(name, action.choices, self)
            case "store_true":
                option = BoolOption(name, False, self)
            case "store_false":
                if name.startswith("no "):
                    name = name[3:]
                option = BoolOption(name, True, self)
            case "append":
                types: list[type] = []
                assert type(action.value_type) is not MISSING_TYPE

                if action.nargs is not None and isinstance(action.nargs, int):
                    types = [action.value_type for _ in range(action.nargs)]
                else:
                    types = [action.value_type]
                option = ItemsOption(name, types, 3, self)
            case _:
                return

        self._actions[action.dest] = (action, option)
        size = option.add_to_grid(self._layout, self.row, 0)
        self.row += size[0]

    def set_value(self, name: str, value: Any):
        assert name in self._actions
        self._actions[name][1].set_value(value)

    def add_confirm_buttons(self):
        self.btn_ok = QtWidgets.QPushButton("OK")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_ok.clicked.connect(self.on_ok)
        self.btn_cancel.clicked.connect(self.close)

        btn_hbox = QtWidgets.QHBoxLayout()
        btn_hbox.addStretch()
        btn_hbox.addWidget(self.btn_ok)
        btn_hbox.addWidget(self.btn_cancel)
        self._layout.addLayout(btn_hbox, self.row, 0, 1, 3)

    def on_ok(self):

        self.result_values = dict()
        missing_required_actions = []
        unparsable_values = []
        for action, option in self._actions.values():
            try:
                value = option.value()
                if value is None and action.required:
                    missing_required_actions.append(action)
                self.result_values[action.dest] = value
            except ValueError:
                unparsable_values.append(action)

        if len(unparsable_values) != 0:
            unparsable_action_names = ", ".join(
                [a.get_display_name() for a in missing_required_actions]
            )

            QtWidgets.QMessageBox.warning(
                self,
                "Invalid",
                f"Could not parse values for {unparsable_action_names}.",
            )
            self.result_values = None
            return
        if len(missing_required_actions) != 0:
            missing_action_names = ", ".join(
                [a.get_display_name() for a in missing_required_actions]
            )

            QtWidgets.QMessageBox.warning(
                self, "Missing", f"Please give values for {missing_action_names}."
            )
            self.result_values = None
            return

        self.close()

    def launch(self, app: QtWidgets.QApplication):
        self.show()
        self.raise_()
        self.activateWindow()
        app.exec()
        return self.result_values


class InteractiveBase:
    @classmethod
    def parse_interactive(cls, prog: str, exclude: list[str] = [], args=None):
        args = args if args is not None else sys.argv[1:]
        interactive_parser = argparse.ArgumentParser(
            "interactive_parser", add_help=False
        )
        for f in fields(cls):
            if f.name == "interactive":
                action = parse_field(f)
                assert action is not None
                add_argument(interactive_parser, action)
                break
        else:
            raise ValueError("Expected to find and interactive field on the class.")
        interactive_args, remaining_args = interactive_parser.parse_known_args(args)

        exclude.append("interactive")
        actions = [a for a in parse_fields(cls) if a.dest not in exclude]

        if interactive_args.interactive:
            parser = argparse.ArgumentParser(prog=prog)

            app = QtWidgets.QApplication(sys.argv)
            window = MainWindow(prog)

            for a in actions:
                window.add_argument(a)
                not_required_a = copy.copy(a)
                not_required_a.required = False
                add_argument(parser, not_required_a)

            final_args = parser.parse_args(remaining_args)

            for k, v in vars(final_args).items():
                window.set_value(k, v)

            window.add_confirm_buttons()

            final_args = window.launch(app)
            if final_args is None:
                print("Canceled")
                exit()
            final_args["interactive"] = True

        else:
            parser = argparse.ArgumentParser(prog=prog)
            add_arguments(parser, actions)
            final_args = vars(parser.parse_args(remaining_args))
            final_args["interactive"] = False
        return cls(**final_args)


@dataclass
class InteractiveArgs(InteractiveBase):
    interactive: Path = arg_field(
        "--int",
        doc="If present will present the arguments interactively, instead of on the command line.",
        required=False,
        action="store_true",
    )


@dataclass
class NoInteractiveArgs(InteractiveBase):
    interactive: Path = arg_field(
        "--no-int",
        "--not-interactive",
        "--cli",
        "--no-interactive",
        arg_type=ArgType.EXPLICIT_ONLY,
        doc="If present will present the arguments on the console, instead of interactively.",
        required=False,
        action="store_false",
    )
