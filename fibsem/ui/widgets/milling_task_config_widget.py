import copy
from typing import Optional

import napari
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from superqt import QCollapsible, QIconifyIcon

from fibsem import utils, constants
from fibsem.microscope import FibsemMicroscope
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.ui.widgets.milling_alignment_widget import FibsemMillingAlignmentWidget
from fibsem.ui.widgets.milling_stage_editor_widget import FibsemMillingStageEditorWidget
from fibsem.ui.widgets.milling_task_acquisition_settings_widget import (
    FibsemMillingTaskAcquisitionSettingsWidget,
)
from fibsem.ui.widgets.milling_widget import FibsemMillingWidget2

# GUI Configuration Constants
WIDGET_CONFIG = {
    "name": {"default": "Milling Task", "placeholder": "Enter task name..."},
    "field_of_view": {
        "range": (0.001, 10000),
        "decimals": 1,
        "step": 5.0,
        "default": 150.0,
        "suffix": " μm",
        "tooltip": "Field of view in micrometers (μm)",
        "keyboard_tracking": False,
    },
}

# TODO: add options checkboxes, show advanced, show milling patterns, etc

class MillingTaskConfigWidget(QWidget):
    """Widget for editing FibsemMillingTaskConfig settings.

    Contains basic task settings (name, field of view),
    alignment settings, and acquisition settings. Stages and
    channel are ignored as requested.
    """

    settings_changed = pyqtSignal(FibsemMillingTaskConfig)
    milling_progress_signal = pyqtSignal(dict)

    def __init__(self, microscope: FibsemMicroscope, 
                 milling_task_config: Optional[FibsemMillingTaskConfig] = None,
                 milling_enabled: bool = True,
                 parent: Optional[QWidget] = None):
        """Initialize the MillingTaskConfig widget.

        Args:
            parent: Parent widget
            show_advanced: Whether to show advanced settings in sub-widgets
        """
        super().__init__(parent)
        self.microscope = microscope
        self._show_advanced = False
        self._milling_enabled = milling_enabled
        self._settings = milling_task_config or FibsemMillingTaskConfig()
        self._setup_ui()
        self.update_from_settings(self._settings)
        self._connect_signals()

    def _setup_ui(self):
        """Create and configure all UI elements.

        Creates a scroll area containing a vertical layout with basic settings at the top,
        followed by alignment and acquisition settings in group boxes.
        """
        
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area)
        
        # Create content widget that will be scrolled
        content_widget = QWidget()
        layout = QVBoxLayout()
        scroll_area.setContentsMargins(0, 0, 0, 0)
        layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        content_widget.setLayout(layout)
        content_widget.setContentsMargins(0, 0, 0, 0)
        scroll_area.setWidget(content_widget)

        # EXTRACT INTO SEPARATE WIDGET

        # Basic settings group
        self.basic_content = QWidget()
        basic_layout = QGridLayout()
        basic_layout.setContentsMargins(0, 0, 0, 0)
        self.basic_content.setLayout(basic_layout)

        # Task name
        basic_layout.addWidget(QLabel("Name"), 0, 0)
        self.name_edit = QLineEdit()
        name_config = WIDGET_CONFIG["name"]
        self.name_edit.setText(name_config["default"])
        self.name_edit.setPlaceholderText(name_config["placeholder"])
        basic_layout.addWidget(self.name_edit, 0, 1)

        # Field of view
        basic_layout.addWidget(QLabel("Field of View"), 1, 0)
        self.field_of_view_spinbox = QDoubleSpinBox()
        fov_config = WIDGET_CONFIG["field_of_view"]
        self.field_of_view_spinbox.setRange(*fov_config["range"])
        self.field_of_view_spinbox.setDecimals(fov_config["decimals"])
        self.field_of_view_spinbox.setSingleStep(fov_config["step"])
        self.field_of_view_spinbox.setValue(fov_config["default"])
        self.field_of_view_spinbox.setSuffix(fov_config["suffix"])
        self.field_of_view_spinbox.setToolTip(fov_config["tooltip"])
        self.field_of_view_spinbox.setKeyboardTracking(fov_config["keyboard_tracking"])
        basic_layout.addWidget(self.field_of_view_spinbox, 1, 1)

        self.task_group = QCollapsible("Milling Task", self)
        self.task_group.addWidget(self.basic_content)
        self.basic_content.setContentsMargins(0, 0, 0, 0)
        self.task_group.setContentsMargins(0, 0, 0, 0)

        ############

        # Alignment settings group
        self.alignment_widget = FibsemMillingAlignmentWidget(
            show_advanced=self._show_advanced
        )
        self.alignment_widget.setContentsMargins(0, 0, 0, 0)
        self.alignment_group = QCollapsible("Alignment", self)
        self.alignment_group.addWidget(self.alignment_widget)
        self.alignment_group.setContentsMargins(0, 0, 0, 0)

        # Acquisition settings group
        self.acquisition_widget = FibsemMillingTaskAcquisitionSettingsWidget(
            show_advanced=self._show_advanced
        )
        self.acquisition_widget.setContentsMargins(0, 0, 0, 0)
        self.acquisition_group = QCollapsible("Acquisition", self)
        self.acquisition_group.addWidget(self.acquisition_widget)
        self.acquisition_group.setContentsMargins(0, 0, 0, 0)

        # Milling stages editor
        self.milling_editor_widget = FibsemMillingStageEditorWidget(
            viewer=napari.current_viewer(),
            microscope=self.microscope,
            milling_stages=[],
            parent=self
        )
        self.milling_editor_widget.setContentsMargins(0, 0, 0, 0)
        self.miling_group = QCollapsible("Milling Stages", self)
        self.miling_group.addWidget(self.milling_editor_widget)
        self.miling_group.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self.task_group)           # type: ignore
        layout.addWidget(self.alignment_group)      # type: ignore
        layout.addWidget(self.acquisition_group)    # type: ignore
        layout.addWidget(self.miling_group)         # type: ignore

        # NOTES: shouldn't know anything about viewer, or microscope
        # only need microscope to get 'dynamic' values such as milling current
        # should make this 'optional' and pass in from parent
        # viewer is used to display the milling stages
        # could we do this at a higher level and just subscribe to the settings_changed signal?
        layout.addStretch()

        self.milling_widget = FibsemMillingWidget2(
            microscope=self.microscope,
            parent=self
        )
        layout.addWidget(self.milling_widget)
        self.milling_widget.setVisible(self._milling_enabled)
        # layout.addStretch()

    def _connect_signals(self):
        """Connect widget signals to their respective handlers.

        Connects all basic controls and sub-widget signals to emit
        settings change notifications when any value changes.
        """
        self.name_edit.textChanged.connect(self._emit_settings_changed)
        self.field_of_view_spinbox.valueChanged.connect(self._emit_settings_changed)
        self.alignment_widget.settings_changed.connect(self._emit_settings_changed)
        self.acquisition_widget.settings_changed.connect(self._emit_settings_changed)
        self.milling_editor_widget._milling_stages_updated.connect(self._emit_settings_changed)

    def _emit_settings_changed(self):
        """Emit the settings_changed signal with current settings.

        Called whenever any control value changes to notify listeners
        of the updated FibsemMillingTaskConfig settings.
        """
        settings = self.get_settings()
        self.settings_changed.emit(settings)

    def get_settings(self) -> FibsemMillingTaskConfig:
        """Get the current FibsemMillingTaskConfig from the widget values.

        Returns:
            FibsemMillingTaskConfig object with values from the UI controls.
        """
        self._settings.name = self.name_edit.text()
        self._settings.field_of_view = self.field_of_view_spinbox.value() * constants.MICRO_TO_SI
        self._settings.alignment = self.alignment_widget.get_settings()
        self._settings.acquisition = self.acquisition_widget.get_settings()
        self._settings.stages = self.milling_editor_widget.get_milling_stages()

        return self._settings

    def update_from_settings(self, settings: FibsemMillingTaskConfig):
        """Update all widget values from a FibsemMillingTaskConfig object.

        Args:
            settings: FibsemMillingTaskConfig object to load values from.
                     All basic controls and sub-widgets are updated from the settings.
        """
        # Block signals to prevent recursive updates
        self.blockSignals(True)

        self.name_edit.setText(settings.name)
        self.field_of_view_spinbox.setValue(settings.field_of_view * constants.SI_TO_MICRO)
        # Update sub-widgets
        self.alignment_widget.update_from_settings(settings.alignment)
        self.acquisition_widget.update_from_settings(settings.acquisition)
        self.milling_editor_widget.update_from_settings(settings.stages)

        self._settings = copy.deepcopy(settings)
        self.blockSignals(False)

    def set_show_advanced(self, show_advanced: bool):
        """Set the visibility of advanced settings in sub-widgets.

        Args:
            show_advanced: True to show advanced settings, False to hide them
        """
        self._show_advanced = show_advanced
        self.alignment_widget.set_show_advanced(show_advanced)
        self.acquisition_widget.set_show_advanced(show_advanced)
        self.milling_editor_widget.set_show_advanced(show_advanced)

    def toggle_advanced(self):
        """Toggle the visibility of advanced settings in sub-widgets.

        Switches between showing and hiding the advanced controls in
        alignment and acquisition widgets.
        """
        self.set_show_advanced(not self._show_advanced)

    def get_show_advanced(self) -> bool:
        """Get the current advanced settings visibility state.

        Returns:
            True if advanced settings are currently visible, False otherwise
        """
        return self._show_advanced

    def set_background_milling_stages(self, background_stages):
        """Set background milling stages to be displayed in the milling stage editor.
        
        Args:
            background_stages: List of FibsemMillingStage objects to show as background
        """
        print(f"Setting background milling stages: {[bs.name for bs in background_stages]}")
        self.milling_editor_widget.set_background_milling_stages(background_stages)



if __name__ == "__main__":

    import os
    from pathlib import Path

    import napari
    from PyQt5.QtWidgets import QTabWidget, QWidget

    from fibsem.applications.autolamella.structures import AutoLamellaProtocol, Experiment

    viewer = napari.Viewer()
    main_widget = QTabWidget()

    # set tab to side
    qwidget = QWidget()
    icon1 = QIconifyIcon("material-symbols:experiment", color="white")
    main_widget.addTab(qwidget, icon1, "Experiment") # type: ignore
    layout = QVBoxLayout()
    qwidget.setLayout(layout)
    qwidget.setContentsMargins(0, 0, 0, 0)
    layout.setContentsMargins(0, 0, 0, 0)

    microscope, settings = utils.setup_session()

    BASE_PATH = "/home/patrick/github/autolamella/autolamella/log/AutoLamella-2025-05-28-17-22/"
    EXPERIMENT_PATH = Path(os.path.join(BASE_PATH, "experiment.yaml"))
    PROTOCOL_PATH = Path(os.path.join(BASE_PATH, "protocol.yaml"))
    exp = Experiment.load(EXPERIMENT_PATH)
    protocol = AutoLamellaProtocol.load(PROTOCOL_PATH)

    milling_task_config = FibsemMillingTaskConfig.from_stages(
        stages=exp.positions[0].milling_workflows["mill_rough"],  # type: ignore
    )

    # Create the MillingTaskConfig widget
    config_widget = MillingTaskConfigWidget(microscope=microscope)
    layout.addWidget(config_widget)
    config_widget.update_from_settings(milling_task_config)

    # Connect to settings change signal
    def on_task_config_changed(task_config: FibsemMillingTaskConfig):
        print(f"Task Config changed: {utils.current_timestamp_v3(timeonly=False)}")
        print(f"  name: {task_config.name}")
        print(f"  field_of_view: {task_config.field_of_view}")
        print(f"  alignment: {task_config.alignment}")
        print(f"  acquisition: {task_config.acquisition}")
        print(f"  stages: {len(task_config.stages)} stages")
        for stage in task_config.stages:
            print(f"    Stage Name: {stage.name}")
            print(f"    Milling: {stage.milling}")
            print(f"    Pattern: {stage.pattern}")
            print(f"    Strategy: {stage.strategy.config}")
            print("---------------------"*3)

    config_widget.settings_changed.connect(on_task_config_changed)
    main_widget.setWindowTitle("MillingTaskConfig Widget Test")

    viewer.window.add_dock_widget(main_widget, add_vertical_stretch=False, area='right')

    napari.run()
